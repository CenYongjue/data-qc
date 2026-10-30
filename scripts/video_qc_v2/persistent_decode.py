"""持久NVDEC生产者与共享内存读取器。

每GPU一个进程，保存六路解码器及下一回合首帧，连续回合不重新seek/open。
全帧只传320宽灰度，4Hz额外传同算法生成的YUV444，避免经队列复制原分辨率视频。
队列仅传共享内存描述符；总在途回合由信号量限制。回合的质检状态仍各自独立。
"""
from fractions import Fraction
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
import copy
import time
import traceback
import av
from av.codec.hwaccel import HWAccel
import cv2
import numpy as np

ATTACHED=[]


def arrays(handle,desc):
    return {k:np.ndarray(tuple(v['shape']),dtype=v['dtype'],buffer=handle.buf,offset=v['offset']) for k,v in desc['arrays'].items()}


def allocate(capacity,colors,height,width):
    layout=[('pts',(capacity,),'int64'),('corrupt',(capacity,),'uint8'),('color_index',(capacity,),'int32'),
            ('gray',(capacity,height,width),'uint8'),('color',(colors,3,height,width),'uint8')]
    desc={'arrays':{}};offset=0
    for name,shape,dtype in layout:
        offset=(offset+7)//8*8;desc['arrays'][name]={'shape':shape,'dtype':dtype,'offset':offset}
        offset+=int(np.prod(shape))*np.dtype(dtype).itemsize
    handle=shared_memory.SharedMemory(create=True,size=offset);desc['name']=handle.name;desc['bytes']=offset
    return handle,desc,arrays(handle,desc)


def cleanup(record):
    for camera in record['cameras'].values():
        d=camera.get('_predecoded')
        if not d:continue
        try:
            h=shared_memory.SharedMemory(name=d['name']);h.close();h.unlink()
        except FileNotFoundError:pass


class Decoder:
    def __init__(self,path,device,stats):
        before=time.perf_counter();self.c=av.open(path,hwaccel=HWAccel('cuda',device=str(device),allow_software_fallback=False))
        self.s=self.c.streams.video[0];self.s.codec_context.thread_count=2;self.s.codec_context.options={'err_detect':'explode'}
        if not self.s.codec_context.is_hwaccel:raise RuntimeError('拒绝CPU回退')
        self.path=path;self.device=device;self.it=iter(self.c.decode(self.s));self.pending=None;self.end=None;self.checked=False
        stats['decoder_opens']+=1;stats['setup_seconds']+=time.perf_counter()-before

    def get(self,camera,config,stats):
        from video_qc_v2.worker import normalized_gray
        start=camera['start_seconds'];end=camera['end_seconds'];fps=float(self.s.average_rate)
        h=max(16,round(self.s.height*config['analysis_width']/self.s.width));w=config['analysis_width']
        cap=camera['expected_frames']+64;cp=int(np.ceil((end-start)*config['sample_hz']))+8
        handle,desc,a=allocate(cap,cp,h,w);a['color_index'].fill(-1)
        desc.update(count=0,color_count=0,source_width=self.s.width,source_height=self.s.height,fps=fps,
                    time_base=[self.s.time_base.numerator,self.s.time_base.denominator],stream_duration=self.s.duration,
                    start_time=self.s.start_time,codec=self.s.codec_context.name,device=self.device,hardware_accelerated=True)
        try:
            if self.end is None or abs(start-self.end)>1e-5:
                if start>0:self.c.seek(max(0,int(start/self.s.time_base)),stream=self.s,backward=True);stats['seeks']+=1
                self.it=iter(self.c.decode(self.s));self.pending=None
            n=nc=0;next_sample=0.
            while True:
                before=time.perf_counter()
                if self.pending is not None:f=self.pending;self.pending=None
                else:
                    try:f=next(self.it)
                    except StopIteration:break
                stats['decode_transfer_seconds']+=time.perf_counter()-before;stats['decoded_frames']+=1
                absolute=float(f.pts*f.time_base) if f.pts is not None else None
                if absolute is not None and absolute<start-1e-6:continue
                if absolute is not None and absolute>=end-1e-6:self.pending=f;break
                if n>=cap:raise RuntimeError('帧数超过元数据与安全缓冲；不能静默截断')
                before=time.perf_counter();gray=normalized_gray(f,w)
                a['gray'][n]=gray;a['pts'][n]=f.pts if f.pts is not None else np.iinfo(np.int64).min
                a['corrupt'][n]=bool(getattr(f,'is_corrupt',False))
                t=absolute-start if absolute is not None else None
                if t is not None and t+1e-6>=next_sample:
                    next_sample=(np.floor(t*config['sample_hz']+1e-6)+1)/config['sample_hz']
                    if nc>=cp:raise RuntimeError('采样数量超过元数据与安全缓冲')
                    color=f.reformat(width=w,height=h,format='yuvj444p')
                    a['color'][nc]=color.to_ndarray();a['color_index'][n]=nc;nc+=1
                stats['prepare_and_shm_seconds']+=time.perf_counter()-before
                if not self.checked and f.pts is not None:
                    # 每个持久视频解码器只抽查一帧；比较同PTS，不逐帧重复验证。
                    with av.open(self.path) as ref:
                        s=ref.streams.video[0];s.codec_context.thread_count=2
                        if absolute>0:ref.seek(max(0,int(absolute/s.time_base)),stream=s,backward=True)
                        for cpu in ref.decode(s):
                            if cpu.pts is not None and abs(float(cpu.pts*s.time_base)-absolute)<1e-6:
                                delta=np.abs(normalized_gray(cpu,w).astype(float)-gray)
                                mae=float(delta.mean())
                                if mae>1:raise RuntimeError(f'CPU/GPU灰度差异过大: {mae}')
                                stats['validation'].append({'file':self.path,'pts':f.pts,'gray_mae':mae});self.checked=True;break
                            if cpu.pts is not None and float(cpu.pts*s.time_base)>absolute+1e-5:raise RuntimeError('CPU/GPU首帧PTS不一致')
                    if not self.checked:raise RuntimeError('未找到CPU参考帧')
                n+=1
            self.end=end;desc['count']=n;desc['color_count']=nc
            return desc
        except BaseException:
            handle.unlink();raise
        finally:handle.close()

    def close(self):self.c.close()


def producer(device,records,config,ready,status,credits,stop,deadline,decoder_class=None):
    cv2.setNumThreads(1);decoders={}
    stats={'device':device,'decoder_opens':0,'seeks':0,'decoded_frames':0,'setup_seconds':0.,
           'decode_transfer_seconds':0.,'prepare_and_shm_seconds':0.,'credit_wait_seconds':0.,'episodes':0,'validation':[]}
    try:
        for original in records:
            if stop.is_set() or time.perf_counter()>=deadline:break
            before=time.perf_counter()
            while not credits.acquire(timeout=1):
                if stop.is_set() or time.perf_counter()>=deadline:break
            else:
                stats['credit_wait_seconds']+=time.perf_counter()-before
                if stop.is_set() or time.perf_counter()>=deadline:credits.release();break
                record=copy.deepcopy(original)
                try:
                    from video_qc_v2.lerobot import check_record
                    try:check_record(record)
                    except (ValueError,OSError):
                        # 已明确坏文件只交CPU记录原因，不浪费GPU解码。
                        ready.put(record);stats['episodes']+=1;continue
                    for camera in record['cameras'].values():
                        path=camera['path']
                        if path not in decoders:decoders[path]=(decoder_class or Decoder)(path,device,stats)
                        camera['_predecoded']=decoders[path].get(camera,config,stats)
                    ready.put(record);stats['episodes']+=1
                except BaseException:cleanup(record);credits.release();raise
                continue
            break
    except BaseException:status.put({'type':'error','device':device,'traceback':traceback.format_exc()});stop.set()
    finally:
        for d in decoders.values():d.close()
        status.put({'type':'done','stats':stats})


class Frame:
    def __init__(self,owner,index):
        self.o=owner;self.i=index;v=int(owner.a['pts'][index]);self.pts=None if v==np.iinfo(np.int64).min else v
        self.time_base=owner.tb;self.width=owner.desc['source_width'];self.height=owner.desc['source_height'];self.is_corrupt=bool(owner.a['corrupt'][index])
    def reformat(self,width,height,format):
        if format=='gray':out=av.VideoFrame.from_ndarray(self.o.a['gray'][self.i],format='gray')
        elif format=='yuvj444p':
            j=int(self.o.a['color_index'][self.i])
            if j<0:raise RuntimeError('CPU与GPU的4Hz采样位置不一致')
            out=av.VideoFrame.from_ndarray(self.o.a['color'][j],format='yuvj444p')
        else:raise RuntimeError('持久解码只支持当前无块检测的机器人流水线')
        if out.width!=width or out.height!=height:raise RuntimeError('预处理尺寸与CPU规则不一致')
        out.pts=self.pts;out.time_base=self.time_base;return out


class Container:
    def __init__(self,desc):
        self.desc=desc;h=shared_memory.SharedMemory(name=desc['name']);ATTACHED.append(h);self.a=arrays(h,desc)
        self.tb=Fraction(*desc['time_base']);self.decode_device=desc['device']
        codec=SimpleNamespace(thread_count=2,options={},is_hwaccel=True,name=desc['codec'])
        stream=SimpleNamespace(codec_context=codec,average_rate=Fraction(str(desc['fps'])),time_base=self.tb,
                               start_time=desc['start_time'],duration=desc['stream_duration'],frames=desc['count'],
                               width=desc['source_width'],height=desc['source_height'])
        self.streams=SimpleNamespace(video=[stream]);self.duration=0
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def seek(self,*args,**kwargs):pass  # 生产者已按该回合边界定位，无需再次seek。
    def decode(self,stream):return (Frame(self,i) for i in range(self.desc['count']))


def cpu_init():cv2.setNumThreads(1)


def cpu_job(job):
    from video_quality_pipeline_v2 import job_wrapper
    try:return job_wrapper(job)
    finally:
        # 报告/复核全部完成后才关闭映射；父进程确认结果落盘后统一unlink。
        for handle in ATTACHED:handle.close()
        ATTACHED.clear()
