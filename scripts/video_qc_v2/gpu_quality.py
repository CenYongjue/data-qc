"""实验：显存NV12 -> 批量GPU像素指标，仅头部灰度回CPU做LK。

PyNvVideoCodec + DLPack不把原图搬到CPU；灰度、缩放、Sobel/Laplacian、曝光、
相邻帧差在Torch CUDA上批量计算。CPU复用正式阈值和时序判定。
GPU双线性缩放与swscale存在舍入差异，启动时记录并限制差异，不宣称逐像素等价。
"""
from fractions import Fraction
from multiprocessing import shared_memory
from types import SimpleNamespace
import itertools
import time
import av
import cv2
import numpy as np
from video_qc_v2.persistent_decode import arrays,ATTACHED


class StaticFilter:
    def __init__(self,rows):self.rows=rows
    def finish(self):return self.rows


class GPUDecoder:
    def __init__(self,path,device,stats):
        import torch
        import torch.nn.functional as F
        import PyNvVideoCodec as nvc
        self.torch=torch;self.F=F;self.nvc=nvc;self.path=path;self.device=device
        torch.cuda.set_device(device);torch.set_num_threads(1)
        before=time.perf_counter()
        with av.open(path) as c:
            s=c.streams.video[0];self.width=s.width;self.height=s.height;self.fps=float(s.average_rate);self.tb=s.time_base
            self.duration=s.duration;self.start=s.start_time
        # 使用官方高层seek接口负责关键帧定位/解码器同步，避免手工Seek包生命周期问题。
        self.demux=None
        self.decoder=nvc.SimpleDecoder(path,gpu_id=device,cuda_stream=torch.cuda.current_stream().cuda_stream,
            use_device_memory=True,output_color_type=nvc.OutputColorType.NATIVE)
        self.it=None;self.pending=None;self.checked=False;self.last_end=None
        self.kernel=torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]],
                                  [[-1,-2,-1],[0,0,0],[1,2,1]],
                                  [[0,1,0],[1,-4,1],[0,1,0]]],device=device,dtype=torch.float32)[:,None]
        stats['decoder_opens']+=1;stats['setup_seconds']+=time.perf_counter()-before

    def iterator(self,start,stats):
        if start>0:self.decoder.seek_to_index(round(start*self.fps));stats['seeks']+=1
        while True:
            batch=self.decoder.get_batch_frames(32)
            if not batch:break
            for frame in batch:yield frame

    def resize(self,frames,width):
        torch=self.torch
        # NV12的Y平面为TV范围，保持与正式gray8转换相同的0..255意义。
        raw=torch.stack(frames).float()[:,None]
        height=max(16,round(self.height*width/self.width))
        gray=self.F.interpolate(raw,size=(height,width),mode='bilinear',align_corners=False,antialias=True)
        return ((gray[:,0]-16)*(255/219)).round().clamp_(0,255).to(torch.uint8)

    def check_first(self,pts,gray,stats):
        from video_qc_v2.worker import normalized_gray
        torch=self.torch;absolute=float(pts*self.tb)
        with av.open(self.path) as c:
            s=c.streams.video[0];s.codec_context.thread_count=2
            if absolute>0:c.seek(max(0,int(absolute/s.time_base)),stream=s,backward=True)
            for f in c.decode(s):
                if f.pts is None or float(f.pts*s.time_base)<absolute-1e-6:continue
                if abs(float(f.pts*s.time_base)-absolute)>1e-5:raise RuntimeError('GPU/CPU首帧PTS不一致')
                ref=normalized_gray(f,gray.shape[1]);actual=gray.cpu().numpy();mae=float(np.abs(ref.astype(float)-actual).mean())
                def measure(a):
                    x=cv2.Sobel(a,cv2.CV_32F,1,0);y=cv2.Sobel(a,cv2.CV_32F,0,1);mask=cv2.magnitude(x,y)>20
                    return float(mask.mean()),float(np.abs(cv2.Laplacian(a,cv2.CV_32F))[mask].mean()) if mask.any() else 0.
                re,rs=measure(ref);ge,gs=measure(actual)
                record={'file':self.path,'pts':int(pts),'gray_mae':mae,'gray_mean_cpu':float(ref.mean()),'gray_mean_gpu':float(actual.mean()),'edge_fraction_cpu':re,'edge_fraction_gpu':ge,
                        'sharpness_cpu':rs,'sharpness_gpu':gs,'resize':'CUDA bilinear antialias; approximate swscale'}
                stats['validation'].append(record)
                if mae>1.5 or abs(re-ge)>.04 or abs(rs-gs)>max(2,rs*.2):raise RuntimeError('GPU预处理与原指标差异过大: '+str(record))
                self.checked=True;return
        raise RuntimeError('没有找到CPU参考帧')

    def get(self,camera,config,stats):
        torch=self.torch;start=camera['start_seconds'];end=camera['end_seconds']
        if self.it is None:self.it=self.iterator(start,stats)
        gray_batches=[];batch=[];pts=[]
        def flush():
            if batch:
                before=time.perf_counter();g=self.resize(batch,config['analysis_width']);gray_batches.append(g)
                torch.cuda.synchronize();stats['prepare_and_shm_seconds']+=time.perf_counter()-before;batch.clear()
        while True:
            before=time.perf_counter()
            if self.pending is not None:f=self.pending;self.pending=None
            else:
                try:f=next(self.it)
                except StopIteration:break
                stats['decoded_frames']+=1
            stats['decode_transfer_seconds']+=time.perf_counter()-before
            value=int(f.getPTS())
            if value==np.iinfo(np.int64).min:raise RuntimeError('GPU解码输出缺少PTS')
            absolute=float(value*self.tb)
            if absolute<start-1e-6:continue
            if absolute>=end-1e-6:self.pending=f;break
            # NVDEC输出缓冲可能在后续Decode时复用。先在显存中持有独立副本，
            # 不把裸DecodedFrame引用积攒到下一批再读，避免“PTS正确但像素已覆盖”。
            pts.append(value);batch.append(torch.from_dlpack(f)[:self.height,:self.width].clone())
            if len(batch)==32:flush()
        flush()
        if not pts:raise RuntimeError('无GPU解码帧')
        g=torch.cat(gray_batches);del gray_batches
        validate=not self.checked
        if validate:self.check_first(pts[0],g[0],stats)
        before=time.perf_counter();values=g.float();n,h,w=g.shape;times=np.asarray(pts)*float(self.tb)-start
        mean=values.mean((1,2));diff=(values[1:]-values[:-1]).abs().mean((1,2))
        mad=torch.cat([torch.zeros(1,device=values.device),diff]);identical=torch.cat([torch.zeros(1,device=values.device,dtype=torch.bool),diff==0])
        timeline_values=torch.stack([mean,mad,identical.float()],1).cpu().numpy()
        selected=[];next_sample=0.
        for i,t in enumerate(times):
            if t+1e-6>=next_sample:selected.append(i);next_sample=(np.floor(t*config['sample_hz']+1e-6)+1)/config['sample_hz']
        ix=torch.tensor(selected,device=values.device);s=values[ix]
        filtered=self.F.conv2d(self.F.pad(s[:,None],(1,1,1,1),mode='reflect'),self.kernel)
        edges=torch.sqrt(filtered[:,0].square()+filtered[:,1].square())>config['edge_gradient']
        edge_fraction=edges.float().mean((1,2));sharp=(filtered[:,2].abs()*edges).sum((1,2))/edges.sum((1,2)).clamp_min(1)
        dark=(s<config['dark_pixel']).float().mean((1,2));bright=(s>config['bright_pixel']).float().mean((1,2))
        tile=s.reshape(len(selected),4,h//4,4,w//4).mean((2,4))
        dark_tile=(tile<config['dark_mean']).float().mean((1,2));bright_tile=(tile>config['bright_mean']).float().mean((1,2))
        sample_stats=torch.stack([s.mean((1,2)),edge_fraction,sharp,dark,bright,dark_tile,bright_tile],1).cpu().numpy()
        if validate:
            r=stats['validation'][-1];r['edge_fraction_gpu_ops']=float(sample_stats[0,1]);r['sharpness_gpu_ops']=float(sample_stats[0,2])
            if abs(r['edge_fraction_gpu_ops']-r['edge_fraction_gpu'])>1e-5 or abs(r['sharpness_gpu_ops']-r['sharpness_gpu'])>.001:
                raise RuntimeError('GPU Sobel/Laplacian与同图OpenCV结果不一致: '+str(r))
        sad=(s[1:]-s[:-1]).abs().mean((1,2))*100/256
        scene=torch.cat([torch.zeros(1,device=s.device),torch.minimum(sad,(sad-torch.cat([torch.zeros(1,device=s.device),sad[:-1]])).abs())]).cpu().numpy()
        samples=[];signals=[]
        for j,i in enumerate(selected):
            avg,edge,sh,d,b,dt,bt=map(float,sample_stats[j]);low=edge<config['min_edge_fraction'];sh=None if low else sh
            samples.append({'seconds':float(times[i]),'frame_index':i,'brightness':avg,'edge_fraction':edge,'sharpness':sh,
                'dark_fraction':d,'bright_fraction':b,'dark_tile_fraction':dt,'bright_tile_fraction':bt,
                'low_texture':low,'blur':sh is not None and sh<config['blur_score'],'block_tile_scores':[],
                'block_score':None,'hand_count':None,'hand_boxes':[]})
            signals.append({'lavfi.signalstats.YAVG':avg,'lavfi.scd.score':round(float(scene[j]),3)})
        # FFmpeg freezedetect与参考帧比较，不以相邻帧相同直接代替。
        frozen_rows=[];reference=0;already_frozen=False;noise=config['freeze_noise']*256
        for i in range(1,n):
            distance=float(timeline_values[i,1]) if reference==i-1 else float((values[i]-values[reference]).abs().mean().item())
            frozen=distance<=noise;duration=times[i]-times[reference]
            if duration+1e-6>=config['freeze_seconds']:
                if not already_frozen:frozen_rows.append({'lavfi.freezedetect.freeze_start':float(times[reference]+start)})
                if not frozen:frozen_rows.append({'lavfi.freezedetect.freeze_end':float(times[i]+start)})
                already_frozen=frozen
            if not frozen:reference=i
        timeline=[{'seconds':float(t),'mean':float(v[0]),'mad':float(v[1]),'identical':bool(v[2])} for t,v in zip(times,timeline_values)]
        desc={'gpu_quality':True,'count':n,'source_width':self.width,'source_height':self.height,'fps':self.fps,
              'time_base':[self.tb.numerator,self.tb.denominator],'pts':pts,'device':self.device,'hardware_accelerated':True,
              'timeline':timeline,'samples':samples,'signals':signals,'freeze':frozen_rows,'arrays':{},
              'normalization':'NV12 TV Y -> CUDA antialiased bilinear gray8; approximate swscale',
              'corruption_note':'NVDEC错误由异常报告；接口不提供逐帧is_corrupt标记'}
        # 只有头部的低分辨率图像返回CPU；夹爪只返回标量/时间戳。
        if camera.get('motion_check',True):
            cpu=g.cpu().numpy();handle=shared_memory.SharedMemory(create=True,size=cpu.nbytes)
            np.ndarray(cpu.shape,dtype='uint8',buffer=handle.buf)[:]=cpu;desc['name']=handle.name;desc['bytes']=cpu.nbytes
            desc['arrays']={'gray':{'shape':list(cpu.shape),'dtype':'uint8','offset':0}};handle.close()
        else:
            handle=shared_memory.SharedMemory(create=True,size=1);desc['name']=handle.name;desc['bytes']=1;handle.close()
        torch.cuda.synchronize();stats['prepare_and_shm_seconds']+=time.perf_counter()-before;self.last_end=end
        return desc

    def close(self):self.it=None;self.pending=None;self.decoder=None;self.demux=None


def scan_precomputed(record,key,camera,expected,report,c,work):
    import tempfile
    from video_qc_v2.worker import finalize_camera,reason
    from video_qc_v2.operators import CameraMotion
    d=camera['_predecoded'];before=time.perf_counter();timing={k:0. for k in ['decoder_setup','decode','gray','temporal','lk','signalstats','blockdetect','cache','hands_submit_wait']}
    result={'samples':d['samples'],'timeline':d['timeline'],'frame_count':d['count'],'fps':d['fps'],
            'duration_seconds':camera['end_seconds']-camera['start_seconds'],'source_size':[d['source_width'],d['source_height']],
            'codec':'h264','normalization':d['normalization'],'decoder':{'requested':'nvdec_gpu_pixels',
            'hardware_accelerated':True,'device':d['device'],'zero_copy_input':True,'pixel_metrics_on_gpu':True,
            'per_frame_corrupt_flag_available':False},'_cache':tempfile.SpooledTemporaryFile(mode='w+b',dir=work),'_cache_shape':None}
    pts=np.asarray(d['pts'],dtype=np.int64);times=pts*float(Fraction(*d['time_base']))-camera['start_seconds'];errors=int(np.sum(np.diff(pts)<=0))
    gaps=np.flatnonzero(np.diff(times)>1.5/d['fps'])
    if len(gaps):result['gaps']=[{'start_seconds':float(times[i]),'end_seconds':float(times[i+1]),'duration_seconds':float(times[i+1]-times[i])} for i in gaps];reason(report,'drop_frames','FAIL','CFR时间轴存在缺口',key,result['gaps'])
    if d['count']!=camera['expected_frames']:reason(report,'count_mismatch','FAIL',f'decoded={d["count"]}, expected={camera["expected_frames"]}',key)
    if abs(d['count']/d['fps']-result['duration_seconds'])>1.1/d['fps']:reason(report,'count_mismatch','FAIL','解码时长与声明区间不一致',key)
    motion=CameraMotion(c['motion']) if camera.get('motion_check',True) else None
    if motion is not None:
        handle=shared_memory.SharedMemory(name=d['name']);ATTACHED.append(handle);gray=arrays(handle,d)['gray']
        before=time.perf_counter()
        for image,point in zip(gray,result['timeline']):
            if point['mad']>c['scene_change_mad']:motion.previous=None;motion.segment+=1;motion.global_matrix=np.eye(3);motion.segment_start=point['seconds']
            motion.push(image,point['seconds'])
        timing['lk']+=time.perf_counter()-before
    return finalize_camera(result,report,camera,key,c,motion,StaticFilter(d['signals']),StaticFilter(d['freeze']),errors,float(times[-1]),timing)
