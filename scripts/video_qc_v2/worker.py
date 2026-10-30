"""单回合的分层检查。CPU完成所有前置检查后，才把缓存的4Hz帧交给GPU。

缓存使用SpooledTemporaryFile：短视频驻内存，超过预算自动落盘，回合结束立即回收。
这既保留每回合的阶段顺序，又允许“CPU检查回合B / GPU检测回合A”的流水。
"""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import av
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import egodex_video_qc as old
from video_qc_v2.operators import MetadataFilter,CameraMotion,intervals,spatial_block_scores

CLIENT=None
WORKER_SLOT=0
LABELS={'integrity':'文件/字段/数据块完整性异常','decode':'视频解码异常','timestamps':'时间戳异常',
        'drop_frames':'时间轴缺口/疑似丢帧','count_mismatch':'解码/标注帧数不一致','multi_camera':'多路时间范围或局部缺口异常',
        'freeze':'低变化/重复画面候选','freeze_jump':'冻结后出现大幅画面跳变','blockiness':'块状压缩失真候选',
        'color_corruption':'瞬时色块/花屏候选','flicker':'反复亮度闪烁候选','flash':'短时全局亮度尖峰候选','shake':'画面高频抖动候选',
        'motion_uncertain':'中央特征集中，镜头运动估计待查','motion_unscorable':'背景支撑不足，抖动无法可靠评分',
        'underexposed':'持续过暗/低可见性候选','overexposed':'持续过亮/裁剪候选','blur':'持续模糊候选',
        'low_texture':'低纹理，模糊判断不可靠','hands_missing':'双手长时间未检出候选','operator_error':'检测算子未正常执行'}


def reason(r,code,severity,detail,camera=None,spans=None,at=None):
    spans=spans or [];t=at if at is not None else ((spans[0]['start_seconds']+spans[0]['end_seconds'])/2 if spans else 0.)
    r['reasons'].append({'code':code,'label':LABELS[code],'severity':severity,'detail':detail,
                         'camera':camera,'intervals':spans,'at_seconds':float(t)})


def init_cpu(slots,requests,replies,imgsz):
    global CLIENT,WORKER_SLOT
    from egodex_yolo_pipeline import AsyncHands
    cv2.setNumThreads(1);slot=slots.get();WORKER_SLOT=slot;CLIENT=AsyncHands(slot,requests,replies[slot],imgsz)


def file_check(record,report):
    """只检查必要数据：文件可读、结构/序列长度、非有限值，不做人体运动质检。"""
    expected=None;hdf=record.get('hdf5')
    if record.get('lerobot'):
        try:
            from video_qc_v2.lerobot import check_record
            expected=check_record(record)
        except (OSError,ValueError,KeyError,TypeError) as exc:reason(report,'integrity','FAIL',f'LeRobot Parquet: {exc}')
    if hdf:
        try:
            with h5py.File(hdf,'r') as f:
                for name in ['camera/intrinsic','transforms/camera','transforms/leftHand','transforms/rightHand']:
                    if name not in f:raise ValueError('缺少 '+name)
                expected=len(f['transforms/camera'])
                if f['camera/intrinsic'].shape!=(3,3):raise ValueError('相机内参形状错误')
                def read(name,obj):
                    if not isinstance(obj,h5py.Dataset):return
                    a=np.asarray(obj[()])
                    if name.startswith('transforms/') and a.shape!=(expected,4,4):raise ValueError(name+'长度/形状错误')
                    if name.startswith('confidences/') and a.shape!=(expected,):raise ValueError(name+'长度错误')
                    if np.issubdtype(a.dtype,np.number) and not np.isfinite(a).all():raise ValueError(name+'含NaN/Inf')
                f.visititems(read)
        except (OSError,KeyError,ValueError,TypeError,RuntimeError) as exc:reason(report,'integrity','FAIL',f'HDF5: {exc}')
    for key,camera in record['cameras'].items():
        path=Path(camera['path'])
        if not path.is_file() or path.stat().st_size==0:reason(report,'integrity','FAIL',str(path),key)
    return expected


def normalized_gray(frame,width):
    # swscale转换到8bit灰度，范围统一为0..255；不直接混用TV范围Y值与OpenCV阈值。
    h=max(16,round(frame.height*width/frame.width))
    return frame.reformat(width=width,height=h,format='gray').to_ndarray()


def scan_camera(record,key,camera,expected,report,config,work):
    c=config;timing={k:0. for k in ['decoder_setup','decode','gray','temporal','lk','signalstats','blockdetect','cache','hands_submit_wait']}
    result={'samples':[],'timeline':[],'frame_count':0,'status':{},'fps':0.,'duration_seconds':0.}
    cache=tempfile.SpooledTemporaryFile(max_size=c['cache_memory_mb']*1024**2,mode='w+b',dir=work)
    result['_cache']=cache;result['_cache_shape']=None
    motion=CameraMotion(c['motion']) if camera.get('motion_check',True) else None
    frames=[];signal=None;block=None;freeze=None
    previous=None;previous_t=None;next_sample=0.;declared=0;errors=0;last_frame=None
    try:
        before=time.perf_counter();open_kwargs={}
        if c.get('decode_backend','cpu')=='cuda':
            from av.codec.hwaccel import HWAccel
            device=c['decode_gpus'][WORKER_SLOT%len(c['decode_gpus'])]
            # 显式禁用软件回退，失败就报算子异常，不能把CPU解码误计为GPU硬解。
            open_kwargs['hwaccel']=HWAccel('cuda',device=str(device),allow_software_fallback=False)
        if camera.get('_predecoded'):
            from video_qc_v2.persistent_decode import Container
            source=Container(camera['_predecoded'])
        else:source=av.open(camera['path'],**open_kwargs)
        with source as container:
            if not container.streams.video:raise ValueError('无视频流')
            stream=container.streams.video[0];stream.codec_context.thread_count=2;stream.codec_context.options={'err_detect':'explode'}
            result['decoder']={'requested':c.get('decode_backend','cpu'),'hardware_accelerated':bool(stream.codec_context.is_hwaccel),
                               'device':getattr(container,'decode_device',device if open_kwargs else None),'software_fallback_allowed':False if open_kwargs or camera.get('_predecoded') else None}
            if open_kwargs and not stream.codec_context.is_hwaccel:raise RuntimeError('GPU硬解未启用，拒绝CPU回退')
            timing['decoder_setup']+=time.perf_counter()-before
            fps=float(stream.average_rate or camera.get('fps',30));result['fps']=fps
            start=camera.get('start_seconds',float((stream.start_time or 0)*stream.time_base))
            full_duration=float(stream.duration*stream.time_base) if stream.duration is not None else float(container.duration or 0)/1e6
            end=camera.get('end_seconds',start+full_duration)
            result.update(duration_seconds=end-start,source_size=[stream.width,stream.height],codec=stream.codec_context.name,
                          normalization='swscale to full-range gray8; signalstats on gray-converted YUVJ444')
            declared=camera.get('expected_frames',expected if expected is not None else (stream.frames if start==0 else None))
            if start>0:container.seek(max(0,int(start/stream.time_base)),stream=stream,backward=True)
            iterator=iter(container.decode(stream));last_abs_t=None
            def accept(frame,t):
                nonlocal next_sample,signal,block
                before=time.perf_counter();gray=normalized_gray(frame,c['analysis_width']);timing['gray']+=time.perf_counter()-before
                before=time.perf_counter()
                point={'seconds':float(t),'mean':float(gray.mean()),'mad':float(np.mean(cv2.absdiff(gray,frames[-1][1]))) if frames else 0.,
                       'identical':bool(np.array_equal(gray,frames[-1][1])) if frames else False}
                # 只保留上一帧像素；长期保留的仅是标量时间序列。
                frames[:]=[(t,gray)]
                result['timeline'].append(point)
                fresh=av.VideoFrame.from_ndarray(gray,format='gray');fresh.pts=frame.pts;fresh.time_base=frame.time_base
                if freeze is not None:freeze.push(fresh)
                timing['temporal']+=time.perf_counter()-before
                before=time.perf_counter()
                if motion is not None:
                    if point['mad']>c['scene_change_mad']:
                        motion.previous=None;motion.segment+=1;motion.global_matrix=np.eye(3);motion.segment_start=t
                    motion.push(gray,t);timing['lk']+=time.perf_counter()-before
                if t+1e-6<next_sample:return
                next_sample=(np.floor(t*c['sample_hz']+1e-6)+1)/c['sample_hz']
                before=time.perf_counter()
                # FFmpeg原生signalstats，只在4Hz输入上算曝光；闪烁的全帧均值已廉价计算。
                color=frame.reformat(width=gray.shape[1],height=gray.shape[0],format='yuvj444p')
                if signal is None:signal=MetadataFilter(color,'signalstats','',Path(work)/f'{key}_signal.txt')
                signal.push(color);timing['signalstats']+=time.perf_counter()-before
                before=time.perf_counter()
                # 大面积异常需要空间证据：4x4分区保留原像素分辨率，逐区调用现成滤镜。
                # 不再同时重复计算旧全局blockdetect；后面的全局兼容分数取分区中位数。
                tile_scores=[]
                if c.get('block_detection',True):
                    tile_scores=spatial_block_scores(frame,c['block_grid'],c['block_period_min'],c['block_period_max'],Path(work)/f'{key}_block_tile.txt')
                    timing['blockdetect']+=time.perf_counter()-before
                # 模糊采用原有边缘区域指标；低纹理不作为曝光不足的证据。
                gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
                edges=cv2.magnitude(gx,gy)>c['edge_gradient'];edge_fraction=float(edges.mean())
                sharp=float(np.abs(cv2.Laplacian(gray,cv2.CV_32F))[edges].mean()) if edge_fraction>=c['min_edge_fraction'] else None
                dark=float((gray<c['dark_pixel']).mean());bright=float((gray>c['bright_pixel']).mean())
                # 网格只统计亮度，不因为纹理少就判曝光故障。
                tiles=[gray[y:y+max(1,len(gray)//4),x:x+max(1,gray.shape[1]//4)] for y in range(0,len(gray),max(1,len(gray)//4)) for x in range(0,gray.shape[1],max(1,gray.shape[1]//4))]
                sample={'seconds':float(t),'frame_index':result['frame_count']-1,'brightness':point['mean'],'dark_fraction':dark,'bright_fraction':bright,
                    'dark_tile_fraction':float(np.mean([a.mean()<c['dark_mean'] for a in tiles])),
                    'bright_tile_fraction':float(np.mean([a.mean()>c['bright_mean'] for a in tiles])),
                    'edge_fraction':edge_fraction,'sharpness':sharp,'low_texture':sharp is None,
                    'blur':sharp is not None and sharp<c['blur_score'],'block_tile_scores':tile_scores,
                    'block_score':float(np.median([v for v in tile_scores if v is not None])) if any(v is not None for v in tile_scores) else None,
                    'hand_count':None,'hand_boxes':[]}
                # 机器人/UMI不检测人手，不必生成和缓存YOLO输入图；复核时再按需读取源视频。
                if record.get('kind')=='robot':
                    result['samples'].append(sample);return
                before=time.perf_counter();image=frame.to_ndarray(format='bgr24')
                ratio=min(1.,c['hand_imgsz']/max(frame.width,frame.height));size=(round(frame.width*ratio),round(frame.height*ratio))
                small=cv2.resize(image,size,interpolation=cv2.INTER_LINEAR) if ratio<1 else image
                if result['_cache_shape'] is None:result['_cache_shape']=small.shape
                if small.shape!=result['_cache_shape']:raise ValueError('回合中途尺寸变化')
                cache.write(small.tobytes());timing['cache']+=time.perf_counter()-before
                result['samples'].append(sample)
            for_first=True
            while True:
                before=time.perf_counter()
                try:frame=next(iterator)
                except StopIteration:timing['decode']+=time.perf_counter()-before;break
                timing['decode']+=time.perf_counter()-before
                if frame.pts is None:errors+=1;continue
                absolute=float(frame.pts*frame.time_base)
                if absolute<start-1e-6:continue
                if absolute>=end-1e-6:break
                t=absolute-start;result['frame_count']+=1
                if getattr(frame,'is_corrupt',False):reason(report,'decode','FAIL','解码器标记损坏帧',key,at=t)
                if previous_t is not None:
                    dt=t-previous_t
                    if dt<=0:errors+=1
                    elif camera.get('timing_mode','cfr')=='cfr' and dt>1.5/fps:
                        result.setdefault('gaps',[]).append({'start_seconds':previous_t,'end_seconds':t,'duration_seconds':dt,'estimated_missing':max(1,round(dt*fps)-1)})
                previous_t=t;last_abs_t=absolute;last_frame=frame
                if for_first:
                    gray=normalized_gray(frame,c['analysis_width']);fresh=av.VideoFrame.from_ndarray(gray,format='gray');fresh.pts=frame.pts;fresh.time_base=frame.time_base
                    freeze=MetadataFilter(fresh,'freezedetect',f'noise={c["freeze_noise"]}:duration={c["freeze_seconds"]}',Path(work)/f'{key}_freeze.txt');for_first=False
                accept(frame,t)
            # 不额外复制/转换尾帧，4Hz规则按真实PTS取样，完整性仍覆盖所有帧。
            if result['frame_count']==0:raise ValueError('无可解码帧')
            if declared and result['frame_count']!=declared:reason(report,'count_mismatch','FAIL',f'decoded={result["frame_count"]}, expected={declared}',key)
            if camera.get('timing_mode','cfr')=='cfr' and abs(result['frame_count']/fps-(end-start))>1.1/fps:
                reason(report,'count_mismatch','FAIL','解码时长与声明区间不一致',key)
    except (OSError,ValueError,av.error.FFmpegError) as exc:reason(report,'decode','FAIL',str(exc),key,at=previous_t or 0.)
    return finalize_camera(result,report,camera,key,c,motion,signal,freeze,errors,previous_t,timing)


def finalize_camera(result,report,camera,key,c,motion,signal,freeze,errors,previous_t,timing):
    """复用同一套时长阈值、冻结/闪烁/抖动判定；CPU/GPU只负责提供底层指标。"""
    if errors:reason(report,'timestamps','FAIL',f'{errors}个缺失/非递增PTS',key)
    if result.get('gaps'):reason(report,'drop_frames','FAIL','CFR时间轴存在缺口；不声称能发现已被补帧掩盖的采集丢帧',key,result['gaps'])
    try:
        signal_rows=signal.finish() if signal else [];freeze_rows=freeze.finish() if freeze else []
    except (OSError,ValueError,av.error.FFmpegError) as exc:
        reason(report,'operator_error','REVIEW',f'FFmpeg metadata: {exc}',key);signal_rows=[];freeze_rows=[]
    samples=result['samples']
    if len(signal_rows)!=len(samples):reason(report,'operator_error','REVIEW','滤镜指标数量不匹配',key)
    for sample,sr in zip(samples,signal_rows):
        sample.update(YAVG=sr.get('lavfi.signalstats.YAVG'),YLOW=sr.get('lavfi.signalstats.YLOW'),YHIGH=sr.get('lavfi.signalstats.YHIGH'),UAVG=sr.get('lavfi.signalstats.UAVG'),VAVG=sr.get('lavfi.signalstats.VAVG'),scene_score=sr.get('lavfi.scd.score',0.))
        sample['underexposed']=sample['YAVG'] is not None and sample['YAVG']<c['dark_mean'] and sample['dark_fraction']>c['dark_fraction'] and sample['dark_tile_fraction']>=c['exposure_tile_fraction']
        sample['overexposed']=sample['YAVG'] is not None and sample['YAVG']>c['bright_mean'] and sample['bright_fraction']>c['bright_fraction'] and sample['bright_tile_fraction']>=c['exposure_tile_fraction']
    times=np.array([s['seconds'] for s in samples]);duration=result['duration_seconds']
    for code in ['underexposed','overexposed','blur','low_texture']:
        spans=intervals(times,[s.get(code,False) for s in samples],c['exposure_seconds'] if code in ('underexposed','overexposed') else c['quality_seconds'],.4)
        if spans:reason(report,code,'REVIEW','持续触发画质候选，阈值待人工校准',key,spans)
    scores=np.array([s.get('block_score',0.) or 0. for s in samples])
    if c.get('block_detection',True) and len(scores):
        # 每个固定空间区域独立建立稳健基线，避免桌面边缘/栅格纹理长期偏高而误报。
        # 面积比例指“异常分区覆盖率”，不是精确的损坏像素比例；区域等面积。
        tiles=np.asarray([s['block_tile_scores'] for s in samples],dtype=float)
        thresholds=[]
        for column in tiles.T:
            valid=column[np.isfinite(column)]
            median=float(np.median(valid)) if len(valid) else 0.
            mad=1.4826*np.median(abs(valid-median)) if len(valid) else 0.
            thresholds.append(max(c['block_absolute'],median+c['block_mad']*mad) if len(valid) else float('inf'))
        thresholds=np.asarray(thresholds)
        fractions=np.mean(tiles>thresholds,axis=1)
        result['block_tile_thresholds']=[float(v) if np.isfinite(v) else None for v in thresholds]
        for sample,fraction,local in zip(samples,fractions,tiles):
            sample['block_area_fraction']=float(fraction)
            sample['block_unscorable_fraction']=float(np.mean(~np.isfinite(local)))
        spans=intervals(times,fractions>=c['block_area_fraction'],c['block_seconds'],.4)
        if spans:reason(report,'blockiness','REVIEW',f'至少{c["block_area_fraction"]:.0%}分区分数异常，持续至少{c["block_seconds"]}秒；规则纹理仍需复核',key,spans)
    timeline=result['timeline'];ts=np.array([v['seconds'] for v in timeline]);means=np.array([v['mean'] for v in timeline]);mads=np.array([v['mad'] for v in timeline])
    cuts=[s['seconds'] for s in samples if s.get('scene_score',0)>15]
    result['scene_cut_times']=cuts
    # 花屏的颜色证据：异常色块短时出现又恢复，并且同时有异常块结构；不以彩色物体本身判坏。
    if c.get('block_detection',True) and len(samples)>2:
        uv=np.array([[s.get('UAVG') or 128,s.get('VAVG') or 128] for s in samples])
        color_bad=np.zeros(len(samples),bool)
        color_bad[1:-1]=(np.max(abs(uv[1:-1]-(uv[:-2]+uv[2:])/2),axis=1)>c['color_spike'])&(np.max(abs(uv[2:]-uv[:-2]),axis=1)<c['color_return'])&(scores[1:-1]>c['block_absolute'])
        color_intervals=intervals(times,color_bad,0.,.4)
        if color_intervals:reason(report,'color_corruption','REVIEW','颜色瞬变并恢复，且伴随块状结构；待复核排除真实物体变化',key,color_intervals)
    # freezedetect提供候选；结束后的帧差超附近正常水平才增加“恢复跳变”证据。
    freeze_start=None;freezes=[]
    camera_start=camera.get('start_seconds',0.)
    for row in freeze_rows:
        if 'lavfi.freezedetect.freeze_start' in row:freeze_start=max(0.,row['lavfi.freezedetect.freeze_start']-camera_start)
        if 'lavfi.freezedetect.freeze_end' in row and freeze_start is not None:
            end_time=row['lavfi.freezedetect.freeze_end']-camera_start
            freezes.append({'start_seconds':freeze_start,'end_seconds':end_time,'duration_seconds':end_time-freeze_start});freeze_start=None
    if freeze_start is not None:freezes.append({'start_seconds':freeze_start,'end_seconds':duration,'duration_seconds':duration-freeze_start})
    for span in freezes:
        j=int(np.argmin(abs(ts-span['end_seconds']))) if len(ts) else 0
        baseline=np.median(mads[max(0,j-30):min(len(mads),j+30)]) if len(mads) else 0
        jump=float(mads[j]) if len(mads) else 0.
        strong=jump>max(c['freeze_jump_absolute'],baseline*c['freeze_jump_ratio'])
        reason(report,'freeze_jump' if strong else 'freeze','REVIEW' if strong else 'WARN',
               f'恢复帧差={jump:.2f}, 邻域基线={baseline:.2f}；普通停顿不直接判坏',key,[span])
    # 闪烁检测复用全帧亮度均值；需要“变化够大且来回”，单次场景改变单独排除。
    if len(ts)>5:
        diffs=np.diff(means,prepend=means[0]);window=max(3,round(result['fps']*.5))
        reversals=np.r_[False,(diffs[1:]*diffs[:-1]<0)&(abs(diffs[1:])>c['flicker_delta'])&(abs(diffs[:-1])>c['flicker_delta'])]
        counts=np.convolve(np.pad(reversals.astype(int),(window//2,window-1-window//2)),np.ones(window),mode='valid')
        near_cut=np.array([any(abs(t-cut)<.5 for cut in cuts) for t in ts])
        flags=(~near_cut)&(counts>=c['flicker_reversals'])&(abs(diffs)>c['flicker_delta'])&(mads<c['scene_change_mad'])
        spans=intervals(ts,flags,c['flicker_seconds'],.15)
        if spans:reason(report,'flicker','REVIEW','反复明暗变化，可能照明/自动曝光变化，待复核',key,spans)
        flash=np.zeros(len(ts),bool)
        flash[1:-1]=(~near_cut[1:-1])&(abs(means[1:-1]-(means[:-2]+means[2:])/2)>c['flash_delta'])&(abs(means[2:]-means[:-2])<c['flash_return_delta'])&(mads[1:-1]<c['scene_change_mad'])
        spans=intervals(ts,flash,0.,.15)
        if spans:reason(report,'flash','REVIEW','单帧亮度尖峰并恢复，待复核排除真实光照变化',key,spans[:20])
    # 禁用视角不运行LK、不输出抖动或运动置信度候选，并明确报告SKIPPED而非PASS。
    summary=motion.summarize() if motion is not None else {'scorable':False,'disabled':True,'shake_intervals':[],'unreliable_intervals':[],'reason':'camera motion_check=false'}
    result['motion']=summary
    if summary['shake_intervals']:reason(report,'shake','REVIEW','可靠全局运动中出现持续高频往复和残差超限',key,summary['shake_intervals'])
    uncertain=summary['unreliable_intervals']
    if uncertain:
        central=[row.get('central_fraction',0)>c['motion']['max_central_fraction'] for row in motion.rows]
        concentrated=intervals([row['seconds'] for row in motion.rows],central,c['motion']['unreliable_run_seconds'],c['motion']['max_frame_gap_seconds'])
        reason(report,'motion_uncertain' if concentrated else 'motion_unscorable','REVIEW' if concentrated else 'WARN',
               '背景内点覆盖/拟合置信度不足，不当作真实镜头抖动',key,concentrated or uncertain)
    result['_motion_rows']=motion.rows if motion is not None else [];result['timing_seconds']=timing
    result['status']={'integrity':'FAIL' if any(v['severity']=='FAIL' and v['camera']==key for v in report['reasons']) else 'PASS',
                      'block_detection':'CHECKED' if c.get('block_detection',True) else 'SKIPPED_DISABLED',
                      'color_corruption':'CHECKED' if c.get('block_detection',True) else 'SKIPPED_MISSING_BLOCK_EVIDENCE',
                      'motion':'SKIPPED_CAMERA_ROLE' if motion is None else 'REVIEW' if summary['shake_intervals'] else 'UNSCORABLE' if not summary.get('scorable') else 'CHECKED',
                      'hand_detection':'PENDING'}
    return result


def check_multicam(record,r,c):
    cameras=r['cameras']
    if len(record['cameras'])>1 and len(cameras)!=len(record['cameras']):return {'status':'SKIPPED_FILE_ERROR','reason':'前置文件失败，未完成所有相机扫描'}
    if len(cameras)<2:return {'status':'NOT_APPLICABLE','reason':'单路视频'}
    if not record.get('shared_clock',False):return {'status':'UNSCORABLE','reason':'未声明共同时间基准，不能推断真实同步'}
    arrays={key:np.array([v['seconds']+record['cameras'][key].get('clock_offset_seconds',0.) for v in cam['timeline']]) for key,cam in cameras.items()}
    if any(not len(v) for v in arrays.values()):return {'status':'FAIL','reason':'某路没有有效帧'}
    reference=next(iter(arrays));a=arrays[reference];issues=[]
    for key,b in arrays.items():
        if key==reference:continue
        tolerance=c['sync_tolerance_seconds']
        # 支持不同帧率：每路覆盖同一重叠时间窗的参考时刻，不比较帧数是否完全相等。
        tol=max(tolerance,.6/min(cameras[reference]['fps'],cameras[key]['fps']))
        indices=np.searchsorted(b,a);right=b[np.minimum(indices,len(b)-1)];left=b[np.maximum(indices-1,0)]
        missing=np.minimum(abs(a-right),abs(a-left))>tol
        spans=intervals(a,missing,c['sync_missing_seconds'],.15)
        if spans:issues.append({'camera':key,'intervals':spans})
        if abs(a[0]-b[0])>tol or abs(a[-1]-b[-1])>tol:issues.append({'camera':key,'start_end_mismatch':True})
    if issues:reason(r,'multi_camera','FAIL',json.dumps(issues,ensure_ascii=False));return {'status':'FAIL','issues':issues}
    return {'status':'PASS','reference':reference,'shared_clock_assumed':True}


def process_record(job):
    """一次解码收集必要指标；显式分阶段判断后才投递YOLO，失败回合不浪费GPU。"""
    record,c,work_root=job;started=time.perf_counter();cv2.setNumThreads(1)
    r={'id':record['id'],'task':record['task'],'kind':record.get('kind','human'),'reasons':[],'cameras':{},'status':'PASS','source_modified':False}
    with tempfile.TemporaryDirectory(prefix='record_',dir=work_root) as work:
        before=time.perf_counter();expected=file_check(record,r);r['file_check_seconds']=time.perf_counter()-before
        if not any(v['severity']=='FAIL' for v in r['reasons']):
            for key,camera in record['cameras'].items():
                try:
                    if camera.get('_predecoded',{}).get('gpu_quality'):
                        from video_qc_v2.gpu_quality import scan_precomputed
                        r['cameras'][key]=scan_precomputed(record,key,camera,expected,r,c,work)
                    else:r['cameras'][key]=scan_camera(record,key,camera,expected,r,c,work)
                except Exception as exc:reason(r,'operator_error','REVIEW',f'{type(exc).__name__}: {exc}',key)
        r['multi_camera']=check_multicam(record,r,c)
        if record.get('clock_basis'):r['multi_camera']['clock_basis']=record['clock_basis']
        blocked=any(v['severity']=='FAIL' for v in r['reasons']) or not r['cameras']
        r['stages']={'files':'FAIL' if any(v['code']=='integrity' for v in r['reasons']) else 'PASS',
            'timeline':'FAIL' if blocked else 'CHECKED','image_analysis':'CHECKED' if r['cameras'] else 'SKIPPED',
            'hands':'SKIPPED_FILE_ERROR' if blocked else 'NOT_APPLICABLE_ROBOT' if record.get('kind')=='robot' else 'CHECKED'}
        for key,camera in r['cameras'].items():
            cache=camera.pop('_cache');shape=camera.pop('_cache_shape')
            try:
                if not blocked and record.get('kind')!='robot' and camera['samples']:
                    cache.seek(0);nbytes=int(np.prod(shape));pending=[];before=time.perf_counter()
                    for sample in camera['samples']:
                        raw=cache.read(nbytes)
                        if len(raw)!=nbytes:raise RuntimeError('临时图像缓存不完整')
                        pending.append(CLIENT.submit(np.frombuffer(raw,np.uint8).reshape(shape)))
                    width,height=camera['source_size'];sx=width/shape[1];sy=height/shape[0]
                    for sample,token in zip(camera['samples'],pending,strict=True):
                        boxes=CLIENT.resolve(token)
                        for box in boxes:
                            x1,y1,x2,y2=box['xyxy'];box['xyxy']=[x1*sx,y1*sy,x2*sx,y2*sy]
                        sample['hand_boxes']=boxes;sample['hand_count']=len(boxes)
                    camera['timing_seconds']['hands_submit_wait']+=time.perf_counter()-before
                    times=[v['seconds'] for v in camera['samples']]
                    spans=intervals(times,[v['hand_count']==0 for v in camera['samples']],c['hands_missing_seconds'],.4)
                    if spans:reason(r,'hands_missing','REVIEW','连续4Hz抽样未检出任何手；仅为模型候选',key,spans)
                    camera['status']['hand_detection']='REVIEW' if spans else 'PASS_SAMPLED'
                else:camera['status']['hand_detection']=r['stages']['hands']
            finally:cache.close()
            # 普通回合只留摘要/4Hz指标，不存全帧特征坐标。异常时留紧凑运动轨迹以复核。
            if not any(v['code'] in ['shake','motion_uncertain','motion_unscorable'] for v in r['reasons']):camera.pop('_motion_rows',None)
            camera.pop('timeline',None)
        severity=[v['severity'] for v in r['reasons']]
        r['status']='FAIL' if 'FAIL' in severity else 'REVIEW' if 'REVIEW' in severity else 'PASS'
        r['total_worker_seconds']=time.perf_counter()-started
    return r
