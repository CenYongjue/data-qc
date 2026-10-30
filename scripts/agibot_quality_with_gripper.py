#!/usr/bin/env python3
"""AgiBot 视频质量与夹爪存在检测整合试验；保留速览和限量问题片段。

所有阈值集中在 JSON 配置中；统计阈值不是机器人物理限位。
视频采用固定时间抽样，不声称完整排查所有坏帧、丢帧或短暂异常。
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'yolo'))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'matplotlib'))

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import hashlib
import html
import json
import subprocess
import time
import os
import platform
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = ['observation.images.head','observation.images.hand_left','observation.images.hand_right']
JOINT = 'observation.states.joint.position'
END = 'observation.states.end.position'
GRIP = 'observation.states.effector.position'
LABELS = {
 'gripper_absence':'夹爪连续未检出候选（模型漏检可能，须人工复核）',
 'nonfinite':'必需数值字段含 NaN/Inf', 'timestamp':'时间戳或帧索引异常',
 'video_read':'视频抽样读取失败/时间不匹配', 'underexposed':'抽样画面持续偏暗',
 'overexposed':'抽样画面持续过曝', 'blur':'抽样画面模糊候选',
 'low_texture':'纹理不足，无法可靠判断清晰度', 'freeze':'机器人运动期间画面冻结候选',
 'joint_outlier':'关节位置统计越界候选（非物理限位）', 'jitter':'末端运动突变/抖动候选',
 'near_duplicate':'近重复轨迹候选', 'exact_duplicate':'状态、动作和抽样图像完全重复',
 'no_motion':'没有检测到足够运动', 'invalid_orientation':'机器人姿态包含零范数四元数',
}


def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def log(text):
    print(text,flush=True)


def numpy_column(table, key, shape):
    """直接展开 Arrow 缓冲区，避免把多维数值先转成大量 Python 对象。"""
    arr=table[key].combine_chunks()
    if isinstance(arr.type,pa.ExtensionType): arr=arr.storage
    while pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type): arr=arr.flatten()
    return arr.to_numpy(zero_copy_only=False).reshape((len(table),)+tuple(shape))


def runs(mask, fps):
    """把连续异常帧合并成 [start,end) 时间区间，便于复查和解释。"""
    mask=np.asarray(mask,dtype=bool)
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return [{'start_frame':int(a),'end_frame_exclusive':int(b),
             'start_seconds':float(a/fps),'end_seconds':float(b/fps),'duration_seconds':float((b-a)/fps)}
            for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def max_run(mask,fps):
    return max((r['duration_seconds'] for r in runs(mask,fps)),default=0.)


def add_reason(report, code, severity, detail, at=0., camera=None, intervals=None):
    report['reasons'].append({'code':code,'label':LABELS[code],'severity':severity,
        'detail':detail,'camera':camera,'at_seconds':float(at),'intervals':intervals or []})


def robust_range(x):
    lo,hi=np.nanpercentile(x,[1,99],axis=0)
    return lo,hi,np.maximum(hi-lo,1e-6)


def moving_mean(x,window):
    # 居中平滑只用于异常检测，不写回状态/动作；边界另外排除，避免补边产生假异常。
    return np.stack([np.convolve(np.pad(col,(window//2,window//2),mode='edge'),
                    np.ones(window)/window,mode='valid') for col in x.T],axis=1)


def jitter_features(x,scale,fps,window):
    x=x.reshape(len(x),-1)/scale
    residual=x-moving_mean(x,window)
    velocity=np.gradient(residual,1/fps,axis=0)
    acceleration=np.gradient(velocity,1/fps,axis=0)
    return np.stack([np.linalg.norm(a,axis=1) for a in (residual,velocity,acceleration)],axis=1)


# 分阶段计时可重叠于总阶段：解码、检测和编码不能再次相加到总墙钟时间。
TIMING = {'decode_seconds': 0., 'gripper_seconds': 0., 'movie_seconds': 0., 'image_metrics_seconds': 0.}


class Reader:
    """稀疏抽样使用seek；连续读取复用解码迭代器，避免解码抽样点间整段视频。"""
    def __init__(self): self.containers = {}

    def frame(self, path, seconds, tolerance):
        started = time.perf_counter()
        try:
            key = str(path)
            if key not in self.containers:
                container = av.open(key); stream = container.streams.video[0]
                stream.codec_context.thread_count = 2
                self.containers[key] = [container, stream, None, None]
            state = self.containers[key]; container, stream, iterator, previous = state
            previous_time = float(previous.pts * previous.time_base) if previous is not None else None
            if iterator is None or (previous_time is not None and (seconds < previous_time - tolerance or seconds - previous_time > .2)):
                container.seek(max(0, int(seconds / stream.time_base)), stream=stream, backward=True)
                iterator = container.decode(stream); previous = None
            best = previous
            distance = abs(float(best.pts * best.time_base) - seconds) if best is not None else float('inf')
            if best is None or float(best.pts * best.time_base) < seconds - 1e-6:
                for frame in iterator:
                    if frame.pts is None: continue
                    t = float(frame.pts * frame.time_base)
                    if abs(t-seconds) < distance: best = frame; distance = abs(t-seconds)
                    previous = frame
                    if t >= seconds - 1e-6: break
            state[2:] = [iterator, previous]
            if best is None or distance > tolerance:
                raise ValueError(f'Expected {seconds:.6f}, difference {distance:.6f}s')
            return best.to_ndarray(format='bgr24'), float(best.pts * best.time_base)
        finally: TIMING['decode_seconds'] += time.perf_counter() - started

    def close(self):
        for container, *_ in self.containers.values(): container.close()


class GripperDetector:
    """复用已保存的三路视觉提示；夹爪未检出只能作为复查候选。"""
    def __init__(self, weights, profiles, confidence):
        import torch
        from ultralytics import YOLOE
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        self.models = {}; self.confidence = confidence
        source = YOLOE(str(weights)).model.state_dict()
        for camera in CAMERAS:
            model = YOLOE('yoloe-26n.yaml'); model.model.eval().fuse(verbose=False)
            target = model.model.state_dict()
            coverage = sum(v.numel() for k,v in target.items() if k in source and source[k].shape == v.shape) / sum(v.numel() for v in target.values())
            if coverage < .99: raise RuntimeError(f'Pretrained coverage {coverage:.2%}')
            model.load(str(weights))
            model.load_prompt_embeddings(profiles / (camera.split('.')[-1] + '_prompt.npz'))
            model.predict(np.zeros((480,640,3),np.uint8), imgsz=320, device='cpu', conf=confidence, verbose=False)
            self.models[camera] = model
        # Ultralytics 初始化 CPU predictor 时会重设线程数；所有模型热身后恢复限制。
        torch.set_num_threads(2)
        self.cpu_threads = torch.get_num_threads()

    def predict(self, camera, image):
        started = time.perf_counter()
        result = self.models[camera].predict(image, imgsz=320, device='cpu', conf=self.confidence,
                                            max_det=10, verbose=False, save=False)[0]
        boxes = [{'xyxy': b.tolist(), 'confidence': float(c)} for b,c in
                 zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())]
        TIMING['gripper_seconds'] += time.perf_counter() - started
        return boxes


class Movie:
    def __init__(self,path,fps):
        path.parent.mkdir(parents=True,exist_ok=True)
        self.process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
            '-f','rawvideo','-pix_fmt','bgr24','-s','960x300','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',
            '-threads','2','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def write(self,frame):
        started=time.perf_counter()
        self.process.stdin.write(frame.tobytes())
        TIMING['movie_seconds']+=time.perf_counter()-started
    def close(self):
        if not self.process.stdin.closed:
            self.process.stdin.close()
            started=time.perf_counter()
            result=self.process.wait()
            TIMING['movie_seconds']+=time.perf_counter()-started
            if result: raise RuntimeError('Review MP4 encoding failed')


def montage(frames,episode,t,caption,trim=None):
    out=np.zeros((300,960,3),dtype=np.uint8)
    for i,(name,frame) in enumerate(zip(CAMERAS,frames)):
        if frame is not None: out[30:270,i*320:(i+1)*320]=cv2.resize(frame,(320,240))
        cv2.putText(out,name.split('.')[-1],(i*320+8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
    tail=f'episode={episode:03d}  t={t:.2f}s  {caption}'
    if trim: tail+=f'  KEEP [{trim[0]:.2f}, {trim[1]:.2f})s'
    cv2.putText(out,tail,(8,290),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    return out


def image_metrics(frame,config):
    # 所有相机统一到 320x240 计算，阈值只适用于此尺寸；原视频没有被缩放修改。
    gray=cv2.cvtColor(cv2.resize(frame,(320,240)),cv2.COLOR_BGR2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    edge=cv2.magnitude(gx,gy)>config['edge_gradient_threshold']
    density=float(edge.mean()); lap=np.abs(cv2.Laplacian(gray,cv2.CV_32F))
    score=float(lap[edge].mean()) if density>=config['minimum_edge_fraction'] else None
    mean=float(gray.mean()); dark=float((gray<config['dark_pixel_threshold']).mean())
    bright=float((gray>config['bright_pixel_threshold']).mean())
    return {'mean_brightness':mean,'dark_fraction':dark,'bright_fraction':bright,
        'edge_fraction':density,'edge_laplacian':score,
        'underexposed':mean<config['dark_mean_threshold'] and dark>config['dark_fraction_threshold'],
        'overexposed':mean>config['bright_mean_threshold'] and bright>config['bright_fraction_threshold'],
        'blur':score is not None and score<config['blur_edge_laplacian_threshold'],
        'low_texture':score is None},gray


def video_path(root,info,ep,key):
    return root/info['video_path'].format(video_key=key,chunk_index=ep[f'videos/{key}/chunk_index'],
                                       file_index=ep[f'videos/{key}/file_index'])


def propose_trim(x,ep,scales,config,fps):
    speeds=[]
    for key in [JOINT,END,GRIP]:
        a=x[key].reshape(len(x[key]),-1)
        speeds.append(np.max(np.abs(np.diff(a,axis=0)/scales[key]),axis=1)*fps)
    speed=np.r_[0,np.maximum.reduce(speeds)]
    active=np.flatnonzero(speed>config['static_normalized_speed']); n=len(speed)
    if not len(active): return {'start_frame':0,'end_frame_exclusive':n,'no_motion':True,'changed':False},speed
    pad=round(config['trim_padding_seconds']*fps)
    start=max(0,int(active[0])-pad); end=min(n,int(active[-1])+pad+1)
    motion_range=[start,end]
    # 保护已有任务步骤：不把标注覆盖的抓取、保持或等待阶段当作无用静止删掉。
    # action_config 的结束边界语义未确认，保守地多保留一帧。
    annotations=ep.get('action_config') or []
    if annotations:
        start=min(start,min(int(a['start_frame']) for a in annotations))
        end=max(end,min(n,max(int(a['end_frame']) for a in annotations)+1))
    minimum=round(config['minimum_trim_seconds']*fps)
    if start<minimum: start=0
    if n-end<minimum: end=n
    if end-start<config['minimum_retained_seconds']*fps: start,end=0,n
    return {'start_frame':start,'end_frame_exclusive':end,'changed':start>0 or end<n,'no_motion':False,
        'start_seconds':start/fps,'end_seconds':end/fps,'removed_seconds':(start+n-end)/fps,
        'motion_only_range':motion_range,'annotation_protection':bool(annotations)},speed


def evidence(root,info,ep,report,out,reader,config):
    fps=info['fps']; n=ep['length']; targets=[]
    for r in sorted(report['reasons'],key=lambda r: {'FAIL':0,'REVIEW':1,'WARN':2}[r['severity']]):
        if r['severity']=='WARN': continue
        t=r['at_seconds']
        if all(abs(t-other)>config['evidence_clip_seconds']/2 for other,_ in targets): targets.append((t,r['code']))
        if len(targets)>=config['max_evidence_clips_per_episode']: break
    # 若只有裁剪建议，也给一个原速片段，方便判断是否误裁任务开始或结束。
    trim=report['trim_proposal']
    if not targets and trim['changed']:
        # 仅裁尾时应展示尾部边界，不能错误地把起点 0 秒当作裁剪证据。
        if trim['start_frame']>0: targets.append((trim['start_frame']/fps,'trim_start'))
        if trim['end_frame_exclusive']<n: targets.append((trim['end_frame_exclusive']/fps,'trim_end'))
        targets=targets[:config['max_evidence_clips_per_episode']]
    report['evidence_videos']=[]
    for j,(t,code) in enumerate(targets):
        width=round(config['evidence_clip_seconds']*fps)
        start=max(0,min(n-width,round(t*fps)-width//2)); end=min(n,start+width)
        relative=f'evidence/episode_{ep["episode_index"]:06d}_{j:02d}_{code}.mp4'
        movie=Movie(out/relative,fps); errors=0
        try:
            for fi in range(start,end):
                frames=[]
                for key in CAMERAS:
                    try:
                        image,_=reader.frame(video_path(root,info,ep,key),ep[f'videos/{key}/from_timestamp']+fi/fps,.5/fps)
                    except (ValueError,OSError,av.error.FFmpegError): image=None; errors+=1
                    frames.append(image)
                movie.write(montage(frames,ep['episode_index'],fi/fps,f'ORIGINAL SPEED / {code}'))
        finally: movie.close()
        report['evidence_videos'].append({'path':relative,'start_seconds':start/fps,'end_seconds':end/fps,
            'playback_fps':fps,'reason':code,'unreadable_camera_frames':errors})


def write_html(out,reports,summary):
    cards=[]
    for r in reports:
        reasons=''.join(f'<li><b>{html.escape(a["severity"]+" / "+a["label"])}</b> '
            f'{html.escape(a["camera"] or "")}：{html.escape(a["detail"])}</li>' for a in r['reasons']) or '<li>未触发本轮规则</li>'
        clips=''.join(f'<p>{html.escape(v["reason"])}：{v["start_seconds"]:.2f}–{v["end_seconds"]:.2f}s（原速）</p>'
            f'<video controls preload="none" src="{v["path"]}"></video>' for v in r['evidence_videos'])
        trim=r['trim_proposal']
        cards.append(f'<section class="{r["status"]}"><h2>Episode {r["episode_index"]:03d} — {r["status"]}</h2>'
            f'<ul>{reasons}</ul><p>建议保留帧 [{trim["start_frame"]}, {trim["end_frame_exclusive"]})；仅建议，未裁剪。</p>'
            f'<p>以下速览是 1 fps 时间抽样、5 fps 播放，不能用于判断真实动作速度或高频抖动。</p>'
            f'<video controls preload="none" src="{r["overview_video"]}"></video>{clips}'
            f'<p><a href="episodes/episode_{r["episode_index"]:06d}.json">详细指标和异常区间</a></p></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AgiBot 第一轮质检</title>'
        '<style>body{max-width:1100px;margin:30px auto;font:16px sans-serif;background:#f5f6f8;color:#17212b}'
        'section{background:white;padding:20px;margin:20px 0;border-left:6px solid #31885b}'
        '.FAIL{border-color:#d04444}.REVIEW{border-color:#d5a329}video{width:100%;max-width:960px}li{margin:8px 0}</style>'
        '<h1>AgiBot 第一轮质量报告</h1><p>本轮不删除或修改数据。FAIL 为规则不合格候选，REVIEW 为待人工复核；'
        'PASS 仅表示通过已执行的抽样规则，不代表完整视频或遮挡检查通过。三路相机必须分别合格。</p>'
        f'<pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>'
        '<p><a href="episodes.csv">轨迹汇总 CSV</a> · <a href="config.json">阈值配置</a> · '
        '<a href="manual_review.csv">人工复核填写表</a> · <a href="performance.md">处理速度报告</a></p>'+''.join(cards))


def write_performance(out,summary,timings):
    """吞吐分母用墙钟执行时间；三相机共享任务时长不重复计为三条轨迹。"""
    seconds=summary['frames']/summary['fps']
    wall=timings['total_seconds']
    cpu='unknown'
    try:
        cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except (OSError,StopIteration): pass
    performance={'timing':timings,'source_trajectory_seconds':seconds,
        'source_camera_video_seconds':seconds*len(CAMERAS),
        'trajectory_hours_per_wall_hour':seconds/wall,
        'camera_video_hours_per_wall_hour':seconds*len(CAMERAS)/wall,
        'episodes_per_wall_hour':summary['episodes']*3600/wall,
        'source_frames_per_wall_second':summary['frames']/wall,
        'cpu':cpu,'logical_cpus_visible':os.cpu_count(),'machine':platform.machine(),
        'decoder_threads_per_container':2,'encoder_threads':2,'opencv_threads':cv2.getNumThreads(),
        'gpu_used':False,'workers':summary['workers'],'gripper_enabled':summary['gripper_enabled'],'detector_cpu_threads':summary['detector_cpu_threads'],'sample_hz':json.loads((out/'config.json').read_text())['sample_hz'],
        'scope':'数值检查、三相机1Hz图像质量、速览MP4、限量原速证据、CSV/JSON/HTML报告；夹爪检测启用='+str(summary['gripper_enabled']),
        'extrapolation':'按当前选定轨迹估算；不同编码、关键帧间隔、异常数量和磁盘负载会改变吞吐'}
    if timings.get('one_off_media_revision_seconds') is not None:
        total=wall+timings['one_off_media_revision_seconds']
        performance['including_one_off_revision']={'seconds':total,'trajectory_hours_per_wall_hour':seconds/total,
                                                   'camera_video_hours_per_wall_hour':seconds*3/total}
    dump(out/'performance.json',performance)
    rows='\n'.join(f'| {key} | {value:.2f} 秒 |' for key,value in timings.items() if key.endswith('_seconds') and isinstance(value,(int,float)))
    text=f'''# 处理速度实测

本批 {summary['episodes']} 条轨迹、{summary['frames']} 帧、{summary['fps']} fps。
共享轨迹时长 {seconds/60:.2f} 分钟；三路视频累计 {seconds*3/60:.2f} 分钟。

| 阶段 | 墙钟时间 |
|---|---:|
{rows}

按首轮执行时间 {wall:.2f} 秒计算：
- **轨迹时间口径：{seconds/wall:.2f} 倍实时速度，即运行一小时可处理约 {seconds/wall:.2f} 小时三视角轨迹。**
- **相机视频累计口径：每小时约 {seconds*3/wall:.2f} 小时视频。**
- 约 {summary['episodes']*3600/wall:.0f} 条类似长度的轨迹/小时。

计时方式：{timings.get('measurement','perf_counter 墙钟计时')}。
硬件：{cpu}，GPU 未使用；单视频解码容器和编码器各 2 线程。

这是 1 Hz 视频抽样并生成复查材料的处理吞吐，不是逐帧完整清洗速度。
外推只适用于当前编码/分辨率、关键帧间隔、异常比例及缓存/磁盘条件。
夹爪存在检测启用：{summary['gripper_enabled']}；未包含学习式 IQA。顺序解码会经过抽样点之间的帧，但仅对抽样帧运行质检。
解码/检测/图像指标/编码为阶段内明细，与 analysis/evidence 等总阶段重叠，不能重复相加。
'''
    if 'including_one_off_revision' in performance:
        extra=performance['including_one_off_revision']
        text+=f"\n本次另有一次裁剪证据位置修正，额外用时 {timings['one_off_media_revision_seconds']:.2f} 秒；计入该一次性修正后，实际累计执行 {extra['seconds']:.2f} 秒，折合轨迹 {extra['trajectory_hours_per_wall_hour']:.2f} 倍实时 / 相机累计 {extra['camera_video_hours_per_wall_hour']:.2f} 倍。未计人工等待时间。\n"
    (out/'performance.md').write_text(text)
    return performance


def analyze_episode(job, detector=None):
    """按轨迹并行；输入数值切片和统一阈值，输出报告及去重特征。"""
    root,info,ep,x,jit,scales,limits,jthreshold,config,out,args=job
    cv2.setNumThreads(1)
    for key in TIMING: TIMING[key]=0.
    fps=info['fps'];reader=Reader()
    reports=[];video_samples=[];signatures=[];hashes=[]
    i=ep['episode_index'];n=ep['length'];duration=n/fps
    r={'episode_index':i,'length':n,'duration_seconds':duration,'task':ep['tasks'],
       'reasons':[],'cameras':{},'metrics':{},'status':'PASS'}
    for key,values in x.items():
        if not np.isfinite(values).all(): add_reason(r,'nonfinite','FAIL',key)
    ts=x['timestamp'].reshape(-1); fi=x['frame_index'].reshape(-1)
    if not np.array_equal(fi,np.arange(n)) or not np.all(np.diff(ts)>0) or not np.all(abs(ts-fi/fps)<1e-4):
        add_reason(r,'timestamp','FAIL','非递增时间或 timestamp 与 frame_index/fps 不一致')
    norm=np.linalg.norm(x['observation.states.robot.orientation'],axis=1)
    if (norm<1e-6).any():
        add_reason(r,'invalid_orientation','WARN',f'零范数比例 {float((norm<1e-6).mean()):.1%}；按无效辅助字段标记，不单独剔除轨迹')
    for key in [JOINT,'actions.joint.position']:
        lim=limits[key]; mask=np.any((x[key]<lim['lower'])|(x[key]>lim['upper']),axis=1)
        fraction=float(mask.mean()); longest=max_run(mask,fps)
        r['metrics'][key]={'outlier_fraction':fraction,'longest_seconds':longest}
        # 提供具体关节及范围，避免复查者只能看到笼统的“动作异常”。
        names=info['features'][key].get('names',{})
        names=names.get('motors',[]) if isinstance(names,dict) else (names or [])
        r['metrics'][key]['joint_details']=[{'joint':names[j] if j<len(names) else str(j),
            'outlier_frames':int(((x[key][:,j]<lim['lower'][j])|(x[key][:,j]>lim['upper'][j])).sum()),
            'minimum_observed':float(np.min(x[key][:,j])),'maximum_observed':float(np.max(x[key][:,j])),
            'lower_threshold':lim['lower'][j],'upper_threshold':lim['upper'][j]} for j in range(x[key].shape[1])]
        if fraction>=config['motion_outlier_fraction'] or longest>=config['motion_outlier_run_seconds']:
            add_reason(r,'joint_outlier','REVIEW',f'{key}: 越界比例 {fraction:.2%}，最长 {longest:.2f}s',
                np.flatnonzero(mask)[0]/fps,intervals=runs(mask,fps))
    mask=(jit>jthreshold).all(axis=1);padding=config['smoothing_window_frames']//2+2
    mask[:padding]=False;mask[-padding:]=False
    r['metrics']['jitter']={'thresholds':jthreshold.tolist(),'fraction':float(mask.mean()),'longest_seconds':max_run(mask,fps)}
    if max_run(mask,fps)>=config['jitter_run_seconds']:
        add_reason(r,'jitter','REVIEW','平滑残差及其一、二阶导数同时超过稳健阈值',np.flatnonzero(mask)[0]/fps,intervals=runs(mask,fps))
    trim,speed=propose_trim(x,ep,scales,config,fps);r['trim_proposal']=trim
    if trim['no_motion']: add_reason(r,'no_motion','REVIEW','整条轨迹未达到相对运动阈值')
    # 去重首先比较完整状态+动作，再用抽样图像摘要确认场景；近重复只报候选。
    h=hashlib.sha256(); keys=[k for k in x if k.startswith(('observation.states.','actions.'))]
    for key in sorted(keys):h.update(key.encode());h.update(x[key].tobytes())
    normalized=np.concatenate([x[JOINT]/scales[JOINT],x['actions.joint.position']/scales['actions.joint.position']],axis=1)
    signature=np.stack([np.interp(np.linspace(0,n-1,64),np.arange(n),col) for col in normalized.T],axis=1)
    signatures.append(signature); visual_hash=hashlib.sha256()
    sample_ids=sorted(set(np.r_[np.arange(0,n,max(1,round(fps/config['sample_hz']))),n-1].astype(int).tolist()))
    overview=f'overviews/episode_{i:06d}.mp4';r['overview_video']=overview;movie=Movie(out/overview,config['overview_playback_fps'])
    cam_samples={k:[] for k in CAMERAS}; previous={}
    try:
        for sample_index,frame_id in enumerate(sample_ids):
            t=frame_id/fps;frames=[]
            for key in CAMERAS:
                item={'episode_index':i,'camera':key,'frame_index':frame_id,'seconds':t}
                try:
                    image,pts=reader.frame(video_path(root,info,ep,key),ep[f'videos/{key}/from_timestamp']+t,.5/fps)
                    metrics_started=time.perf_counter()
                    metrics,gray=image_metrics(image,config)
                    TIMING['image_metrics_seconds']+=time.perf_counter()-metrics_started
                    item.update(metrics); item['pts']=pts; item['read_ok']=True
                    # 完全相同的原尺寸解码帧比缩略图更严格；静止场景不直接判冻结。
                    digest=hashlib.sha256(image.tobytes()).hexdigest();visual_hash.update(digest.encode())
                    item['same_as_previous_sample']=previous.get(key)==digest;previous[key]=digest
                except (ValueError,OSError,av.error.FFmpegError) as exc:
                    image=None;item.update(read_ok=False,error=str(exc));previous[key]=None
                if image is not None and detector is not None:
                    boxes=detector.predict(key,image)
                    item.update(gripper_detected=bool(boxes),gripper_boxes=boxes,
                                gripper_max_score=max((b['confidence'] for b in boxes),default=0.))
                    # 只在已有速览帧上画框，不另外输出逐帧 JPEG 或完整标注视频。
                    image=image.copy()
                    for box in boxes:
                        x1,y1,x2,y2=map(round,box['xyxy'])
                        cv2.rectangle(image,(x1,y1),(x2,y2),(0,220,0),2)
                cam_samples[key].append(item);video_samples.append(item);frames.append(image)
            movie.write(montage(frames,i,t,'SAMPLED OVERVIEW / NOT REAL TIME',
                                [trim['start_frame']/fps,trim['end_frame_exclusive']/fps]))
    finally: movie.close()
    hashes.append((h.hexdigest(),visual_hash.hexdigest(),n))
    for key,samples in cam_samples.items():
        errors=[s for s in samples if not s['read_ok']];valid=[s for s in samples if s['read_ok']]
        camera={'samples':len(samples),'read_errors':len(errors),'metrics':{}}
        if errors:add_reason(r,'video_read','FAIL',f'{len(errors)}/{len(samples)} 抽样点失败；{errors[0]["error"]}',errors[0]['seconds'],key)
        for code in ['underexposed','overexposed','blur','low_texture']:
            bads=[s for s in valid if s[code]];fraction=len(bads)/max(1,len(valid))
            camera['metrics'][code]={'fraction':fraction,'samples':len(bads),'threshold':config['blur_edge_laplacian_threshold'] if code=='blur' else None}
            cutoff=config['unscorable_fraction'] if code=='low_texture' else config['camera_bad_fraction']
            if fraction>=cutoff and len(bads)>=config['camera_bad_min_samples']:
                severity='REVIEW' if code=='low_texture' else 'FAIL'
                add_reason(r,code,severity,f'{len(bads)}/{len(valid)} 有效抽样点触发（{fraction:.1%}）；阈值待人工校准',bads[len(bads)//2]['seconds'],key,
                           intervals=[{'seconds':s['seconds'],'frame_index':s['frame_index']} for s in bads])
        # 不把两个相同抽样帧称为丢帧：仅报告持续相同且机器人同期明显运动的冻结候选。
        start=0
        for j in range(1,len(samples)+1):
            same=j<len(samples) and samples[j].get('same_as_previous_sample',False)
            if same:continue
            if j-start>=2:
                a,b=samples[start]['frame_index'],samples[j-1]['frame_index']
                span=(b-a)/fps
                motion=float(np.max(np.ptp(x[JOINT][a:b+1],axis=0)/scales[JOINT]))
                if span>=config['freeze_min_seconds'] and motion>=config['freeze_motion_fraction']:
                    add_reason(r,'freeze','REVIEW',f'抽样图像连续完全相同 {span:.2f}s，同期关节变化占范围 {motion:.3f}',a/fps,key,
                               intervals=[{'start_seconds':a/fps,'end_seconds':b/fps}])
            start=j
        # 时间段表示抽样点证据，不证明两点之间始终没有夹爪。
        intervals=[]; first=None; last=None
        for sample in samples+[None]:
            missing=detector is not None and sample is not None and sample['read_ok'] and not sample.get('gripper_detected',False)
            if missing:
                if first is None: first=sample['seconds']
                last=sample['seconds']
            elif first is not None:
                if last-first>=args.absence_seconds:
                    intervals.append({'start_seconds':first,'end_seconds':last,'duration_seconds':last-first})
                first=last=None
        camera['gripper']={'enabled':detector is not None,'valid_samples':len(valid),'detected_samples':sum(s.get('gripper_detected',False) for s in valid),
                           'absence_candidates':intervals,'confidence':args.confidence}
        if intervals:
            longest=max(intervals,key=lambda a:a['duration_seconds'])
            add_reason(r,'gripper_absence','REVIEW',
                       '连续抽样未检出；头部相机已有明显漏检，不能据此自动剔除',
                       (longest['start_seconds']+longest['end_seconds'])/2,key,intervals)
        severities=[a['severity'] for a in r['reasons'] if a['camera']==key]
        camera['status']='FAIL' if 'FAIL' in severities else 'REVIEW' if 'REVIEW' in severities else 'PASS'
        r['cameras'][key]=camera
    reports.append(r)
    reader.close()
    return reports[0], video_samples, signatures[0], hashes[0], dict(TIMING)


def render_evidence(job):
    root,info,ep,report,out,config=job
    cv2.setNumThreads(1)
    for key in TIMING: TIMING[key]=0.
    reader=Reader()
    try: evidence(root,info,ep,report,out,reader,config)
    finally: reader.close()
    return report, dict(TIMING)


def main():
    run_started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('agibot_qc_config.json'))
    parser.add_argument('--episodes', default='0,3,16,31', help='逗号分隔 episode ID；默认4条')
    parser.add_argument('--workers',type=int,default=1,choices=[1,2,4],help='CPU工作进程数，2/4进程仅支持关闭模型')
    parser.add_argument('--no-gripper', action='store_true', help='跳过模型加载和夹爪检测，实测原质量流程')
    parser.add_argument('--weights', type=Path, default=Path('models/agibot_gripper/yoloe-26n-seg.hf.pt'))
    parser.add_argument('--profiles', type=Path, default=Path('lerobotv3_output/agibot_gripper_pilot'))
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--absence-seconds', type=float, default=3., help='抽样连续未检出复查阈值')
    args=parser.parse_args(); root=args.dataset.resolve(); out=args.output.resolve()
    if args.workers>1 and not args.no_gripper: parser.error('Parallel benchmark requires --no-gripper')
    cv2.setNumThreads(1)
    config=json.loads(args.config.read_text()); info=json.loads((root/'meta/info.json').read_text()); fps=info['fps']
    if info['codebase_version']!='v3.0': raise ValueError('Expected LeRobot v3.0')
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    config.update(workers=args.workers,gripper_enabled=not args.no_gripper, gripper_confidence=args.confidence, gripper_absence_seconds=args.absence_seconds,
                  gripper_imgsz=320, gripper_model='YOLOE-26n detection-only', selected_episodes=args.episodes)
    dump(out/'config.json',config)
    columns=['episode_index','length','dataset_from_index','dataset_to_index','tasks','action_config']
    columns += [f'videos/{key}/{part}' for key in CAMERAS for part in ['chunk_index','file_index','from_timestamp','to_timestamp']]
    episodes=pq.read_table(root/'meta/episodes',columns=columns).to_pylist()
    table=pq.read_table(root/'data')
    numeric={k:numpy_column(table,k,f['shape']) for k,f in info['features'].items() if f['dtype']!='video'}
    scales={}; limits={}
    for key in [JOINT,END,GRIP,'actions.joint.position']:
        x=numeric[key].reshape(len(table),-1).astype(float);x[~np.isfinite(x)]=np.nan
        lo,hi,scale=robust_range(x); scales[key]=scale
        limits[key]={'p01':lo.tolist(),'p99':hi.tolist(),'scale':scale.tolist(),
            'lower':(lo-config['quantile_margin_fraction']*scale).tolist(),
            'upper':(hi+config['quantile_margin_fraction']*scale).tolist()}
    per_episode=[]; jitter=[]
    for ep in episodes:
        a,b=ep['dataset_from_index'],ep['dataset_to_index']
        x={k:v[a:b] for k,v in numeric.items()};per_episode.append(x)
        jitter.append(jitter_features(x[END],scales[END],fps,config['smoothing_window_frames']))
    all_jitter=np.concatenate(jitter); med=np.nanmedian(all_jitter,axis=0)
    mad=1.4826*np.nanmedian(abs(all_jitter-med),axis=0)
    jthreshold=med+config['jitter_mad_multiplier']*np.maximum(mad,1e-6)
    dump(out/'calibration.json',{'scope':'当前32条轨迹；非物理限位；未使用独立人工标注集校准',
         'joint_ranges':limits,'jitter_thresholds':jthreshold.tolist(),
         'jitter_units':'range-normalized residual, residual/sec, residual/sec^2'})
    # 全数据数值校准维持原阈值口径；仅选定轨迹执行视频和模型检查。
    selected={int(i) for i in args.episodes.split(',')}
    if not selected.issubset({e['episode_index'] for e in episodes}): raise ValueError('Unknown episode IDs')
    chosen=[(e,x,j) for e,x,j in zip(episodes,per_episode,jitter) if e['episode_index'] in selected]
    episodes,per_episode,jitter=map(list,zip(*chosen))
    model_started=time.perf_counter()
    detector=None if args.no_gripper else GripperDetector(args.weights,args.profiles,args.confidence)
    model_setup_seconds=0. if detector is None else time.perf_counter()-model_started
    reader=Reader(); reports=[]; video_samples=[]; signatures=[]; hashes=[]
    analysis_started=time.perf_counter()
    jobs=[(root,info,ep,x,jit,scales,limits,jthreshold,config,out,args)
          for ep,x,jit in zip(episodes,per_episode,jitter,strict=True)]
    executor=ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) if args.workers>1 else None
    # 校准和跨轨迹去重留在主进程，避免拆分任务改变阈值或遗漏跨进程重复。
    results=executor.map(analyze_episode,jobs) if executor else (analyze_episode(job,detector) for job in jobs)
    timing_totals={key:0. for key in TIMING}
    for report,samples,signature,hashed,timing in results:
        reports.append(report);video_samples.extend(samples);signatures.append(signature);hashes.append(hashed)
        for key,value in timing.items(): timing_totals[key]+=value
        log(f'Analyzed episode {report["episode_index"]}')
    analysis_finished=time.perf_counter()
    duplicate_pairs=[]
    for i in range(len(reports)):
        for j in range(i):
            if hashes[i]==hashes[j]:
                add_reason(reports[i],'exact_duplicate','FAIL',f'与 episode {reports[j]["episode_index"]} 的数值序列及全部抽样图像一致')
                duplicate_pairs.append({'a':reports[j]['episode_index'],'b':reports[i]['episode_index'],'kind':'exact_numeric_and_sampled_visual'})
            else:
                distance=float(np.sqrt(np.mean((signatures[i]-signatures[j])**2)))
                if distance<config['near_duplicate_distance']:
                    add_reason(reports[i],'near_duplicate','REVIEW',f'与 episode {reports[j]["episode_index"]} 时间归一化状态/动作距离 {distance:.5f}；未证实视觉场景相同')
                    duplicate_pairs.append({'a':reports[j]['episode_index'],'b':reports[i]['episode_index'],'kind':'motion_candidate','distance':distance})
    dedup_finished=time.perf_counter()
    for ep,r in zip(episodes,reports,strict=True):
        severity=[a['severity'] for a in r['reasons']]
        r['status']='FAIL' if 'FAIL' in severity else 'REVIEW' if 'REVIEW' in severity else 'PASS'
        r['suggested_operation']='exclude_whole_episode_candidate' if r['status']=='FAIL' else 'manual_review' if r['status']=='REVIEW' else 'trim_ends_candidate' if r['trim_proposal']['changed'] else 'keep_candidate'
    evidence_jobs=[(root,info,ep,r,out,config) for ep,r in zip(episodes,reports,strict=True)]
    rendered=executor.map(render_evidence,evidence_jobs) if executor else map(render_evidence,evidence_jobs)
    completed=[]
    for report,timing in rendered:
        completed.append(report)
        for key,value in timing.items(): timing_totals[key]+=value
        dump(out/f'episodes/episode_{report["episode_index"]:06d}.json',report)
    reports=completed
    if executor: executor.shutdown(wait=True)
    reader.close()
    media_finished=time.perf_counter()
    summary={'input':str(root),'episodes':len(reports),'frames':sum(ep['length'] for ep in episodes),'fps':fps,
        'status_counts':{s:sum(r['status']==s for r in reports) for s in ['PASS','REVIEW','FAIL']},
        'trim_candidates':sum(r['trim_proposal']['changed'] for r in reports),'duplicate_pairs':len(duplicate_pairs),
        'workers':args.workers,'gripper_enabled':detector is not None,'detector_cpu_threads':detector.cpu_threads if detector else 0,'gripper_samples':len([s for s in video_samples if 'gripper_detected' in s]),
        'source_modified':False,'applied_deletions':0,'applied_trims':0,
        'limitations':['视频1Hz抽样，不能排除抽样点之间的损坏、冻结或模糊',
                      '没有机械限位/单位说明；运动异常是统计候选',
                      ('夹爪检测仅为候选；头部相机漏检明显，未判断遮挡或任务成功' if detector else '本轮关闭夹爪模型，未检查夹爪是否出镜'),
                      '未执行学习式IQA；图像阈值需要人工复核校准',
                      '无原始传感器时间戳，不推断真实跨模态同步或控制延迟']}
    dump(out/'summary.json',summary);dump(out/'duplicate_pairs.json',duplicate_pairs)
    pd.DataFrame(video_samples).to_csv(out/'video_samples.csv',index=False)
    rows=[{'episode_index':r['episode_index'],'status':r['status'],'length':r['length'],
        'reason_codes':';'.join(sorted(set(a['code'] for a in r['reasons']))),
        'reasons':'; '.join(a['severity']+':'+a['label']+':'+a['detail'] for a in r['reasons']),
        'suggested_operation':r['suggested_operation'],'trim_start':r['trim_proposal']['start_frame'],
        'trim_end_exclusive':r['trim_proposal']['end_frame_exclusive'],'overview':r['overview_video']} for r in reports]
    pd.DataFrame(rows).to_csv(out/'episodes.csv',index=False)
    pd.DataFrame([{'episode_index':r['episode_index'],'automatic_status':r['status'],
        'human_decision':'','reason_correct':'','approved_trim_start':'','approved_trim_end_exclusive':'','notes':''} for r in reports]).to_csv(out/'manual_review.csv',index=False)
    write_html(out,reports,summary)
    write_performance(out,summary,{'measurement':'总阶段为墙钟计时；decode/gripper/movie/image_metrics为各工作进程累计，不能作为墙钟占比相加',
        'setup_seconds':analysis_started-run_started,'model_setup_seconds':model_setup_seconds,**timing_totals,'analysis_and_overviews_seconds':analysis_finished-analysis_started,
        'dedup_seconds':dedup_finished-analysis_finished,'evidence_seconds':media_finished-dedup_finished,
        'report_seconds':time.perf_counter()-media_finished,'total_seconds':time.perf_counter()-run_started})
    log(json.dumps(summary,ensure_ascii=False));log(f'DONE: {out}/index.html')


if __name__=='__main__':main()
