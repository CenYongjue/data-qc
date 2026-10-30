#!/usr/bin/env python3
"""EgoDex样本质检：全程解码/文件可读性、2Hz图像质量及画面双手同时未检出。

仅输出报告与限量复查材料，不删除、转换、裁剪源数据，不做运动合理性检查。
使用192x192手掌检测器，忽略模型附带的粗关键点，不运行手部姿态网络。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import html
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time
import av
import cv2
import h5py
import numpy as np

LABELS={'missing_file':'文件缺失或空文件','hdf5_read':'HDF5无法读取或必要字段异常',
        'video_read':'视频解码错误/损坏帧','frame_count':'视频与HDF5帧数不一致',
        'timestamp':'视频时间戳异常','blur':'持续模糊候选','underexposed':'持续曝光不足',
        'overexposed':'持续过曝','low_texture':'低纹理导致清晰度无法可靠判定',
        'both_hands_missing':'画面双手长时间未检出候选','freeze':'完全相同画面候选（可能自然停顿）'}


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


class PalmDetector:
    """OpenCV Zoo MediaPipe palm ONNX解码；仅取框和存在分数。"""
    def __init__(self,path,threshold):
        self.net=cv2.dnn.readNet(str(path));self.threshold=threshold
        anchors=[]
        for side,repeats in [(24,2),(12,6)]:
            anchors.extend([(x+.5,y+.5) for y in range(side) for x in range(side) for _ in range(repeats)])
        self.anchors=np.concatenate([np.array(anchors[:1152])/24,np.array(anchors[1152:])/12]).astype(np.float32)
    def predict(self,image):
        h,w=image.shape[:2];ratio=192/max(h,w);rh,rw=int(h*ratio),int(w*ratio)
        resized=cv2.resize(image,(rw,rh));top=(192-rh)//2;left=(192-rw)//2
        padded=cv2.copyMakeBorder(resized,top,192-rh-top,left,192-rw-left,cv2.BORDER_CONSTANT,value=0)
        blob=cv2.cvtColor(padded,cv2.COLOR_BGR2RGB).astype(np.float32)[None]/255
        self.net.setInput(blob);regression,logits=self.net.forward(self.net.getUnconnectedOutLayersNames())
        assert regression.shape[1]==len(self.anchors)
        score=1/(1+np.exp(-np.clip(logits[0,:,0],-80,80)))
        selected=score>=self.threshold
        if not selected.any():return []
        delta=regression[0,selected,:4]/192;centers=delta[:,:2]+self.anchors[selected];wh=delta[:,2:4]
        xy1=(centers-wh/2)*max(h,w)-np.array([left,top])/ratio
        xy2=(centers+wh/2)*max(h,w)-np.array([left,top])/ratio
        boxes=np.concatenate([xy1,xy2-xy1],axis=1);scores=score[selected]
        indices=cv2.dnn.NMSBoxes(boxes.tolist(),scores.tolist(),self.threshold,.3,top_k=50)
        result=[]
        for index in np.asarray(indices).reshape(-1):
            x1,y1=xy1[index];x2,y2=xy2[index]
            clipped=[max(0.,float(x1)),max(0.,float(y1)),min(float(w-1),float(x2)),min(float(h-1),float(y2))]
            if clipped[2]>clipped[0] and clipped[3]>clipped[1]:result.append({'xyxy':clipped,'score':float(scores[index])})
        return result


def image_metrics(image,c):
    gray=cv2.cvtColor(cv2.resize(image,(320,240)),cv2.COLOR_BGR2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    edge=cv2.magnitude(gx,gy)>c['edge_gradient_threshold'];density=float(edge.mean())
    lap=abs(cv2.Laplacian(gray,cv2.CV_32F));sharp=float(lap[edge].mean()) if density>=c['minimum_edge_fraction'] else None
    mean=float(gray.mean());dark=float((gray<c['dark_pixel_threshold']).mean());bright=float((gray>c['bright_pixel_threshold']).mean())
    return {'brightness':mean,'dark_fraction':dark,'bright_fraction':bright,'edge_fraction':density,'sharpness':sharp,
            'blur':sharp is not None and sharp<c['blur_edge_laplacian_threshold'],'low_texture':sharp is None,
            'underexposed':mean<c['dark_mean_threshold'] and dark>c['dark_fraction_threshold'],
            'overexposed':mean>c['bright_mean_threshold'] and bright>c['bright_fraction_threshold']}


def spans(samples,duration,predicate):
    # 仅按实际抽样点估计持续证据；不把一次模型漏检扩展为长时间不可见。
    result=[];start=None;last=None
    for sample in samples+[None]:
        if sample is not None and predicate(sample):
            if start is None:start=sample['seconds']
            last=sample['seconds']
        elif start is not None:
            end=min(duration,last+1e-6)
            result.append({'start_seconds':start,'end_seconds':end,'duration_seconds':max(0.,end-start)})
            start=last=None
    return result


def add_reason(r,code,severity,detail,at=0.,intervals=None):
    r['reasons'].append({'code':code,'label':LABELS[code],'severity':severity,'detail':detail,'at_seconds':float(at),'intervals':intervals or []})


def check_one(job):
    video,model,config=job;video=Path(video);hdf=video.with_suffix('.hdf5');cv2.setNumThreads(1)
    started=time.perf_counter();timing={'hdf5_seconds':0.,'decode_seconds':0.,'pixel_conversion_seconds':0.,'quality_metrics_seconds':0.,'hand_detection_seconds':0.,'model_setup_seconds':0.}
    r={'id':video.parent.name+'__'+video.stem,'task':video.parent.name,'clip':video.stem,'video':str(video),'hdf5':str(hdf),
       'reasons':[],'samples':[],'duration_seconds':0.,'status':'PASS','motion_checks_enabled':False,'source_modified':False}
    for path in (video,hdf):
        if not path.is_file() or path.stat().st_size==0:add_reason(r,'missing_file','FAIL',str(path))
    if r['reasons']:r.update(timing=timing,total_seconds=time.perf_counter()-started);return r
    before=time.perf_counter();expected=None;confidence={}
    try:
        with h5py.File(hdf,'r') as f:
            for key in ['camera/intrinsic','transforms/camera','transforms/leftHand','transforms/rightHand']:
                if key not in f:raise ValueError(f'Missing {key}')
            if f['camera/intrinsic'].shape!=(3,3):raise ValueError('Bad intrinsic shape')
            expected=len(f['transforms/camera']);issues=[];read_count=0
            def read_dataset(name,obj):
                nonlocal read_count
                if not isinstance(obj,h5py.Dataset):return
                array=np.asarray(obj[()]);read_count+=1
                if name.startswith('transforms/') and obj.shape!=(expected,4,4):issues.append(f'{name}: {obj.shape}')
                if name.startswith('confidences/') and obj.shape!=(expected,):issues.append(f'{name}: {obj.shape}')
                if np.issubdtype(array.dtype,np.number) and not np.isfinite(array).all():issues.append(f'{name}: NaN/Inf')
            f.visititems(read_dataset)
            if issues:raise ValueError('; '.join(issues[:10]))
            for side in ['leftHand','rightHand']:
                if 'confidences/'+side in f:confidence[side]=f['confidences/'+side][()].astype(float)
            r['hdf5_datasets_read']=read_count;r['hdf5_frame_count']=expected
            which=str(f.attrs.get('which_llm_description','1'))
            r['description']=str(f.attrs.get('llm_description2' if which=='2' else 'llm_description',''))
    except (OSError,ValueError,KeyError,RuntimeError,TypeError,AttributeError) as exc:add_reason(r,'hdf5_read','FAIL',str(exc))
    timing['hdf5_seconds']=time.perf_counter()-before
    before=time.perf_counter();detector=PalmDetector(model,config['hand_score_threshold'])
    if hasattr(detector,'warmup'):detector.warmup()
    else:detector.predict(np.zeros((192,192,3),np.uint8))
    timing['model_setup_seconds']=time.perf_counter()-before
    decoded=0;previous_pts=None;next_sample=0.;last_frame=None;last_t=None;last_index=None;previous_digest=None;timestamp_bad=0;corrupt=0
    def sample_frame(frame,t,index):
        nonlocal previous_digest
        before=time.perf_counter();image=frame.to_ndarray(format='bgr24');timing['pixel_conversion_seconds']+=time.perf_counter()-before
        before=time.perf_counter();metrics=image_metrics(image,config);timing['quality_metrics_seconds']+=time.perf_counter()-before
        before=time.perf_counter()
        request=detector.submit(image) if hasattr(detector,'submit') else None
        boxes=[] if request is not None else detector.predict(image)
        timing['hand_detection_seconds']+=time.perf_counter()-before
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        item={'seconds':t,'frame_index':index,'hand_count':len(boxes),'hand_boxes':boxes,'same_as_previous':digest==previous_digest,**metrics}
        if request is not None:item['_hand_request']=request
        previous_digest=digest
        for side,values in confidence.items():
            if index<len(values):item[side+'_tracking_confidence']=float(values[index])
        r['samples'].append(item)
    try:
        with av.open(str(video)) as container:
            if not container.streams.video:raise ValueError('No video stream')
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            if not stream.average_rate:raise ValueError('Missing video frame rate')
            stream.codec_context.options={'err_detect':'explode'}
            fps=float(stream.average_rate);origin=float((stream.start_time or 0)*stream.time_base)
            duration=float(stream.duration*stream.time_base) if stream.duration else float(container.duration)/1e6
            r.update(fps=fps,header_frame_count=stream.frames,codec=stream.codec_context.name,width=stream.width,height=stream.height,duration_seconds=duration)
            iterator=iter(container.decode(stream))
            while True:
                before=time.perf_counter()
                try:frame=next(iterator)
                except StopIteration:timing['decode_seconds']+=time.perf_counter()-before;break
                timing['decode_seconds']+=time.perf_counter()-before
                index=decoded;decoded+=1
                if frame.pts is None:timestamp_bad+=1;continue
                t=float(frame.pts*frame.time_base)-origin
                if previous_pts is not None and (t<=previous_pts or abs(t-previous_pts-1/fps)>max(.001,.2/fps)):timestamp_bad+=1
                previous_pts=t
                if frame.is_corrupt:corrupt+=1
                last_frame=frame;last_t=t;last_index=index
                if t+1e-6>=next_sample:
                    sample_frame(frame,t,index);next_sample=(np.floor(t*config['sample_hz']+1e-6)+1)/config['sample_hz']
            if last_frame is not None and (not r['samples'] or r['samples'][-1]['frame_index']!=last_index):sample_frame(last_frame,last_t,last_index)
            if expected is not None and decoded!=expected:add_reason(r,'frame_count','FAIL',f'decoded={decoded}, HDF5={expected}')
            if stream.frames and decoded!=stream.frames:add_reason(r,'frame_count','FAIL',f'decoded={decoded}, video header={stream.frames}')
            if abs(decoded/fps-duration)>1.1/fps:add_reason(r,'frame_count','FAIL',f'解码帧数/帧率与时长不一致：{decoded/fps:.4f} vs {duration:.4f}')
    except (OSError,ValueError,av.error.FFmpegError) as exc:add_reason(r,'video_read','FAIL',str(exc),last_t or 0.)
    if hasattr(detector,'resolve'):
        # 等到当前回合解码结束再收集结果；解码与GPU批推理已经重叠执行。
        before=time.perf_counter()
        for sample in r['samples']:
            boxes=detector.resolve(sample.pop('_hand_request'))
            sample['hand_boxes']=boxes;sample['hand_count']=len(boxes)
        timing['hand_detection_seconds']+=time.perf_counter()-before
    if corrupt:add_reason(r,'video_read','FAIL',f'{corrupt} corrupt frames')
    if timestamp_bad:add_reason(r,'timestamp','FAIL',f'{timestamp_bad} PTS anomalies')
    r['decoded_frames']=decoded;samples=r['samples'];duration=r['duration_seconds']
    if samples:
        for code in ['blur','underexposed','overexposed','low_texture']:
            bad=[s for s in samples if s[code]];fraction=len(bad)/len(samples)
            cutoff=config['unscorable_fraction'] if code=='low_texture' else config['camera_bad_fraction']
            if fraction>=cutoff and len(bad)>=config['camera_bad_min_samples']:
                add_reason(r,code,'REVIEW' if code=='low_texture' else 'FAIL',f'{len(bad)}/{len(samples)}抽样点触发，{fraction:.1%}',bad[len(bad)//2]['seconds'],spans(samples,duration,lambda s:s[code]))
        missing=spans(samples,duration,lambda s:s['hand_count']==0)
        long=[v for v in missing if v['duration_seconds']>=config['both_hands_missing_seconds']]
        r['hand_detection']={'sample_count':len(samples),'samples_with_any_hand':sum(s['hand_count']>0 for s in samples),
                             'missing_intervals':missing,'long_missing_intervals':long,'threshold_seconds':config['both_hands_missing_seconds']}
        if long:
            worst=max(long,key=lambda v:v['duration_seconds'])
            add_reason(r,'both_hands_missing','REVIEW','连续抽样未检出任何手；需要区分真实出画与手掌模型漏检',
                       (worst['start_seconds']+worst['end_seconds'])/2,long)
        frozen=spans(samples,duration,lambda s:s['same_as_previous'])
        long_freeze=[v for v in frozen if v['duration_seconds']>=config['freeze_min_seconds']]
        if long_freeze:add_reason(r,'freeze','REVIEW','连续抽样完全相同，可能正常停顿，不直接判损坏',long_freeze[0]['start_seconds'],long_freeze)
    severity=[a['severity'] for a in r['reasons']]
    r['status']='FAIL' if 'FAIL' in severity else 'REVIEW' if 'REVIEW' in severity else 'PASS'
    r['timing']=timing;r['total_seconds']=time.perf_counter()-started
    return r


class Movie:
    def __init__(self,path,fps):
        path.parent.mkdir(parents=True,exist_ok=True)
        self.proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n','-f','rawvideo','-pix_fmt','bgr24','-s','960x588','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def write(self,image):self.proc.stdin.write(image.tobytes())
    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():raise RuntimeError('Evidence encoder failed')


def annotate(image,sample,t,caption):
    canvas=np.zeros((588,960,3),np.uint8);canvas[:540]=cv2.resize(image,(960,540))
    if sample is not None:
        sx=960/image.shape[1];sy=540/image.shape[0]
        for box in sample['hand_boxes']:
            x1,y1,x2,y2=box['xyxy'];cv2.rectangle(canvas,(round(x1*sx),round(y1*sy)),(round(x2*sx),round(y2*sy)),(50,220,50),2)
        text=f'nearest sample={sample["seconds"]:.2f}s, hands={sample["hand_count"]} (2Hz boxes)'
    else:text='No valid detector sample'
    cv2.putText(canvas,f't={t:.2f}s  {caption}',(10,560),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    cv2.putText(canvas,text,(10,581),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,220,255),1)
    return canvas


def export_review(job):
    r,out,config=job;out=Path(out);cv2.setNumThreads(1);started=time.perf_counter();result={'id':r['id'],'videos':[],'snapshots':[]}
    if not r['samples']:return result
    reasons=sorted(r['reasons'],key=lambda v:(v['severity']!='FAIL',v['code']!='both_hands_missing'))
    targets=[]
    for reason in reasons:
        t=reason['at_seconds']
        if all(abs(t-v[0])>config['evidence_clip_seconds']/2 for v in targets):targets.append((t,reason['code']))
        if len(targets)>=config['max_evidence_clips_per_episode']:break
    for j,(t,code) in enumerate(targets):
        duration=r['duration_seconds'];lower=r.get('start_seconds',0.)
        length=min(config['evidence_clip_seconds'],duration-lower)
        start=max(lower,min(duration-length,t-length/2));end=start+length
        relative=f'evidence/{r["id"]}_{j}_{code}.mp4';movie=Movie(out/relative,r['fps']);snapshot=False
        try:
            with av.open(r['video']) as c:
                s=c.streams.video[0];s.codec_context.thread_count=2;origin=float((s.start_time or 0)*s.time_base)
                c.seek(max(0,int((start+origin)/s.time_base)),stream=s,backward=True)
                for frame in c.decode(s):
                    if frame.pts is None:continue
                    now=float(frame.pts*frame.time_base)-origin
                    if now<start-1e-6:continue
                    if now>=end-1e-6:break
                    sample=min(r['samples'],key=lambda v:abs(v['seconds']-now));image=frame.to_ndarray(format='bgr24')
                    canvas=annotate(image,sample,now,code+' / ORIGINAL SPEED')
                    movie.write(canvas)
                    if not snapshot and now>=t-1e-6:
                        name=f'snapshots/{r["id"]}_{j}.jpg';(out/'snapshots').mkdir(exist_ok=True);cv2.imwrite(str(out/name),canvas)
                        result['snapshots'].append(name);snapshot=True
        except (ValueError,OSError,av.error.FFmpegError) as exc:result.setdefault('render_errors',[]).append(str(exc))
        finally:movie.close()
        result['videos'].append({'path':relative,'start_seconds':start,'end_seconds':end,'reason':code,'fps':r['fps']})
    # 只为问题样本生成速览，2Hz抽样、4fps播放；不另外跑模型或保存全段JPEG。
    if config['problem_overviews']:
        relative=f'overviews/{r["id"]}.mp4';movie=Movie(out/relative,4)
        try:
            with av.open(r['video']) as c:
                s=c.streams.video[0];s.codec_context.thread_count=2
                origin=float((s.start_time or 0)*s.time_base)
                for sample in r['samples']:
                    c.seek(max(0,int((sample['seconds']+origin)/s.time_base)),stream=s,backward=True)
                    for frame in c.decode(s):
                        now=float(frame.pts*frame.time_base)-origin
                        if now>=sample['seconds']-1e-6:
                            movie.write(annotate(frame.to_ndarray(format='bgr24'),sample,now,'2Hz OVERVIEW / 2x SPEED'));break
        except (ValueError,OSError,av.error.FFmpegError) as exc:result.setdefault('render_errors',[]).append(str(exc))
        finally:movie.close()
        result['overview']=relative
    result['export_seconds']=time.perf_counter()-started
    return result


def main():
    started=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('datasets/egoDex/part1'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261027);p.add_argument('--workers',type=int,default=4,choices=[1,2,4])
    p.add_argument('--model',type=Path,default=Path('models/hand_presence/palm_detection_mediapipe_2023feb.onnx'))
    p.add_argument('--sample-hz',type=float,default=2.);p.add_argument('--missing-seconds',type=float,default=3.)
    p.add_argument('--review-count',type=int,default=6)
    args=p.parse_args();root=args.dataset.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    if args.count<1 or args.sample_hz<=0 or args.missing_seconds<=0:p.error('Positive count/hz/seconds required')
    out.mkdir(parents=True);cv2.setNumThreads(1)
    config=json.loads(Path(__file__).with_name('agibot_qc_config.json').read_text())
    config.update(sample_hz=args.sample_hz,both_hands_missing_seconds=args.missing_seconds,hand_score_threshold=.5,
                  motion_checks_enabled=False,problem_overviews=True)
    dump(out/'config.json',config)
    # 四个定向例子只为覆盖任务场景/追踪置信度差异；其余跨任务固定种子抽样，不称为随机无偏样本。
    selected=[]
    for task in ['fry_egg','fry_bread','boil_serve_egg','arrange_topple_dominoes']:
        path=root/task/'0.mp4'
        if path.exists() and len(selected)<args.count:selected.append(path)
    rng=np.random.default_rng(args.seed);tasks=sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {v.parent.name for v in selected})
    rng.shuffle(tasks)
    for folder in tasks:
        if len(selected)>=args.count:break
        paths=sorted(folder.glob('*.mp4'))
        if paths:selected.append(paths[int(rng.integers(len(paths)))])
    if len(selected)<args.count:
        others=sorted(set(root.rglob('*.mp4'))-set(selected));rng.shuffle(others);selected.extend(others[:args.count-len(selected)])
    dump(out/'selection.json',{'seed':args.seed,'method':'4 targeted scene examples plus seeded task-diverse sampling','files':[str(v) for v in selected]})
    setup_done=time.perf_counter()
    print(f'Checking {len(selected)} episodes; quality/hands={args.sample_hz}Hz; full decode; workers={args.workers}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        reports=[]
        for report in pool.map(check_one,[(str(v),str(args.model.resolve()),config) for v in selected]):
            reports.append(report);print(f'{report["id"]}: {report["status"]}, {report["decoded_frames"] if "decoded_frames" in report else 0} frames',flush=True)
            dump(out/'episodes'/f'{report["id"]}.json',report)
        analysis_done=time.perf_counter()
        def priority(r):
            longs=r.get('hand_detection',{}).get('long_missing_intervals',[])
            return (int(r['status']=='FAIL'),len({v['code'] for v in r['reasons']}),max((v['duration_seconds'] for v in longs),default=0))
        issues=sorted([r for r in reports if r['reasons']],key=priority,reverse=True)
        views=list(pool.map(export_review,[(r,str(out),config) for r in issues[:args.review_count]]))
    media_done=time.perf_counter();view_by_id={v['id']:v for v in views}
    sample_rows=[]
    for r in reports:
        r['review_media']=view_by_id.get(r['id'],{})
        dump(out/'episodes'/f'{r["id"]}.json',r)
        sample_rows.extend({'id':r['id'],**s} for s in r['samples'])
    if sample_rows:
        columns=list(dict.fromkeys(k for row in sample_rows for k in row))
        with (out/'samples.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(sample_rows)
    rows=[{'id':r['id'],'status':r['status'],'duration_seconds':r['duration_seconds'],'decoded_frames':r.get('decoded_frames',0),
           'reason_codes':';'.join(sorted({v['code'] for v in r['reasons']})),'video':r['video'],'hdf5':r['hdf5']} for r in reports]
    with (out/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'episodes':len(reports),'source_seconds':sum(r['duration_seconds'] for r in reports),
             'status_counts':{s:sum(r['status']==s for r in reports) for s in ['PASS','REVIEW','FAIL']},
             'reason_counts':{code:sum(any(v['code']==code for v in r['reasons']) for r in reports) for code in LABELS},
             'sample_hz':args.sample_hz,'full_video_decode':True,'decoded_frames':sum(r.get('decoded_frames',0) for r in reports),
             'motion_checks_enabled':False,'source_modified':False,'applied_trims':0,'applied_deletions':0,
             'hand_model':'OpenCV Zoo MediaPipe palm detector, 192x192 ONNX, no pose network',
             'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'exported_problem_samples':[r['id'] for r in views],
             'limitations':['2Hz抽样未检出不证明两点间始终不可见，手掌模型可能漏掉局部手指/遮挡的手',
                            '双手未检出只记REVIEW，待人工核实；正常单手操作不触发',
                            'MPEG4全程解码检查可发现解码错误，不等价于对所有画面内容正确性作保证',
                            '暂不做轨迹运动质检、裁剪、LeRobot转换或源文件删除']}
    dump(out/'summary.json',summary)
    cards=[]
    for r in sorted(reports,key=lambda r:(r['status']=='PASS',r['id'])):
        reasons=''.join(f'<li>{html.escape(v["severity"]+" / "+v["label"]+": "+v["detail"])}</li>' for v in r['reasons']) or '<li>未触发本次规则</li>'
        media=r['review_media'];images=''.join(f'<img src="{v}">' for v in media.get('snapshots',[]))
        clips=''.join(f'<p>{v["reason"]} {v["start_seconds"]:.2f}–{v["end_seconds"]:.2f}s，原速</p><video controls preload="none" src="{v["path"]}"></video>' for v in media.get('videos',[]))
        if 'overview' in media:clips+=f'<p>2Hz速览，2倍速播放</p><video controls preload="none" src="{media["overview"]}"></video>'
        missing=r.get('hand_detection',{}).get('long_missing_intervals',[])
        spans_text='；'.join(f'{v["start_seconds"]:.2f}–{v["end_seconds"]:.2f}s' for v in missing)
        cards.append(f'<section><h2>{html.escape(r["id"])} / {r["status"]}</h2><ul>{reasons}</ul><p>双手未检出候选区间：{spans_text or "无≥阈值区间"}</p>{images}{clips}'
                     f'<p><a href="episodes/{r["id"]}.json">完整指标/区间/耗时</a></p></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>EgoDex视频与手部出镜质检</title>'
        '<style>body{max-width:1100px;margin:24px auto;font:16px sans-serif;background:#f5f6f8}section{padding:20px;background:white;margin:20px 0}video,img{max-width:100%;width:100%}</style>'
        '<h1>EgoDex样本质检：2Hz画面/手掌检测</h1><p>文件完整性全程解码；未启用运动质检或裁剪。任意一只手被检出即视为有手。连续双手未检出≥3秒为待复查候选，不自动删除。</p>'
        '<p>框来自最近2Hz抽样点，不是逐帧检测；原速片段和2倍速速览分别标注。手掌模型漏检部分手指、遮挡或模糊手时也会产生候选。</p>'
        '<p><a href="episodes.csv">逐样本汇总</a> · <a href="samples.csv">逐抽样指标</a> · <a href="performance.md">速度报告</a></p>'
        '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    done=time.perf_counter();total=done-started;source_seconds=summary['source_seconds']
    cumulative={key:sum(r['timing'][key] for r in reports) for key in reports[0]['timing']}
    performance={'workers':args.workers,'source_seconds':source_seconds,'sample_count':len(sample_rows),'decoded_frames':summary['decoded_frames'],
                 'sample_hz':args.sample_hz,'trajectory_hours_per_wall_hour':source_seconds/total,'video_hours_per_wall_hour':source_seconds/total,
                 'timing':{'setup_seconds':setup_done-started,'full_integrity_quality_hands_seconds':analysis_done-setup_done,
                           'review_media_seconds':media_done-analysis_done,'reports_seconds':done-media_done,'total_seconds':total},
                 'worker_cumulative_seconds':cumulative,'conditions':'CPU；全程视频解码和HDF5全数据集读取；质量和手掌检测仅2Hz；含问题样本MP4/速览及报告；不含下载、Python启动和顶层导入；共享服务器，进程累计细项不等于墙钟阶段。'}
    dump(out/'performance.json',performance)
    (out/'performance.md').write_text('# EgoDex样本质检速度\n\n'+f'样本{len(reports)}条，视频共{source_seconds:.2f}秒，处理{total:.2f}秒，**{source_seconds/total:.2f}倍实时速度**。\n\n'+
        '\n'.join(f'- {k}: {v:.3f}秒' for k,v in performance['timing'].items())+'\n\n'+performance['conditions']+'\n')
    print(json.dumps({'summary':summary,'performance':performance},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
