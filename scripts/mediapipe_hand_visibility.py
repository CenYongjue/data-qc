#!/usr/bin/env python3
"""用官方 MediaPipe Hand Landmarker 验证真人手可见性，输出标注视频和未检出区间。

运行环境：/data_all/cenyj/.venvs/mediapipe_hand_qc。
“未检出”是候选异常，不等于手确实出画；模型也会因模糊、遮挡或截断而漏检。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 官方 21 点拓扑；直接用 OpenCV 绘图，兼容不提供 legacy solutions 的新版 MediaPipe。
CONNECTIONS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
             (13,17),(0,17),(17,18),(18,19),(19,20)]


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def no_hand_intervals(samples,duration):
    """边界按检测抽样时间估计；最后一段以视频时长收尾，不按帧号假设 CFR。"""
    spans=[];start=None
    for sample in samples:
        if sample['hand_count']==0 and start is None:start=sample['seconds']
        elif sample['hand_count']>0 and start is not None:
            spans.append({'start_seconds':start,'end_seconds':sample['seconds'],
                          'duration_seconds':sample['seconds']-start});start=None
    if start is not None:
        spans.append({'start_seconds':start,'end_seconds':duration,'duration_seconds':duration-start})
    return spans


def annotate(frame,result,t):
    display=cv2.resize(frame,(960,540));canvas=np.zeros((588,960,3),np.uint8);canvas[:540]=display
    for i,landmarks in enumerate(result.hand_landmarks):
        color=(70,230,70) if i==0 else (50,170,255)
        points=[(round(p.x*960),round(p.y*540)) for p in landmarks]
        for a,b in CONNECTIONS:cv2.line(canvas,points[a],points[b],color,2,cv2.LINE_AA)
        for point in points:cv2.circle(canvas,point,3,(30,30,255),-1)
        xs=[p[0] for p in points];ys=[p[1] for p in points]
        cv2.rectangle(canvas,(max(0,min(xs)),max(0,min(ys))),(min(959,max(xs)),min(539,max(ys))),color,2)
        label=result.handedness[i][0]
        # score 是左右手分类置信度，不是手部存在概率，不能用来代替 presence 阈值。
        cv2.putText(canvas,f'{label.category_name} (handedness={label.score:.2f})',
                    (max(5,min(xs)),max(20,min(ys)-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    count=len(result.hand_landmarks)
    color=(70,230,70) if count else (50,80,255)
    cv2.putText(canvas,f't={t:.2f}s  HANDS={count}  '+('DETECTED' if count else 'NO HAND DETECTED'),
                (12,563),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
    cv2.putText(canvas,'10 Hz sampled preview; no detection may also mean blur / occlusion / partial hand',
                (12,582),cv2.FONT_HERSHEY_SIMPLEX,.43,(220,220,220),1)
    return canvas


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('models/mediapipe/hand_landmarker.task'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sample-hz',type=float,default=10.)
    parser.add_argument('--missing-seconds',type=float,default=2.)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.sample_hz<=0 or args.missing_seconds<=0:parser.error('Sampling rate and duration must be positive')
    args.output.mkdir(parents=True);cv2.setNumThreads(1)
    options=vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model),delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,num_hands=2,
        min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    started=time.perf_counter();samples=[];thumbs=[];next_sample=0.;next_thumb=0.;last_ms=-1;decoded=0
    movie=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
        '-f','rawvideo','-pix_fmt','bgr24','-s','960x588','-r',str(args.sample_hz),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-threads','2',
        '-movflags','+faststart',str(args.output/'annotated.mp4')],stdin=subprocess.PIPE)
    infer_seconds=0.
    try:
        with av.open(str(args.video)) as container,vision.HandLandmarker.create_from_options(options) as detector,\
             (args.output/'landmarks.jsonl').open('w') as records:
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            origin=float((stream.start_time or 0)*stream.time_base)
            duration=float(stream.duration*stream.time_base) if stream.duration is not None else container.duration/1e6
            for frame in container.decode(stream):
                decoded+=1
                if frame.pts is None:raise ValueError('Video frame has no PTS')
                t=float(frame.pts*frame.time_base)-origin
                if t+1e-6<next_sample:continue
                next_sample+=1/args.sample_hz
                # 使用真实 PTS；MediaPipe VIDEO 模式要求毫秒时间戳严格递增。
                stamp=max(last_ms+1,round(t*1000));last_ms=stamp
                bgr=frame.to_ndarray(format='bgr24')
                rgb=cv2.cvtColor(cv2.resize(bgr,(960,540)),cv2.COLOR_BGR2RGB)
                before=time.perf_counter()
                result=detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),stamp)
                infer_seconds+=time.perf_counter()-before
                hands=[]
                for i,points in enumerate(result.hand_landmarks):
                    xs=[float(p.x) for p in points];ys=[float(p.y) for p in points]
                    hands.append({'handedness':result.handedness[i][0].category_name,
                        'handedness_score':float(result.handedness[i][0].score),
                        'bbox_xyxy_normalized':[min(xs),min(ys),max(xs),max(ys)],
                        'landmarks':[{'x':float(p.x),'y':float(p.y),'z':float(p.z)} for p in points]})
                sample={'seconds':t,'source_frame_index':decoded-1,'timestamp_ms':stamp,'hand_count':len(hands),
                        'handedness':';'.join(h['handedness'] for h in hands)}
                samples.append(sample);records.write(json.dumps({**sample,'hands':hands})+'\n')
                canvas=annotate(bgr,result,t);movie.stdin.write(canvas.tobytes())
                if t>=next_thumb-1e-6:
                    thumbs.append(cv2.resize(canvas,(320,196)));next_thumb+=1
                if len(samples)%50==0:print(f'Detected through {t:.1f}/{duration:.1f}s ({len(samples)} samples)',flush=True)
    finally:
        movie.stdin.close()
        if movie.wait():raise RuntimeError('Annotated MP4 encoding failed')
    spans=no_hand_intervals(samples,duration)
    long_spans=[s for s in spans if s['duration_seconds']>=args.missing_seconds]
    present=sum(s['hand_count']>0 for s in samples);elapsed=time.perf_counter()-started
    summary={'video':str(args.video.resolve()),'duration_seconds':duration,'sample_hz':args.sample_hz,
        'sample_count':len(samples),'hand_detected_samples':present,'no_hand_samples':len(samples)-present,
        'detected_fraction':present/len(samples),'no_hand_intervals':spans,
        'long_no_hand_intervals':long_spans,'candidate_threshold_seconds':args.missing_seconds,
        'status':'REVIEW' if long_spans else 'PASS_SAMPLED_CHECK',
        'processing_seconds_including_decode_detection_and_preview':elapsed,'inference_seconds':infer_seconds,
        'realtime_factor':duration/elapsed,'gpu_used':False,'decoded_frames':decoded,
        'thresholds':{'detection':.5,'presence':.5,'tracking':.5,'num_hands':2},
        'mediapipe_version':importlib.metadata.version('mediapipe'),
        'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'limitations':['真人手模型，不能据此证明机器人夹爪检测有效',
          '未检出包含真实未出镜和模型漏检，需观看标注视频核对',
          '没有人工逐帧标注，不能报告准确率/召回率','标注视频为10Hz抽样预览、无音频，时间以画面PTS为准']}
    dump(args.output/'summary.json',summary)
    with (args.output/'samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    columns=4;sheet=np.zeros((((len(thumbs)+columns-1)//columns)*196,columns*320,3),np.uint8)
    for i,image in enumerate(thumbs):sheet[i//columns*196:(i//columns+1)*196,i%columns*320:(i%columns+1)*320]=image
    cv2.imwrite(str(args.output/'contact_sheet.jpg'),sheet)
    rows=''.join(f'<tr><td>{s["start_seconds"]:.2f}</td><td>{s["end_seconds"]:.2f}</td><td>{s["duration_seconds"]:.2f}</td></tr>' for s in long_spans)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MediaPipe 真人手可见性验证</title>'
        '<style>body{max-width:1050px;margin:25px auto;font:16px sans-serif}video,img{max-width:100%}td,th{padding:8px;border:1px solid #ddd}</style>'
        '<h1>MediaPipe 真人手可见性验证</h1><p>红色“NO HAND DETECTED”表示模型未检出，不直接证明没有手。'
        '模糊、遮挡、只露出部分手指也可能漏检。本次使用CPU，检测阈值均为0.5。</p>'
        '<video src="annotated.mp4" controls></video><h2>连续未检出候选区间（秒）</h2>'
        f'<table><tr><th>开始</th><th>结束</th><th>持续时间</th></tr>{rows}</table>'
        '<p><a href="samples.csv">逐抽样点CSV</a> · <a href="summary.json">详细报告及耗时</a></p>'
        '<img src="contact_sheet.jpg"><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
