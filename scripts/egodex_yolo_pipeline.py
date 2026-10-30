#!/usr/bin/env python3
"""4个CPU解码进程 + 常驻双A100批量检测，队列解耦解码/推理。

与原EgoDex规则相同。只为真正告警生成原速证据与速览，去掉旧Palm对照。
队列有界并有独立结果接收线程，防止长视频积压或结果队列回压死锁。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('YOLO_CONFIG_DIR',os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'yolo'))
os.environ.setdefault('MPLCONFIGDIR',os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'matplotlib'))
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import html
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import threading
import time
import traceback
import cv2
import numpy as np
import egodex_video_qc as qc

CLIENT=None


class AsyncHands:
    def __init__(self,slot,requests,replies,imgsz,max_pending=64):
        self.slot=slot;self.requests=requests;self.replies=replies;self.imgsz=imgsz
        self.counter=0;self.inflight=0;self.completed={};self.condition=threading.Condition();self.error=None
        self.max_pending=max_pending
        threading.Thread(target=self.receive,daemon=True).start()
    def warmup(self):pass
    def receive(self):
        try:
            while True:
                token,boxes=self.replies.get()
                with self.condition:
                    self.completed[token]=boxes;self.inflight-=1;self.condition.notify_all()
        except Exception as exc:
            with self.condition:self.error=str(exc);self.condition.notify_all()
    def submit(self,image):
        h,w=image.shape[:2];ratio=min(1.,self.imgsz/max(h,w))
        # 与YOLO全图letterbox中的resize相同，提前缩小仅为降低进程间传输量。
        small=cv2.resize(image,(round(w*ratio),round(h*ratio)),interpolation=cv2.INTER_LINEAR) if ratio<1 else image
        with self.condition:
            if not self.condition.wait_for(lambda:self.inflight<self.max_pending or self.error,timeout=90):raise RuntimeError('GPU backpressure timeout')
            if self.error:raise RuntimeError(self.error)
            token=self.counter;self.counter+=1;self.inflight+=1
        self.requests.put((self.slot,token,small,w/small.shape[1],h/small.shape[0]),timeout=90)
        return token
    def resolve(self,token):
        with self.condition:
            if not self.condition.wait_for(lambda:token in self.completed or self.error,timeout=90):raise RuntimeError('GPU result timeout')
            if self.error:raise RuntimeError(self.error)
            return self.completed.pop(token)


def init_cpu(slots,requests,replies,imgsz):
    global CLIENT
    cv2.setNumThreads(1)
    slot=slots.get();CLIENT=AsyncHands(slot,requests,replies[slot],imgsz)
    qc.PalmDetector=lambda *_:CLIENT


def gpu_loop(device,weights,requests,replies,status,batch_size,flush_seconds,imgsz,confidence):
    try:
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(1)
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; refusing CPU fallback')
        started=time.perf_counter();model=YOLO(weights)
        if model.task!='detect' or not all('hand' in str(v).lower() for v in model.names.values()):raise ValueError('Not a human-hand detector')
        kwargs={'device':str(device),'imgsz':imgsz,'quantize':16,'conf':confidence,'max_det':10,'verbose':False,'save':False,'rect':True}
        model.predict([np.zeros((360,640,3),np.uint8) for _ in range(batch_size)],**kwargs)
        torch.cuda.synchronize();torch.set_num_threads(2)
        fp16=model.predictor.model.fp16
        if not fp16 or model.predictor.device.type!='cuda':raise RuntimeError('CUDA FP16 was not activated')
        status.put({'type':'ready','gpu':device,'seconds':time.perf_counter()-started,'actual_device':str(model.predictor.device),
                    'dtype':str(next(model.predictor.model.backend.model.parameters()).dtype),'name':torch.cuda.get_device_name(int(device))})
        histogram=Counter();frames=0;inference_seconds=0.;waiting_seconds=0.;stop=False
        while not stop:
            before=time.perf_counter();first=requests.get();waiting_seconds+=time.perf_counter()-before
            if first is None:break
            batch=[first];deadline=time.perf_counter()+flush_seconds
            while len(batch)<batch_size:
                remaining=deadline-time.perf_counter()
                if remaining<=0:break
                try:item=requests.get(timeout=remaining)
                except Empty:break
                if item is None:stop=True;break
                batch.append(item)
            before=time.perf_counter();results=model.predict([v[2] for v in batch],**kwargs)
            converted=[]
            for request,result in zip(batch,results,strict=True):
                slot,token,_,sx,sy=request;boxes=[]
                for box,score in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                    x1,y1,x2,y2=box
                    boxes.append({'xyxy':[float(x1*sx),float(y1*sy),float(x2*sx),float(y2*sy)],'score':float(score)})
                converted.append((slot,token,boxes))
            torch.cuda.synchronize();inference_seconds+=time.perf_counter()-before
            for slot,token,boxes in converted:replies[slot].put((token,boxes))
            frames+=len(batch);histogram[len(batch)]+=1
        status.put({'type':'done','gpu':device,'frames':frames,'batch_histogram':dict(histogram),
                    'inference_prepost_seconds':inference_seconds,'queue_wait_seconds':waiting_seconds})
    except Exception:
        status.put({'type':'error','gpu':device,'traceback':traceback.format_exc()})
        raise


def main():
    started=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('lerobotv3_output/egodex_qc_2hz_pilot'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=Path('models/hand_presence/hand_yolov8n.pt'))
    p.add_argument('--cpu-workers',type=int,default=4,choices=[4])
    p.add_argument('--gpus',default='0,1');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--flush-ms',type=float,default=30);p.add_argument('--queue-size',type=int,default=64)
    p.add_argument('--imgsz',type=int,default=640);p.add_argument('--confidence',type=float,default=.25)
    args=p.parse_args();baseline=args.baseline.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    if args.batch_size<1 or args.queue_size<args.batch_size or args.flush_ms<=0:p.error('Invalid batching/queue configuration')
    devices=[int(v) for v in args.gpus.split(',')]
    if not devices or len(devices)!=len(set(devices)):p.error('GPU indices must be distinct')
    if os.environ.get('CUDA_VISIBLE_DEVICES')=='':os.environ.pop('CUDA_VISIBLE_DEVICES')
    out.mkdir(parents=True);selection=json.loads((baseline/'selection.json').read_text());config=json.loads((baseline/'config.json').read_text())
    config.update(hand_score_threshold=args.confidence,hand_model='hand_yolov8n.pt',hand_imgsz=args.imgsz,
                  gpu_devices=devices,gpu_batch_size=args.batch_size,gpu_flush_ms=args.flush_ms,cpu_workers=4)
    qc.dump(out/'selection.json',selection);qc.dump(out/'config.json',config)
    ctx=mp.get_context('spawn');requests=ctx.Queue(maxsize=args.queue_size);replies=[ctx.Queue(maxsize=32) for _ in range(4)]
    status=ctx.Queue();slots=ctx.Queue()
    for index in range(4):slots.put(index)
    processes=[ctx.Process(target=gpu_loop,args=(gpu,str(args.weights.resolve()),requests,replies,status,args.batch_size,args.flush_ms/1000,args.imgsz,args.confidence)) for gpu in devices]
    startup=time.perf_counter()
    for process in processes:process.start()
    loading=[];gpu_stats=[]
    try:
        deadline=time.perf_counter()+120
        while len(loading)<len(devices):
            try:message=status.get(timeout=1)
            except Empty:
                if any(p.exitcode not in (None,0) for p in processes):raise RuntimeError('GPU process failed during startup')
                if time.perf_counter()>deadline:raise RuntimeError('GPU startup timeout')
                continue
            if message['type']=='error':raise RuntimeError(message['traceback'])
            loading.append(message)
        initialized=time.perf_counter();print(f'GPUs ready: {devices}; CPU decoders=4; batch<={args.batch_size}; FP16',flush=True)
        with ProcessPoolExecutor(max_workers=4,mp_context=ctx,initializer=init_cpu,initargs=(slots,requests,replies,args.imgsz)) as pool:
            reports=[]
            for r in pool.map(qc.check_one,[(v,str(args.weights.resolve()),config) for v in selection['files']]):
                reports.append(r);qc.dump(out/'episodes'/f'{r["id"]}.json',r)
                print(f'{r["id"]}: {r["status"]}',flush=True)
            checked=time.perf_counter()
            for _ in processes:requests.put(None)
            while len(gpu_stats)<len(devices):
                message=status.get(timeout=60)
                if message['type']=='error':raise RuntimeError(message['traceback'])
                if message['type']=='done':gpu_stats.append(message)
            for process in processes:
                process.join(timeout=30)
                if process.exitcode!=0:raise RuntimeError('GPU process did not exit cleanly')
            # 仅实际告警；不再导出旧Palm对照。继续复用原速问题片段+2Hz速览输出。
            issues=[r for r in reports if r['reasons']]
            views=list(pool.map(qc.export_review,[(r,str(out),config) for r in issues]))
        media_done=time.perf_counter()
    finally:
        for process in processes:
            if process.is_alive():process.terminate();process.join(timeout=5)
    view_by_id={v['id']:v for v in views};rows=[]
    for r in reports:
        r['review_media']=view_by_id.get(r['id'],{});r['hand_model']='YOLOv8n CUDA FP16 batched pipeline'
        qc.dump(out/'episodes'/f'{r["id"]}.json',r)
        rows.extend({'id':r['id'],**sample} for sample in r['samples'])
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'samples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    ep_rows=[{'id':r['id'],'status':r['status'],'reason_codes':';'.join(v['code'] for v in r['reasons']),
              'duration_seconds':r['duration_seconds'],'decoded_frames':r.get('decoded_frames',0),'review_video_count':len(r['review_media'].get('videos',[]))} for r in reports]
    with (out/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ep_rows[0]));w.writeheader();w.writerows(ep_rows)
    total_samples=sum(v['frames'] for v in gpu_stats)
    if total_samples!=len(rows):raise AssertionError(f'GPU/CPU sample mismatch {total_samples} vs {len(rows)}')
    summary={'episodes':len(reports),'source_seconds':sum(r['duration_seconds'] for r in reports),
             'status_counts':{s:sum(r['status']==s for r in reports) for s in ['PASS','REVIEW','FAIL']},
             'reason_counts':{code:sum(any(a['code']==code for a in r['reasons']) for r in reports) for code in qc.LABELS},
             'sample_hz':config['sample_hz'],'sample_count':len(rows),'cpu_decode_workers':4,'decoder_threads_per_video':2,
             'gpu_devices':devices,'dtype':'FP16','max_batch_size':args.batch_size,'gpu_samples_processed':total_samples,
             'full_video_decode':True,'decoded_frames':sum(r.get('decoded_frames',0) for r in reports),
             'exported_problem_samples':[r['id'] for r in views],'old_comparison_videos_exported':False,
             'source_modified':False,'motion_checks_enabled':False,'tensorrt_used':False}
    qc.dump(out/'summary.json',summary)
    cards=[]
    for r in sorted(reports,key=lambda v:(v['status']=='PASS',v['id'])):
        reasons=''.join(f'<li>{html.escape(v["severity"]+" / "+v["label"]+": "+v["detail"])}</li>' for v in r['reasons']) or '<li>通过本次已执行规则</li>'
        media=r['review_media'];videos=''.join(f'<p>{v["reason"]}，原速</p><video controls preload="none" src="{v["path"]}"></video>' for v in media.get('videos',[]))
        if 'overview' in media:videos+=f'<p>2Hz抽样速览，2倍速播放</p><video controls preload="none" src="{media["overview"]}"></video>'
        snapshots=''.join(f'<img src="{name}">' for name in media.get('snapshots',[]))
        cards.append(f'<section><h2>{html.escape(r["id"])} / {r["status"]}</h2><ul>{reasons}</ul>{snapshots}{videos}<p><a href="episodes/{r["id"]}.json">完整指标/区间</a></p></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>EgoDex双GPU流水质检</title>'
        '<style>body{max-width:1100px;margin:24px auto;font:16px sans-serif;background:#f5f6f8}section{padding:20px;background:white;margin:20px 0}video,img{max-width:100%;width:100%}pre{white-space:pre-wrap}</style>'
        '<h1>EgoDex：CPU解码/GPU批推理流水</h1><p>保持4个CPU解码进程，每视频最多2线程。两个GPU各常驻一个FP16模型，共享有界输入队列，跨视频组批，解码时不逐帧等待推理结果。</p>'
        '<p>2Hz质量/手部检测，双手未检出3秒为候选；完整性全程解码。只为真实告警输出复核视频，不输出旧模型对照。PASS仅表示已执行规则未触发，不代表模型无漏检。</p>'
        '<p><a href="performance.md">效率报告</a> · <a href="episodes.csv">样本汇总</a> · <a href="samples.csv">2Hz指标</a></p>'
        '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    done=time.perf_counter();total=done-started;source_seconds=summary['source_seconds']
    performance={'source_seconds':source_seconds,'video_hours_per_wall_hour':source_seconds/total,
                 'excluding_worker_startup_factor':source_seconds/(total-(initialized-startup)),
                 'timing':{'gpu_startup_import_load_warmup_seconds':initialized-startup,
                           'cpu_decode_gpu_batch_overlap_seconds':checked-initialized,
                           'gpu_shutdown_review_video_cpu_shutdown_seconds':media_done-checked,
                           'reports_seconds':done-media_done,'total_seconds':total},
                 'gpu_model_loading':loading,'gpu_batches':gpu_stats,
                 'cpu_decode_configuration_changed':False,'cpu_workers':4,'gpu_workers':len(devices),'queue_capacity_frames':args.queue_size,
                 'conditions':'同20条全解码、同2Hz、同640输入、同置信度0.25和3秒阈值；不同于上一轮，移除旧模型对照视频。总计包含GPU启动、流水、实际告警复核输出；不含主进程顶层导入。无单GPU同架构对照，不将全部收益归因于双GPU。'}
    qc.dump(out/'performance.json',performance)
    (out/'performance.md').write_text('# CPU/GPU流水和双GPU组批实测\n\n'+f'{source_seconds:.2f}秒视频，完整执行{total:.2f}秒，**{source_seconds/total:.2f}倍**。\n\n'+
        '\n'.join(f'- {key}: {value:.3f}秒' for key,value in performance['timing'].items())+'\n\n'+performance['conditions']+'\n')
    print(json.dumps({'summary':summary,'performance':performance},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
