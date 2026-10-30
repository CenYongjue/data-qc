#!/usr/bin/env python3
"""同批EgoDex的YOLOv8n人手专用检测对照；完整性/画质规则与原试验相同。

模型每个工作进程只加载一次；分别记录并行加载热身墙钟时间和完整流程速度。
原Palm告警即使被YOLO消除也导出对照片段，不把对照样本伪装成新告警。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('YOLO_CONFIG_DIR',os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'yolo'))
os.environ.setdefault('MPLCONFIGDIR',os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'matplotlib'))
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import html
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import cv2
import egodex_video_qc as qc

DETECTOR=None


class YOLOHand:
    def __init__(self,model,size,confidence,device='cpu',half=False):
        self.model=model;self.size=size;self.confidence=confidence;self.device=device;self.half=half
    def warmup(self):pass  # 初始化阶段已经热身，逐视频复用，避免重复加载。
    def predict(self,image):
        result=self.model.predict(image,imgsz=self.size,device=self.device,quantize=16 if self.half else 32,conf=self.confidence,
                                  max_det=10,verbose=False,save=False)[0]
        return [{'xyxy':box.astype(float).tolist(),'score':float(score)} for box,score in
                zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy())]


def initialize(queue,weights,size,confidence,device,half):
    global DETECTOR
    started=time.perf_counter()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(1)
    if device!='cpu' and not torch.cuda.is_available():raise RuntimeError('Requested GPU unavailable; refusing CPU fallback')
    load_start=time.perf_counter();model=YOLO(weights)
    if model.task!='detect' or not all('hand' in str(v).lower() for v in model.names.values()):
        raise ValueError(f'Expected dedicated hand detector, got {model.task}/{model.names}')
    model.predict(np.zeros((1080,1920,3),np.uint8),imgsz=size,device=device,quantize=16 if half else 32,conf=confidence,verbose=False)
    if device!='cpu':torch.cuda.synchronize()
    # Ultralytics初始化CPU predictor会重设线程数，必须在热身后恢复限制。
    torch.set_num_threads(2)
    DETECTOR=YOLOHand(model,size,confidence,device,half)
    qc.PalmDetector=lambda *_:DETECTOR
    queue.put({'pid':os.getpid(),'load_start':load_start,'load_end':time.perf_counter(),
               'imports_and_initialization_seconds':time.perf_counter()-started,'classes':model.names,
               'cpu_threads':torch.get_num_threads(),'actual_device':str(model.predictor.device),
               'actual_dtype':str(next(model.predictor.model.backend.model.parameters()).dtype),
               'backend_fp16':model.predictor.model.fp16,
               'device_name':torch.cuda.get_device_name() if device!='cpu' else 'CPU','parameters':sum(v.numel() for v in model.model.parameters())})


def ready():return os.getpid()


def union_seconds(intervals):
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    return sum(b-a for a,b in merged)


def main():
    started=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=Path('lerobotv3_output/egodex_qc_2hz_pilot'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=Path('models/hand_presence/hand_yolov8n.pt'))
    p.add_argument('--workers',type=int,default=4,choices=[1,2,4]);p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--confidence',type=float,default=.25)
    p.add_argument('--device',default='cpu',help='cpu或CUDA索引，例如0；CPU解码器配置不变')
    p.add_argument('--half',action='store_true',help='CUDA FP16推理，不是INT8量化')
    args=p.parse_args()
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    elif os.environ.get('CUDA_VISIBLE_DEVICES')=='':os.environ.pop('CUDA_VISIBLE_DEVICES')
    if args.half and args.device=='cpu':p.error('--half requires a GPU')
    baseline=args.baseline.resolve();out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);selection=json.loads((baseline/'selection.json').read_text())
    config=json.loads((baseline/'config.json').read_text());config['hand_score_threshold']=args.confidence
    config['hand_model']='hand_yolov8n.pt';config['hand_imgsz']=args.imgsz
    qc.dump(out/'selection.json',selection);qc.dump(out/'config.json',config)
    previous={r['id']:r for path in (baseline/'episodes').glob('*.json') for r in [json.loads(path.read_text())]}
    ctx=mp.get_context('spawn');queue=ctx.Queue();startup=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=ctx,initializer=initialize,
                             initargs=(queue,str(args.weights.resolve()),args.imgsz,args.confidence,args.device,args.half)) as pool:
        futures=[pool.submit(ready) for _ in range(args.workers)]
        loading=[queue.get(timeout=120) for _ in range(args.workers)]
        for future in futures:future.result()
        initialized=time.perf_counter();print(f'{args.workers} models ready on {args.device}; {args.imgsz}px; classes={loading[0]["classes"]}',flush=True)
        reports=[]
        for report in pool.map(qc.check_one,[(v,str(args.weights.resolve()),config) for v in selection['files']]):
            report['hand_model']='YOLOv8n dedicated human-hand detector';reports.append(report)
            print(f'{report["id"]}: {report["status"]}',flush=True)
        checked=time.perf_counter()
        # 当前问题+旧Palm漏检候选，均导出，保留真实判定状态不造假。
        current=sorted([r for r in reports if r['reasons']],key=lambda r:(r['status']=='FAIL',len(r['reasons'])),reverse=True)[:6]
        selected={r['id']:r for r in current}
        for r in reports:
            if previous[r['id']]['status']!='PASS':selected[r['id']]=r
        jobs=[];view_types={}
        for name,r in selected.items():
            rendered=copy.deepcopy(r)
            if not rendered['reasons']:
                old=previous[name]['reasons'][0]
                rendered['reasons']=[{'code':'old_palm_candidate','severity':'REVIEW',
                                     'at_seconds':old['at_seconds'],'detail':'旧Palm候选的对照，不是YOLO当前告警','intervals':[]}]
                view_types[name]='旧Palm候选对照；YOLO当前未告警'
            else:view_types[name]='YOLO当前候选'
            jobs.append((rendered,str(out),config))
        views=list(pool.map(qc.export_review,jobs));view_by_id={r['id']:r for r in views}
    queue.close();queue.join_thread();media_done=time.perf_counter()
    comparisons=[];sample_rows=[]
    for r in reports:
        name=r['id'];old=previous[name];r['review_media']=view_by_id.get(name,{})
        r['view_type']=view_types.get(name,'')
        qc.dump(out/'episodes'/f'{name}.json',r)
        sample_rows.extend({'id':name,**s} for s in r['samples'])
        comparisons.append({'id':name,'palm_status':old['status'],'yolo_status':r['status'],
                            'palm_long_missing':old.get('hand_detection',{}).get('long_missing_intervals',[]),
                            'yolo_long_missing':r.get('hand_detection',{}).get('long_missing_intervals',[]),
                            'palm_samples_with_hand':old.get('hand_detection',{}).get('samples_with_any_hand',0),
                            'yolo_samples_with_hand':r.get('hand_detection',{}).get('samples_with_any_hand',0),
                            'sample_count':len(r['samples'])})
    columns=list(dict.fromkeys(k for row in sample_rows for k in row))
    with (out/'samples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(sample_rows)
    episode_rows=[{'id':r['id'],'status':r['status'],'reason_codes':';'.join(v['code'] for v in r['reasons'])} for r in reports]
    with (out/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(episode_rows[0]));w.writeheader();w.writerows(episode_rows)
    qc.dump(out/'comparison.json',comparisons)
    summary={'episodes':len(reports),'source_seconds':sum(r['duration_seconds'] for r in reports),
             'status_counts':{s:sum(r['status']==s for r in reports) for s in ['PASS','REVIEW','FAIL']},
             'reason_counts':{code:sum(any(a['code']==code for a in r['reasons']) for r in reports) for code in qc.LABELS},
             'sample_hz':config['sample_hz'],'sample_count':len(sample_rows),'full_video_decode':True,
             'model':'Bingsu/adetailer hand_yolov8n.pt','model_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
             'model_source':'https://huggingface.co/Bingsu/adetailer','imgsz':args.imgsz,'confidence':args.confidence,
             'device':args.device,'half':args.half,'actual_device':loading[0]['actual_device'],
             'actual_dtype':loading[0]['actual_dtype'],'device_name':loading[0]['device_name'],'workers':args.workers,'source_modified':False,'motion_checks_enabled':False,
             'limitations':['同样20条、2Hz及3秒阈值；不是人工标注的准确率评估','输入尺寸/权重/阈值同时变化，不能将效果差异归因于单一因素',
                            '未检出仍只是候选，模型误检为手也会掩盖真正缺手；旧候选对照片段不属于新告警']}
    qc.dump(out/'summary.json',summary)
    cards=[]
    for r in sorted(reports,key=lambda r:(r['id'] not in selected,r['status']=='PASS',r['id'])):
        name=r['id'];comparison=next(v for v in comparisons if v['id']==name)
        reasons=''.join(f'<li>{html.escape(v["severity"]+" / "+v["label"]+": "+v["detail"])}</li>' for v in r['reasons']) or '<li>YOLO未触发本次规则</li>'
        media=r['review_media'];images=''.join(f'<img src="{v}">' for v in media.get('snapshots',[]))
        clips=''.join(f'<p>{html.escape(v["reason"])} / 原速片段</p><video controls preload="none" src="{v["path"]}"></video>' for v in media.get('videos',[]))
        if 'overview' in media:clips+=f'<p>2Hz速览，2倍速播放</p><video controls preload="none" src="{media["overview"]}"></video>'
        cards.append(f'<section><h2>{html.escape(name)} / {r["status"]}</h2><p>{html.escape(r["view_type"])}</p><ul>{reasons}</ul>'
            f'<p>Palm→YOLO：有手抽样点 {comparison["palm_samples_with_hand"]}→{comparison["yolo_samples_with_hand"]} / {comparison["sample_count"]}</p>'
            f'<pre>{html.escape(json.dumps({"Palm缺手区间":comparison["palm_long_missing"],"YOLO缺手区间":comparison["yolo_long_missing"]},ensure_ascii=False,indent=2))}</pre>'
            f'{images}{clips}<p><a href="episodes/{name}.json">完整报告</a></p></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>EgoDex YOLO人手检测对照</title>'
        '<style>body{max-width:1100px;margin:24px auto;font:16px sans-serif;background:#f5f6f8}section{background:white;padding:20px;margin:24px 0}img,video{width:100%;max-width:100%}pre{white-space:pre-wrap}</style>'
        f'<h1>YOLOv8n人手专用权重：同批EgoDex对照</h1><p>CPU解码{args.workers}进程；推理设备{args.device}，FP16={args.half}；{args.imgsz}输入、置信度{args.confidence}。画质/手部2Hz，双手未检出3秒候选；完整性仍全程解码。不做运动检查或删除源文件。</p>'
        '<p>旧Palm候选即使被YOLO消除，也导出相同时刻用于对照，明确标注为对照而不是新告警。框来自最近2Hz抽样，并非逐帧模型输出。</p>'
        '<p><a href="performance.md">速度及排除模型加载后的速度</a> · <a href="comparison.json">逐样本对照</a> · <a href="samples.csv">逐抽样结果</a></p>'
        '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    done=time.perf_counter();total=done-started
    load_wall=union_seconds([(r['load_start'],r['load_end']) for r in loading])
    source=summary['source_seconds']
    performance={'source_seconds':source,'sample_count':len(sample_rows),'timing':{
        'worker_startup_imports_models_seconds':initialized-startup,'model_load_warmup_wall_seconds':load_wall,
        'integrity_quality_hands_seconds':checked-initialized,'review_media_seconds':media_done-checked,
        'report_seconds':done-media_done,'total_seconds':total},
        'video_hours_per_wall_hour':source/total,'excluding_model_load_warmup_seconds':total-load_wall,
        'excluding_model_load_warmup_factor':source/(total-load_wall),
        'excluding_entire_worker_startup_factor':source/(total-(initialized-startup)),
        'workers':args.workers,'model_loads':loading,
        'conditions':'模型每工作进程只加载一次；模型加载热身墙钟时间为各进程区间的并集，不是累加。总计含完整性全解码、2Hz质检/检测及复查输出；不含下载与主进程顶层导入。CPU解码四进程；实际推理设备和精度见summary.json。'}
    qc.dump(out/'performance.json',performance)
    (out/'performance.md').write_text('# YOLO同批EgoDex效率\n\n'+f'同批{len(reports)}条、{source:.2f}秒，完整执行{total:.2f}秒，**{source/total:.2f}倍**。\n\n'+
        f'模型加载及热身的并行墙钟时间{load_wall:.2f}秒，排除后{total-load_wall:.2f}秒，**{source/(total-load_wall):.2f}倍**。\n\n'+
        '\n'.join(f'- {key}: {value:.3f}秒' for key,value in performance['timing'].items())+'\n\n'+performance['conditions']+'\n')
    print(json.dumps({'summary':summary,'performance':performance,'baseline_review_comparison':[v for v in comparisons if v['palm_status']!='PASS']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
