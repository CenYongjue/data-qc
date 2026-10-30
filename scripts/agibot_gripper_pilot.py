#!/usr/bin/env python3
"""YOLOE nano 夹爪存在检测小样本试验；只保存框/分数，不运行位姿网络。

先用参考图编码夹爪外观，再把编码加载到 detection-only 模型，移除分割分支。
测试图来自不同 episode；检出率不是准确率，没有人工真值时不自动剔除数据。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('YOLO_CONFIG_DIR',os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'yolo'))
os.environ.setdefault('MPLCONFIGDIR',os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'matplotlib'))

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLOE
import ultralytics

# 在 episode 0 的三张参考图上人工指定夹爪/夹指框，坐标对应原始 640x480。
# 腕部相机中夹爪分布在图像两侧；把两个可见夹指作为同一类别的外观示例。
PROMPTS={
 'observation.images.head':[[20,15,215,210]],
 'observation.images.hand_left':[[76,280,302,479],[423,280,639,479]],
 'observation.images.hand_right':[[59,248,271,479],[410,238,639,479]],
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,default=Path('lerobotv3_output/agibot_gripper_pilot'))
    parser.add_argument('--weights',type=Path,default=Path('models/agibot_gripper/yoloe-26n-seg.hf.pt'))
    parser.add_argument('--imgsz',type=int,default=320)
    parser.add_argument('--confidence',type=float,default=.25)
    args=parser.parse_args();out=args.samples.resolve()
    if (out/'summary.json').exists():raise FileExistsError('Refusing to overwrite completed pilot')
    torch.set_num_threads(2);torch.set_num_interop_threads(2);cv2.setNumThreads(1)
    samples=json.loads((out/'sample_manifest.json').read_text())
    (out/'detections').mkdir(exist_ok=True);(out/'reference_boxes').mkdir(exist_ok=True)
    rows=[];raw=[];camera_stats={};all_started=time.perf_counter();panels=[]
    for camera,boxes in PROMPTS.items():
        stem=camera.split('.')[-1]
        ref=next(s for s in samples if s['camera']==camera and s['role']=='reference')
        tests=[s for s in samples if s['camera']==camera and s['role'] in ('test','negative_control')]
        ref_image=cv2.imread(str(out/ref['image']));marked=ref_image.copy()
        for box in boxes:cv2.rectangle(marked,tuple(box[:2]),tuple(box[2:]),(0,255,255),2)
        cv2.imwrite(str(out/'reference_boxes'/f'{stem}.jpg'),marked)
        started=time.perf_counter()
        prompt_model=YOLOE(str(args.weights))
        prompt_model.predict(ref_image,refer_image=ref_image,visual_prompts={'bboxes':np.array(boxes,dtype=np.float32),
            'cls':np.zeros(len(boxes),dtype=np.int32)},imgsz=args.imgsz,device='cpu',
            conf=.01,verbose=False,save=False)
        profile=out/f'{stem}_prompt.npz';prompt_model.save_prompt_embeddings(profile)
        # 官方支持从同一套权重构建 detection-only 架构，再载入视觉提示编码。
        detector=YOLOE('yoloe-26n.yaml')
        # 下载的发布权重已经融合 Conv+BN。先将目标结构同样融合，避免仅加载少量
        # 同名张量、其余仍为随机初始化，却把零检出误认为模型效果差。
        detector.model.eval().fuse(verbose=False)
        source_weights=YOLOE(str(args.weights)).model.state_dict()
        target_weights=detector.model.state_dict()
        covered=sum(v.numel() for k,v in target_weights.items() if k in source_weights and source_weights[k].shape==v.shape)
        coverage=covered/sum(v.numel() for v in target_weights.values())
        if coverage<.99:raise RuntimeError(f'Insufficient pretrained weight coverage: {coverage:.2%}')
        detector.load(str(args.weights))
        detector.load_prompt_embeddings(profile)
        assert detector.task=='detect',detector.task
        setup_seconds=time.perf_counter()-started
        image=cv2.imread(str(out/tests[0]['image']))
        detector.predict(image,imgsz=args.imgsz,device='cpu',conf=.01,max_det=30,verbose=False,save=False)
        del prompt_model
        measured=0.;model_seconds=0.;detected_count=0
        for sample in tests:
            image=cv2.imread(str(out/sample['image']));started=time.perf_counter()
            result=detector.predict(image,imgsz=args.imgsz,device='cpu',conf=.01,
                max_det=30,verbose=False,save=False)[0]
            elapsed=time.perf_counter()-started;measured+=elapsed
            model_seconds+=result.speed['inference']/1000
            candidates=[]
            for box,confidence in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy()):
                candidates.append({'xyxy':box.astype(float).tolist(),'confidence':float(confidence)})
            accepted=[b for b in candidates if b['confidence']>=args.confidence]
            detected_count+=bool(accepted)
            row={**sample,'detected':bool(accepted),'box_count':len(accepted),
                 'max_score':max((b['confidence'] for b in candidates),default=0.),
                 'predict_seconds_including_pre_post':elapsed,'inference_ms':result.speed['inference']}
            rows.append(row);raw.append({**row,'candidates':candidates})
            marked=image.copy()
            for b in accepted:
                x1,y1,x2,y2=map(round,b['xyxy']);cv2.rectangle(marked,(x1,y1),(x2,y2),(0,220,0),2)
                cv2.putText(marked,f'gripper? {b["confidence"]:.2f}',(max(0,x1),max(16,y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX,.5,(0,220,0),1)
            text=f'ep={sample["episode_index"]} {stem} {sample["seconds"]}s det={len(accepted)} max={row["max_score"]:.2f}'
            panel=np.zeros((264,320,3),np.uint8);panel[:240]=cv2.resize(marked,(320,240))
            cv2.putText(panel,text,(3,256),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255),1)
            filename=Path(sample['image']).name;cv2.imwrite(str(out/'detections'/filename),marked);panels.append(panel)
            print(text,flush=True)
        camera_stats[camera]={'test_images':len(tests),'detected_images':detected_count,
            'pretrained_parameter_coverage':coverage,
            'reference_and_model_setup_seconds':setup_seconds,'warm_predict_seconds':measured,
            'mean_predict_ms':1000*measured/len(tests),'mean_inference_ms':1000*model_seconds/len(tests)}
        del detector
    elapsed=time.perf_counter()-all_started
    with (out/'samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'raw_detections.json').write_text(json.dumps(raw,indent=2))
    columns=3;sheet=np.zeros((((len(panels)+columns-1)//columns)*264,columns*320,3),np.uint8)
    for i,panel in enumerate(panels):sheet[i//columns*264:(i//columns+1)*264,i%columns*320:(i%columns+1)*320]=panel
    cv2.imwrite(str(out/'contact_sheet.jpg'),sheet)
    summary={'model':'YOLOE-26n detection-only with visual prompts','imgsz':args.imgsz,
        'confidence':args.confidence,'cpu_threads':2,'gpu_used':False,'mask_branch_removed':True,
        'test_images':len(rows),'detected_images':sum(r['detected'] for r in rows),
        'real_frame_count':sum(r['role']=='test' for r in rows),
        'real_frames_detected':sum(r['role']=='test' and r['detected'] for r in rows),
        'synthetic_negative_count':sum(r['role']=='negative_control' for r in rows),
        'synthetic_negative_detections':sum(r['role']=='negative_control' and r['detected'] for r in rows),
        'cameras':camera_stats,'total_seconds_including_model_setup_and_images':elapsed,
        'ultralytics_version':ultralytics.__version__,'torch_version':torch.__version__,
        'weights_sha256':hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        'limits':['Only 18 held-out frames and 3 synthetic background controls; no exhaustive negative evaluation',
                  'Detection frequency is not accuracy or recall; inspect boxes for false positives',
                  'This is cached-image inference, not end-to-end video cleaning throughput',
                  'Reference boxes contain visible gripper components; model may detect only fingers']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    images=''.join(f'<h3>{html.escape(camera)}</h3><img src="reference_boxes/{camera.split(".")[-1]}.jpg">' for camera in PROMPTS)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AgiBot gripper pilot</title>'
        '<style>body{max-width:1100px;margin:25px auto;font:16px sans-serif}img{max-width:100%}</style>'
        '<h1>AgiBot 夹爪存在检测试验</h1><p>三路独立参考；第0条作为外观示例，第16/31条测试。'
        '只用框和分数，不推理手部位姿，已移除分割分支。检出不等于检测正确，未检出也不证明夹爪出画。</p>'
        '<p><a href="samples.csv">逐图结果CSV</a> · <a href="raw_detections.json">全部候选框</a></p>'
        '<h2>测试结果</h2><img src="contact_sheet.jpg"><h2>参考框</h2>'+images+
        '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
