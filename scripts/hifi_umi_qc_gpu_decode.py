#!/usr/bin/env python3
"""HiFi-UMI五分钟NVDEC对照：24进程、无块检测、仅头部LK，保留复核视频。

PyAV HWAccel直接接入原解码位置，保留PTS和逐帧完整性检查。
硬解后返回CPU图像进行原有算子，传输和解码开销均计入总时间。
先检查少量帧的硬解状态、PTS和灰度差异，禁止静默CPU回退。
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import av
from av.codec.hwaccel import HWAccel
import numpy as np
from summarize_video_qc_run import summarize
from video_qc_v2.lerobot import discover_lerobot
from video_qc_v2.worker import normalized_gray


def preflight(path,device,width):
    def read(hw):
        kw={'hwaccel':HWAccel('cuda',device=str(device),allow_software_fallback=False)} if hw else {}
        started=time.perf_counter();frames=[]
        with av.open(path,**kw) as container:
            stream=container.streams.video[0];stream.codec_context.thread_count=2
            active=bool(stream.codec_context.is_hwaccel)
            if hw and not active:raise RuntimeError('NVDEC未启用')
            for frame in container.decode(stream):
                frames.append((frame.pts,normalized_gray(frame,width)))
                if len(frames)==8:break
        return frames,active,time.perf_counter()-started
    cpu,_,ct=read(False);gpu,active,gt=read(True)
    if [x[0] for x in cpu]!=[x[0] for x in gpu]:raise RuntimeError('CPU/GPU解码PTS不一致')
    differences=[np.abs(a[1].astype(float)-b[1].astype(float)) for a,b in zip(cpu,gpu)]
    mean=float(np.mean([d.mean() for d in differences]));peak=float(max(d.max() for d in differences))
    if mean>1:raise RuntimeError(f'灰度差异过大，拒绝直接使用原阈值: MAE={mean}')
    return {'device':device,'hardware_accelerated':active,'frames':len(gpu),'pts_match':True,
            'gray_mean_absolute_difference':mean,'gray_max_absolute_difference':peak,
            'cpu_short_probe_seconds':ct,'gpu_short_probe_seconds':gt,
            'note':'短探针只验证可用性/图像一致性，不用于推算整流程速度。'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('datasets/hifi_umi'))
    p.add_argument('--output',type=Path,default=Path('lerobotv3_output/hifi_umi_nvdec_5min'))
    p.add_argument('--gpus',default='2,3');p.add_argument('--cpu-workers',type=int,default=24)
    p.add_argument('--seconds',type=float,default=300);p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve();scripts=Path(__file__).resolve().parent
    if out.exists() and not a.resume:raise FileExistsError('输出已存在；续跑请传--resume')
    devices=[int(x) for x in a.gpus.split(',')]
    if not devices:raise ValueError('需要GPU设备')
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((scripts/'video_qc_v2_config.json').read_text())
    config.update(block_detection=False,decode_backend='cuda',decode_gpus=devices)
    records=discover_lerobot(a.input.resolve());path=next(iter(records[0]['cameras'].values()))['path']
    checks=[preflight(path,gpu,config['analysis_width']) for gpu in devices]
    (out/'nvdec_preflight.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    print('NVDEC_PREFLIGHT '+json.dumps(checks),flush=True)
    cp=out/'nvdec_config.json';cp.write_text(json.dumps(config,ensure_ascii=False,indent=2))
    command=[sys.executable,'-u',str(scripts/'video_quality_pipeline_v2.py'),'--input',str(a.input.resolve()),
             '--output',str(out),'--resume','--config',str(cp),'--cpu-workers',str(a.cpu_workers),'--time-budget-seconds',str(a.seconds)]
    with (out/'full_run.log').open('a' if a.resume else 'w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)
    print(json.dumps(summarize(out),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
