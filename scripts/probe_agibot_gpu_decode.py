#!/usr/bin/env python3
"""只读检查GPU及AV1硬件解码；明确失败，不回退CPU冒充GPU成功。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import pyarrow.parquet as pq


def run(command):
    started=time.perf_counter()
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=30)
        return {'command':command,'returncode':p.returncode,'seconds':time.perf_counter()-started,
                'stdout':p.stdout[-8000:],'stderr':p.stderr[-8000:]}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'command':command,'returncode':None,'seconds':time.perf_counter()-started,'error':str(exc)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('lerobotv3_output/agibot'))
    p.add_argument('--h264-control',type=Path,help='可选H.264视频，区分AV1硬件限制与GPU环境故障')
    p.add_argument('--output',type=Path,default=Path('lerobotv3_output/agibot_gpu_decode_probe/sandbox.json'))
    args=p.parse_args();root=args.dataset.resolve();info=json.loads((root/'meta/info.json').read_text())
    ep=pq.read_table(root/'meta/episodes').to_pylist()[0];key='observation.images.head'
    video=root/info['video_path'].format(video_key=key,chunk_index=ep[f'videos/{key}/chunk_index'],file_index=ep[f'videos/{key}/file_index'])
    common=['ffmpeg','-hide_banner','-nostdin','-loglevel','info']
    # CPU与GPU都读取同一AV1源的前2秒，输出回CPU以贴近OpenCV的使用方式。
    cpu=common+['-threads','2','-c:v','libdav1d','-i',str(video),'-t','2','-an','-f','null','-']
    gpu=common+['-hwaccel','cuda','-hwaccel_output_format','cuda','-c:v','av1_cuvid','-i',str(video),
                '-t','2','-an','-vf','hwdownload,format=nv12','-f','null','-']
    result={'input':str(video),'test_seconds':2,'device_nodes':sorted(str(x) for x in Path('/dev').glob('nvidia*')),
            'dri_exists':Path('/dev/dri').exists(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'nvidia_smi':run(['nvidia-smi']),
            'source':run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,pix_fmt','-of','json',str(video)]),
            'cpu_decode':run(cpu),'gpu_decode':run(gpu)}
    if args.h264_control:
        result['h264_gpu_control']=run(common+['-hwaccel','cuda','-hwaccel_output_format','cuda','-c:v','h264_cuvid','-i',str(args.h264_control.resolve()),'-t','2','-an','-vf','hwdownload,format=nv12','-f','null','-'])
    result['gpu_decode_succeeded']=result['gpu_decode']['returncode']==0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(args.output),'gpu_decode_succeeded':result['gpu_decode_succeeded'],
                      'cpu_returncode':result['cpu_decode']['returncode'],
                      'gpu_stderr':result['gpu_decode'].get('stderr',result['gpu_decode'].get('error'))},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
