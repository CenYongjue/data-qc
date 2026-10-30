#!/usr/bin/env python3
"""把当前视频流水线的单帧画质规则搬到图片上，便于用人工样本验证。

复用同一配置、PyAV灰度转换和FFmpeg signalstats，计算低纹理、边缘清晰度和曝光。
单张图片没有时间轴，结果仅表示单帧条件，不等同持续两秒的视频告警。
不检测物体类别，也不把“白色物体”直接当成曝光过度。
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import av
import cv2
import numpy as np
from video_qc_v2.operators import MetadataFilter
from video_qc_v2.worker import normalized_gray


def check(image_path,config,output):
    cv2.setNumThreads(1)
    image=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f'图片无法读取: {image_path}')
    frame=av.VideoFrame.from_ndarray(image,format='bgr24')
    frame.pts=0;frame.time_base=Fraction(1,25)
    # 与视频一致：先缩放为宽320的全范围灰度图，再计算梯度。
    gray=normalized_gray(frame,config['analysis_width'])
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    edges=cv2.magnitude(gx,gy)>config['edge_gradient']
    edge_fraction=float(edges.mean())
    sharp=float(np.abs(cv2.Laplacian(gray,cv2.CV_32F))[edges].mean()) if edge_fraction>=config['min_edge_fraction'] else None
    dark=float((gray<config['dark_pixel']).mean());bright=float((gray>config['bright_pixel']).mean())
    # 保持视频现有网格生成方式，不额外修改正式检测算法。
    tiles=[gray[y:y+max(1,len(gray)//4),x:x+max(1,gray.shape[1]//4)]
           for y in range(0,len(gray),max(1,len(gray)//4))
           for x in range(0,gray.shape[1],max(1,gray.shape[1]//4))]
    dark_tiles=float(np.mean([a.mean()<config['dark_mean'] for a in tiles]))
    bright_tiles=float(np.mean([a.mean()>config['bright_mean'] for a in tiles]))
    color=frame.reformat(width=gray.shape[1],height=gray.shape[0],format='yuvj444p')
    with tempfile.TemporaryDirectory() as tmp:
        filt=MetadataFilter(color,'signalstats','',Path(tmp)/'signal.txt')
        filt.push(color);row=filt.finish()[0]
    yavg=row['lavfi.signalstats.YAVG']
    under=yavg<config['dark_mean'] and dark>config['dark_fraction'] and dark_tiles>=config['exposure_tile_fraction']
    over=yavg>config['bright_mean'] and bright>config['bright_fraction'] and bright_tiles>=config['exposure_tile_fraction']
    result={
        'input':str(image_path.resolve()),'original_size':[image.shape[1],image.shape[0]],
        'analysis_size':[gray.shape[1],gray.shape[0]],
        'metrics':{'edge_fraction':edge_fraction,'sharpness':sharp,'YAVG':yavg,
                   'dark_fraction':dark,'bright_fraction':bright,
                   'dark_tile_fraction':dark_tiles,'bright_tile_fraction':bright_tiles},
        'single_frame_flags':{'low_texture':sharp is None,
                              'blur':sharp is not None and sharp<config['blur_score'],
                              'underexposed':under,'overexposed':over},
        'blur_scorable':sharp is not None,
        'thresholds':{k:config[k] for k in ['analysis_width','edge_gradient','min_edge_fraction','blur_score',
                       'dark_pixel','bright_pixel','dark_mean','bright_mean','dark_fraction','bright_fraction','exposure_tile_fraction']},
        'note':'low_texture指全画面边缘不足，不是低纹理物体面积。blur=false且blur_scorable=false代表无法判定，不代表清晰。单张图片不验证持续时间。',
    }
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    cv2.imwrite(str(output/'analysis_gray.png'),gray)
    cv2.imwrite(str(output/'edge_mask.png'),edges.astype(np.uint8)*255)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path)
    p.add_argument('--output',type=Path,default=Path('lerobotv3_output/image_qc_W'))
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('video_qc_v2_config.json'))
    args=p.parse_args()
    print(json.dumps(check(args.image,json.loads(args.config.read_text()),args.output),ensure_ascii=False,indent=2))
