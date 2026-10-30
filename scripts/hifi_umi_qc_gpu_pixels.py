#!/usr/bin/env python3
"""四卡GPU像素算子五分钟实验入口；结果不覆盖正式CPU流程。"""
import sys
from pathlib import Path
import runpy

if __name__=='__main__':
    sys.argv=[str(Path(__file__).with_name('hifi_umi_qc_persistent_gpu.py')),
              '--gpu-pixels','--output','lerobotv3_output/hifi_umi_gpu_pixels_4gpu_5min',*sys.argv[1:]]
    runpy.run_path(sys.argv[0],run_name='__main__')
