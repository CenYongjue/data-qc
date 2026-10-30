#!/usr/bin/env python3
"""HiFi-UMI去掉压缩失真检测的独立测速入口，默认实际运行五分钟。

复用正式流程，不复制整套检测实现。单独生成配置，只关闭block_detection，
不修改正式配置文件。维持24进程、4Hz、仅头部LK、问题片段和速览导出。
依赖块分数的色度花屏确认分支随之标记SKIPPED，不能算作已检查通过。
五分钟截止后停止派发，完成在途回合，因此实际耗时可能略长于五分钟。
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from summarize_video_qc_run import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('datasets/hifi_umi'))
    p.add_argument('--output',type=Path,default=Path('lerobotv3_output/hifi_umi_no_block_5min'))
    p.add_argument('--cpu-workers',type=int,default=24)
    p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=a.output.resolve();scripts=Path(__file__).resolve().parent
    if out.exists() and not a.resume:raise FileExistsError('输出已存在；续跑请显式传--resume')
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads((scripts/'video_qc_v2_config.json').read_text());config['block_detection']=False
    config_path=out/'no_block_config.json';config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2))
    command=[sys.executable,'-u',str(scripts/'video_quality_pipeline_v2.py'),
             '--input',str(a.input.resolve()),'--output',str(out),'--resume',
             '--config',str(config_path),'--cpu-workers',str(a.cpu_workers),'--time-budget-seconds',str(a.seconds)]
    print(f'START no-block test: workers={a.cpu_workers}, budget={a.seconds}s, log={out / "full_run.log"}',flush=True)
    with (out/'full_run.log').open('a' if a.resume else 'w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise SystemExit(result.returncode)
    print(json.dumps(summarize(out),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
