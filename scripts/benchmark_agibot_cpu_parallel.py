#!/usr/bin/env python3
"""依次实测1/2/4进程，保留每次报告；不同时运行不同配置以免相互争抢资源。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('lerobotv3_output/agibot'))
    parser.add_argument('--output',type=Path,default=Path('lerobotv3_output/agibot_qc_cpu_parallel'))
    parser.add_argument('--episodes',default='0,3,16,31')
    parser.add_argument('--workers',default='1,2,4')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for workers in [int(v) for v in args.workers.split(',')]:
        output=args.output/f'w{workers}'
        if output.exists():raise FileExistsError(output)
        command=[sys.executable,str(Path(__file__).with_name('agibot_quality_with_gripper.py')),
                 '--dataset',str(args.dataset),'--output',str(output),'--episodes',args.episodes,
                 '--no-gripper','--workers',str(workers)]
        started=time.perf_counter();subprocess.run(command,check=True)
        performance=json.loads((output/'performance.json').read_text())
        rows.append({'workers':workers,'process_wall_seconds_including_imports':time.perf_counter()-started,
                     'pipeline_wall_seconds':performance['timing']['total_seconds'],
                     'trajectory_speed':performance['trajectory_hours_per_wall_hour']})
    # 可以先单独运行串行，再用--workers 2,4；汇总所有已完成配置。
    by_workers={r['workers']:r for r in rows}
    for output in sorted(args.output.glob('w*')):
        if not (output/'performance.json').exists():continue
        p=json.loads((output/'performance.json').read_text());workers=p['workers']
        by_workers.setdefault(workers,{'workers':workers,'pipeline_wall_seconds':p['timing']['total_seconds'],
                                     'trajectory_speed':p['trajectory_hours_per_wall_hour']})
    comparison=sorted(by_workers.values(),key=lambda r:r['workers'])
    # 并行不应改变规则结论、指标或采样位置；逐轨迹JSON含所有数值结果。
    baseline=args.output/'w1';baseline_samples=(baseline/'video_samples.csv').read_bytes()
    for row in comparison:
        output=args.output/f'w{row["workers"]}'
        assert (output/'video_samples.csv').read_bytes()==baseline_samples,output
        for report in (baseline/'episodes').glob('*.json'):
            assert json.loads(report.read_text())==json.loads((output/'episodes'/report.name).read_text()),report
        assert json.loads((output/'calibration.json').read_text())==json.loads((baseline/'calibration.json').read_text())
    result={'episodes':args.episodes,'model_enabled':False,'comparison':comparison,
            'equivalent_samples_reports_and_calibration':True,
            'conditions':'共享服务器，依次执行；文件缓存可影响结果；子进程启动和退出包含在pipeline计时内，顶层导入仅计入process_wall。'}
    (args.output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
