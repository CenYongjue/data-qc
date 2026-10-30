#!/usr/bin/env python3
"""从已提交的SQLite记录汇总质检测速；不重新解码或推理，也不改动检测结果。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3


def summarize(out):
    db=sqlite3.connect(out/'results.sqlite')
    rows=[json.loads(value[0]) for value in db.execute('SELECT payload FROM records')]
    db.close()
    progress=json.loads((out/'progress.json').read_text())
    worker=sum(row['worker_seconds'] for row in rows)
    export=sum(view.get('export_seconds',0.) for row in rows for view in row['media'].values())
    timing=Counter()
    for row in rows:timing.update(row['timing'])
    # 多个CPU工作进程并行，累计阶段耗时不能直接从墙钟耗时里相减。
    # 用累计工作时间比例估算“去掉导出”的速度，明确区分估算和实际整链路测量。
    measured=progress['session_realtime_factor_including_startup']
    summary={
        'state':progress['state'],'completed_records':len(rows),'cpu_workers':progress['cpu_workers'],
        'tasks_covered':len({row['task'] for row in rows}),
        'source_minutes':sum(row['seconds'] for row in rows)/60,
        'wall_minutes':progress['session_wall_seconds']/60,
        'measured_realtime_factor_including_export':measured,
        'camera_source_minutes':progress.get('session_camera_seconds',0)/60,
        'measured_camera_realtime_factor_including_export':progress.get('session_camera_realtime_factor'),
        'estimated_realtime_factor_without_export':measured/(1-export/worker) if worker>export else None,
        'status_counts':dict(Counter(row['status'] for row in rows)),
        'issue_episode_counts_nonexclusive':dict(Counter(code for row in rows for code in {r['code'] for r in row['reasons'] if r['severity']!='WARN'})),
        'worker_accumulated_seconds':worker,'export_accumulated_seconds':export,
        'stage_accumulated_seconds':dict(timing),
        'stage_percent_of_worker_time':{key:100*value/worker for key,value in {**timing,'review_export':export}.items()} if worker else {},
        'problem_clips':sum(len(view.get('videos',[])) for row in rows for view in row['media'].values()),
        'overviews':sum(bool(view.get('overview')) for row in rows for view in row['media'].values()),
        'render_errors':sum(bool(view.get('render_error')) for row in rows for view in row['media'].values()),
        'note':'各问题可重叠；无导出倍速是按累计工作时间占比推算，不是单独实测。预算到时完成在途回合，实际时间可略超预算。',
    }
    (out/'benchmark_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    print(json.dumps(summarize(parser.parse_args().output),ensure_ascii=False,indent=2))
