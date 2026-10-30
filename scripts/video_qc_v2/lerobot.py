"""LeRobot v3适配：按回合时间区间读取共享长视频，绝不把整文件当作一个回合。"""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def discover_lerobot(root):
    records=[]
    for info_path in sorted(root.rglob('meta/info.json')):
        part=info_path.parent.parent;info=json.loads(info_path.read_text())
        if info.get('codebase_version')!='v3.0':raise ValueError(f'需要LeRobot v3.0: {part}')
        tasks=pq.read_table(part/'meta/tasks.parquet').to_pylist()
        task_ids={r['task']:int(r['task_index']) for r in tasks}
        video_keys=[k for k,v in info['features'].items() if v.get('dtype')=='video']
        prefix='__'.join(part.relative_to(root).parts) or 'dataset'
        part_rows=[]
        for meta in sorted((part/'meta/episodes').rglob('*.parquet')):
            pf=pq.ParquetFile(meta)
            for row in pf.read(columns=[c for c in pf.schema_arrow.names if not c.startswith('stats/')]).to_pylist():
                idx=int(row['episode_index']);length=int(row['length']);names=list(row['tasks'])
                data=part/info['data_path'].format(chunk_index=int(row['data/chunk_index']),file_index=int(row['data/file_index']))
                cameras={}
                for key in video_keys:
                    stem=f'videos/{key}'
                    video=part/info['video_path'].format(video_key=key,chunk_index=int(row[stem+'/chunk_index']),file_index=int(row[stem+'/file_index']))
                    cameras[key.removeprefix('observation.images.')]={
                        'path':str(video),'start_seconds':float(row[stem+'/from_timestamp']),
                        'end_seconds':float(row[stem+'/to_timestamp']),'expected_frames':length,
                        'fps':float(info['fps']),'timing_mode':'cfr',
                        # 夹爪/腕部运动是操作本身，不应据此判相机抖动。
                        # 仅已知头部视角执行LK；其他相机继续执行完整性与画质检查。
                        'motion_check':key in ('observation.images.head_main','observation.images.head_main_stereo_right')}
                record={'id':f'{prefix}__episode_{idx:06d}','task':f'task_{task_ids[names[0]]:04d}',
                    'task_description':names,'kind':'robot','cameras':cameras,'shared_clock':True,
                    'clock_basis':'导出视频按数据行重建的共同时间轴；不验证原始硬件采集同步',
                    'lerobot':{'data_path':str(data),'episode_index':idx,'length':length,'fps':float(info['fps']),
                        'from_index':int(row['dataset_from_index']),'to_index':int(row['dataset_to_index']),
                        'task_indices':[task_ids[n] for n in names]},
                    'metadata_paths':[str(info_path),str(meta),str(part/'meta/tasks.parquet'),str(data)]}
                part_rows.append(record)
        if len(part_rows)!=int(info['total_episodes']):raise ValueError(f'回合元数据数量不一致: {part}')
        records.extend(sorted(part_rows,key=lambda r:r['lerobot']['episode_index']))
    if not records:raise ValueError('未找到LeRobot v3回合元数据')
    return records


@lru_cache(maxsize=4)
def scalar_data(path):
    """每个工作进程每个数据分片只读一次标量列，避免每回合重复读取几十万行。

    不执行关节/动作合理性检查；这里只验证表结构、行序列和显式坏帧标记。
    """
    f=pq.ParquetFile(path);names=f.schema_arrow.names
    required=['episode_index','frame_index','timestamp','index','task_index','observation.state','action']
    missing=set(required)-set(names)
    if missing:raise ValueError(f'Parquet缺少必要字段: {sorted(missing)}')
    columns=required[:5]+(['valid.frame'] if 'valid.frame' in names else [])
    return f.read(columns=columns,use_threads=False)


def check_record(record):
    spec=record['lerobot'];table=scalar_data(spec['data_path'])
    rows=table.filter(pc.equal(table['episode_index'],spec['episode_index']))
    length=spec['length'];fps=spec['fps']
    if len(rows)!=length:raise ValueError(f'回合表行数={len(rows)}, metadata.length={length}')
    if spec['to_index']-spec['from_index']!=length:raise ValueError('回合索引边界与length不一致')
    for name in rows.column_names:
        if rows[name].null_count:raise ValueError(f'{name}包含空值')
    if not np.array_equal(rows['frame_index'].to_numpy(),np.arange(length)):raise ValueError('frame_index不连续')
    if not np.array_equal(rows['index'].to_numpy(),np.arange(spec['from_index'],spec['to_index'])):raise ValueError('全局index与回合元数据不一致')
    times=rows['timestamp'].to_numpy()
    if not np.isfinite(times).all() or not np.allclose(times,np.arange(length)/fps,atol=.002,rtol=0):raise ValueError('表timestamp缺失/非有限/与帧率不一致')
    if not np.isin(rows['task_index'].to_numpy(),spec['task_indices']).all():raise ValueError('task_index与任务元数据不一致')
    if 'valid.frame' in rows.column_names:
        invalid=np.flatnonzero(~rows['valid.frame'].to_numpy())
        if len(invalid):raise ValueError(f'valid.frame=false共{len(invalid)}帧；导出说明这些帧为无效源帧的黑帧占位，前几个索引={invalid[:10].tolist()}')
    return length
