#!/usr/bin/env python3
"""Use official LeRobot migration functions, preserving the input directory.

Run with the existing conda 'lerobot' environment. No Hub upload or download.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_DATASETS_OFFLINE','1')

import av
import numpy as np
import pyarrow.parquet as pq
from lerobot.scripts import convert_dataset_v21_to_v30 as official
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def concat_arrow_data(paths, root, chunk_idx, file_idx, image_keys):
    """Avoid the Pandas 3 / datasets Array2D extension concat incompatibility.

    Keep the source Arrow schema and HF feature metadata, including float32
    matrix features. Official convert_data still chooses shards and offsets.
    """
    if image_keys:
        raise ValueError('This compatibility shim supports video datasets, not embedded images')
    path=root/official.DEFAULT_DATA_PATH.format(chunk_index=chunk_idx,file_index=file_idx)
    path.parent.mkdir(parents=True,exist_ok=True)
    schema=pq.read_schema(paths[0])
    with pq.ParquetWriter(path,schema,compression='zstd') as writer:
        for source in paths:
            table=pq.read_table(source)
            if not table.schema.equals(schema,check_metadata=False):
                raise ValueError(f'Inconsistent schema: {source}')
            writer.write_table(table)


def validate(source, output):
    info=json.loads((output/'meta/info.json').read_text())
    original=json.loads((source/'meta/info.json').read_text())
    assert info['codebase_version']=='v3.0'
    for key in ['fps','total_frames','total_episodes','total_tasks','robot_type']:
        assert info[key]==original[key],key
    assert set(info['features'])==set(original['features'])
    for key,feature in info['features'].items():
        for attr in ['dtype','shape','names']:
            assert feature[attr]==original['features'][key][attr],(key,attr)
    # Only read small indexing columns; do not compare every state or decode every video.
    scalar=['index','episode_index','frame_index','timestamp','task_index']
    table=pq.read_table(output/'data',columns=scalar)
    assert len(table)==info['total_frames']
    idx=table['index'].to_numpy()
    epids=table['episode_index'].to_numpy()
    fi=table['frame_index'].to_numpy()
    ts=table['timestamp'].to_numpy()
    assert np.array_equal(idx,np.arange(len(table)))
    assert np.isfinite(ts).all() and np.all(abs(ts-fi/info['fps'])<1e-4)
    source_eps=jsonlines(source/'meta/episodes.jsonl')
    videos=[k for k,v in info['features'].items() if v['dtype']=='video']
    columns=['episode_index','length','tasks','action_config','dataset_from_index','dataset_to_index',
             'data/chunk_index','data/file_index']
    columns += [f'videos/{key}/{attr}' for key in videos for attr in
                ['chunk_index','file_index','from_timestamp','to_timestamp']]
    eps=pq.read_table(output/'meta/episodes',columns=columns).to_pylist()
    assert len(eps)==len(source_eps)
    video_groups={}
    for old,new in zip(source_eps,eps,strict=True):
        for key in ['episode_index','length','tasks','action_config']:
            assert new[key]==old[key],(new['episode_index'],key)
        mask=epids==new['episode_index']
        assert np.array_equal(fi[mask],np.arange(new['length']))
        assert np.array_equal(idx[mask],np.arange(new['dataset_from_index'],new['dataset_to_index']))
        data_path=output/info['data_path'].format(chunk_index=new['data/chunk_index'],file_index=new['data/file_index'])
        assert data_path.is_file()
        for key in videos:
            prefix=f'videos/{key}/'
            path=output/info['video_path'].format(video_key=key,chunk_index=new[prefix+'chunk_index'],file_index=new[prefix+'file_index'])
            previous=video_groups.get(path,0)
            assert abs(new[prefix+'from_timestamp']-previous/info['fps'])<1e-3
            video_groups[path]=previous+new['length']
            assert abs(new[prefix+'to_timestamp']-video_groups[path]/info['fps'])<1e-3
    for path,count in video_groups.items():
        with av.open(str(path)) as container:
            video=container.streams.video[0]
            assert video.frames==count,(path,video.frames,count)
            assert float(video.average_rate)==info['fps']
    tasks=jsonlines(source/'meta/tasks.jsonl')
    target_tasks=pq.read_table(output/'meta/tasks.parquet').to_pandas()
    assert {int(row.task_index):task for task,row in target_tasks.iterrows()}=={x['task_index']:x['task'] for x in tasks}
    print('Structural checks passed; sampling official reader...',flush=True)
    ds=LeRobotDataset('local/agibot-task-327',root=output,video_backend='pyav')
    assert len(ds)==info['total_frames']
    sampled=0
    for ep_id in sorted({0,len(eps)//2,len(eps)-1}):
        ep=eps[ep_id]
        start,end=ep['dataset_from_index'],ep['dataset_to_index']
        for i in sorted({start,(start+end)//2,end-1}):
            sample=ds[i]
            assert int(sample['index'])==i and int(sample['episode_index'])==ep_id
            assert sample['task'] in ep['tasks']
            for key,f in info['features'].items():
                if key in scalar:
                    continue
                expected=tuple(f['shape'])
                if f['dtype']=='video':
                    expected=(expected[2],expected[0],expected[1])
                assert tuple(sample[key].shape)==expected,(key,sample[key].shape)
            sampled+=1
        print(f'Official reader: episode {ep_id} first/middle/last OK',flush=True)
    report={'status':'passed','source':str(source),'method':'official LeRobot v2.1 to v3.0 migration functions',
        'episodes':len(eps),'frames':len(ds),'fps':info['fps'],'sampled_frames':sampled,
        'source_unchanged':True,'action_fields_preserved':True,'action_config_preserved':True,
        'video_processing':'stream copy; no re-encoding','full_video_decode':False,
        'compatibility_adjustment':'PyArrow concatenation instead of Pandas for Array2D fields',
        'versions':{name:importlib.metadata.version(name) for name in ['lerobot','datasets','pyarrow','av']}}
    (output/'qc').mkdir(exist_ok=True)
    (output/'qc/migration_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    source=args.source.resolve(); output=args.output.resolve()
    stage=output.with_name(output.name+'.inprogress')
    if output.exists() or stage.exists():
        raise FileExistsError('Output or .inprogress directory already exists; refusing to overwrite')
    official.validate_local_dataset_version(source)
    stage.mkdir(parents=True)
    print('Migrating metadata and Parquet using official functions...',flush=True)
    official.convert_info(source,stage,100,200)
    official.convert_tasks(source,stage)
    original_concat=official.concat_data_files
    try:
        official.concat_data_files=concat_arrow_data
        episodes=official.convert_data(source,stage,100)
    finally:
        official.concat_data_files=original_concat
    print('Joining original video streams without re-encoding...',flush=True)
    videos=official.convert_videos(source,stage,200)
    official.convert_episodes_metadata(source,stage,episodes,videos)
    report=validate(source,stage)
    stage.rename(output)
    print(f'DONE: {output}\n{json.dumps(report)}',flush=True)


if __name__=='__main__':
    main()
