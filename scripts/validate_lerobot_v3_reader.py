#!/usr/bin/env python3
"""Spot-check first/middle/last episodes using the official LeRobot API."""
import argparse
import json
import os
import importlib.metadata
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_DATASETS_OFFLINE','1')

import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--all-episodes',action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    print('Loading official LeRobotDataset...',flush=True)
    ds = LeRobotDataset('local/syndata-glove-rep-bg',root=root,video_backend='pyav')
    info = json.loads((root/'meta/info.json').read_text())
    episodes = pq.read_table(root/'meta/episodes',columns=[
        'episode_index','dataset_from_index','dataset_to_index','source_start_idx','length']).to_pylist()
    assert len(ds)==info['total_frames']
    checked = 0
    selected=episodes if args.all_episodes else [episodes[i] for i in sorted({0,len(episodes)//2,len(episodes)-1})]
    for ep in selected:
        start,end = ep['dataset_from_index'],ep['dataset_to_index']
        for i in sorted({start,(start+end)//2,end-1}):
            frame = ds[i]
            assert int(frame['index'])==i
            assert int(frame['episode_index'])==ep['episode_index']
            assert int(frame['frame_index'])==i-start
            assert int(frame['source_frame_index'])==ep['source_start_idx']+i-start
            assert abs(float(frame['timestamp'])-(i-start)/info['fps']) < 1e-4
            assert 'action' not in frame
            assert isinstance(frame['task'],str) and frame['task']
            for key,f in info['features'].items():
                if key.startswith('observation.'):
                    value = frame[key]
                    expected = tuple(f['shape'])
                    if f['dtype']=='video':
                        expected = (expected[2],expected[0],expected[1])
                    assert tuple(value.shape)==expected,(key,value.shape,expected)
                    assert torch.isfinite(value).all(),key
            checked += 1
        print(f"Read episode {ep['episode_index']}: first/middle/last OK",flush=True)
    result = {'status':'passed','api':'lerobot.datasets.LeRobotDataset','video_backend':'pyav',
              'total_frames':len(ds),'episodes':len(episodes),'sampled_episodes':len(selected),'sampled_frames':checked,
              'versions':{name:importlib.metadata.version(name) for name in
                          ('lerobot','datasets','pyarrow','av','torch')},
              'checks':['episode boundaries','RGB decode','all observation shapes','finite values',
                        'task lookup','source frame mapping','10 fps synthetic timestamps','no action']}
    (root/'qc/official_reader_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
