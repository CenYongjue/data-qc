#!/usr/bin/env python3
"""Convert a SynData Zarr v3 volume to observation-only LeRobot v3.0.

Requires numpy, numcodecs, pyarrow, pandas, datasets, av and ffmpeg.
Reads the explicit bytes+zstd Zarr layout without Zarr's async runtime.
Fails closed on unsupported storage layouts; missing chunks are never filled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone

import av
import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numcodecs import Zstd


VIDEO = "observation.images.head_rgb"
BASE = {k: {"dtype": "float32" if k == "timestamp" else "int64",
            "shape": [1], "names": None}
        for k in ("timestamp", "frame_index", "episode_index", "index", "task_index")}


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def log(message):
    print(message, flush=True)


def extract(archive: Path):
    target = archive.with_suffix("")
    if target.is_dir():
        log(f"Using extracted volume: {target}")
        return target
    stage = archive.parent / (target.name + ".extracting")
    stage.mkdir(exist_ok=False)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in members:
            p = Path(m.name)
            if (p.is_absolute() or ".." in p.parts or not p.parts
                    or p.parts[0] != target.name or not (m.isfile() or m.isdir())):
                raise ValueError(f"Unsafe archive member: {m.name}")
        tf.extractall(stage, members=members, filter="data")
    (stage / target.name).rename(target)
    stage.rmdir()
    log(f"Extracted: {target}")
    return target


class Array:
    def __init__(self, path):
        self.path = path
        self.meta = json.loads((path / "zarr.json").read_text())
        m = self.meta
        if m.get("zarr_format") != 3 or m.get("node_type") != "array":
            raise ValueError(f"Unsupported array: {path}")
        self.shape = tuple(m["shape"])
        self.chunks = tuple(m["chunk_grid"]["configuration"]["chunk_shape"])
        if m["chunk_grid"]["name"] != "regular" or self.chunks[1:] != self.shape[1:]:
            raise ValueError(f"Only full spatial chunks supported: {path}")
        enc = m["chunk_key_encoding"]
        if enc["name"] != "default" or enc.get("configuration", {}).get("separator", "/") != "/":
            raise ValueError(f"Unsupported chunk keys: {path}")
        if [c["name"] for c in m["codecs"]] != ["bytes", "zstd"] or m.get("storage_transformers"):
            raise ValueError(f"Unsupported codecs: {path}")
        self.dtype = np.dtype(m["data_type"])
        if self.dtype.name not in ("float32", "uint8"):
            raise ValueError(f"Unsupported dtype: {path}")
        endian = m["codecs"][0].get("configuration", {}).get("endian", "little")
        self.dtype = self.dtype.newbyteorder("<" if endian == "little" else ">")
        self.count = (self.shape[0] + self.chunks[0] - 1) // self.chunks[0]
        self._cache = (-1, None)

    def chunk(self, i):
        if self._cache[0] == i:
            return self._cache[1]
        p = self.path / "c" / str(i)
        for _ in self.shape[1:]:
            p /= "0"
        raw = Zstd().decode(p.read_bytes())
        expected = int(np.prod(self.chunks)) * self.dtype.itemsize
        if len(raw) != expected:
            raise ValueError(f"Wrong decompressed size: {p}: {len(raw)} != {expected}")
        x = np.frombuffer(raw, dtype=self.dtype).reshape(self.chunks)
        x = x[:min(self.chunks[0], self.shape[0] - i * self.chunks[0])]
        self._cache = (i, x)
        return x

    def read(self, start, end):
        if not 0 <= start < end <= self.shape[0]:
            raise ValueError(f"Invalid range: {start}:{end}")
        pieces = []
        for i in range(start // self.chunks[0], (end - 1) // self.chunks[0] + 1):
            offset = i * self.chunks[0]
            pieces.append(self.chunk(i)[max(0, start-offset):min(end-offset, self.chunks[0])])
        return np.concatenate(pieces)


def inspect(arrays):
    lengths = {a.shape[0] for a in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Unequal modality lengths: {lengths}")
    n = lengths.pop()
    bad = np.zeros(n, dtype=bool)
    report = {}
    for name, a in arrays.items():
        result = {"shape": a.shape, "dtype": a.dtype.name, "chunks": a.count,
                  "unreadable_chunks": [], "nonfinite_frames": [], "zero_values": 0,
                  "negative_values": 0, "quaternion_norm_outliers": []}
        for i in range(a.count):
            start = i * a.chunks[0]
            try:
                x = a.chunk(i)
                if np.issubdtype(x.dtype, np.floating):
                    finite = np.isfinite(x).reshape(len(x), -1).all(axis=1)
                    ids = np.flatnonzero(~finite) + start
                    result["nonfinite_frames"].extend(ids.tolist())
                    bad[ids] = True
                    result["zero_values"] += int(np.count_nonzero(x == 0))
                    result["negative_values"] += int(np.count_nonzero(x < 0))
                if name.endswith("pose") and x.shape[1:] == (7,):
                    norm = np.linalg.norm(x[:, 3:], axis=1)
                    result["quaternion_norm_outliers"].extend((np.flatnonzero(abs(norm-1) > 1e-3)+start).tolist())
            except (OSError, ValueError, RuntimeError) as exc:
                bad[start:min(start+a.chunks[0], n)] = True
                result["unreadable_chunks"].append({"chunk": i, "error": str(exc)})
            if (i+1) % 32 == 0:
                log(f"QC {name}: checked {i+1}/{a.count} chunks")
        a._cache = (-1, None)
        result["zero_fraction"] = result["zero_values"] / int(np.prod(a.shape)) if a.dtype.kind == 'f' else None
        if a.dtype.kind != 'f':
            result['zero_values'] = None
        report[name] = result
        log(f"QC {name}: {a.count} chunks; read errors={len(result['unreadable_chunks'])}; nonfinite frames={len(result['nonfinite_frames'])}")
    return bad, report


def arrow_array(x, target):
    """Build Arrow nested arrays from contiguous NumPy buffers, avoiding pixel lists."""
    x = np.ascontiguousarray(x)
    if isinstance(target, pa.ExtensionType):
        storage = target.storage_type
        dims = x.shape[1:]
        arr = pa.array(x.reshape(-1))
        for width in reversed(dims):
            arr = pa.ListArray.from_arrays(np.arange(len(arr)//width + 1, dtype=np.int32)*width, arr)
        return pa.ExtensionArray.from_storage(target, arr.cast(storage))
    if pa.types.is_fixed_size_list(target):
        return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1), type=target.value_type), target.list_size)
    return pa.array(x, type=target)


class Stats:
    def __init__(self):
        self.values = {}

    def update(self, key, x, video=False):
        if video:
            # Exact uint8 histogram: same full-pixel population statistics without
            # materializing RGB float64 images or repeatedly scanning them.
            pixels = x.reshape(-1,3)
            hist = np.stack([np.bincount(pixels[:,c],minlength=256) for c in range(3)])
            levels = np.arange(256,dtype=np.float64)/255
            n = len(pixels)
            mean = (hist*levels).sum(axis=1)/n
            m2 = (hist*(levels[None,:]-mean[:,None])**2).sum(axis=1)
            low = np.array([levels[np.flatnonzero(h)[0]] for h in hist])
            high = np.array([levels[np.flatnonzero(h)[-1]] for h in hist])
        else:
            x = x.astype(np.float64)
            if x.ndim == 1:
                x = x[:, None]
            n = len(x)
            mean = x.mean(axis=0)
            m2 = ((x-mean)**2).sum(axis=0)
            low, high = x.min(axis=0), x.max(axis=0)
        self._merge(key,n,mean,m2,low,high,video)

    def merge(self, other):
        for key, (n,mean,m2,low,high,video) in other.values.items():
            self._merge(key,n,mean.copy(),m2.copy(),low.copy(),high.copy(),video)

    def _merge(self,key,n,mean,m2,low,high,video):
        if key not in self.values:
            self.values[key] = [n, mean, m2, low, high, video]
        else:
            old = self.values[key]
            delta = mean-old[1]
            total = old[0]+n
            old[2] += m2 + delta**2 * old[0]*n/total
            old[1] += delta*n/total
            old[0] = total
            old[3] = np.minimum(old[3], low)
            old[4] = np.maximum(old[4], high)

    def export(self, frame_count):
        out = {}
        for key, (n, mean, m2, low, high, video) in self.values.items():
            values = {"min": low, "max": high, "mean": mean, "std": np.sqrt(m2/n)}
            out[key] = {k: (v.reshape(3,1,1) if video else v).tolist() for k,v in values.items()}
            out[key]["count"] = [frame_count if video else n]
        return out


def features_for(arrays):
    features = dict(BASE)
    features["source_frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    for name, a in arrays.items():
        key = VIDEO if name == "head_rgb" else "observation."+name
        features[key] = {"dtype": "video" if name == "head_rgb" else a.dtype.name,
                         "shape": list(a.shape[1:]), "names": None}
        if name == "head_rgb":
            features[key]["names"] = ["height", "width", "channels"]
        elif name.endswith("wrist_pose"):
            features[key]["names"] = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        elif name.endswith("qpos"):
            features[key]["names"] = [f"{finger}_Joint{j}" for finger in
                ("Index", "Little", "Middle", "Ring", "Thumb") for j in range(1,7)]
    return features


def hf_schema(features):
    out = {}
    for key, f in features.items():
        shape = tuple(f["shape"])
        if f["dtype"] == "video":
            continue
        if shape == (1,):
            out[key] = datasets.Value(f["dtype"])
        elif len(shape) == 1:
            out[key] = datasets.Sequence(datasets.Value(f["dtype"]), length=shape[0])
        else:
            out[key] = getattr(datasets, f"Array{len(shape)}D")(shape=shape, dtype=f["dtype"])
    return datasets.Features(out).arrow_schema


class VideoWriter:
    def __init__(self, path, shape, fps):
        path.parent.mkdir(parents=True, exist_ok=True)
        h,w,c = shape
        if c != 3:
            raise ValueError("RGB must have 3 channels")
        self.path, self.frames = path, 0
        self.proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264rgb", "-crf", "0", "-preset", "fast", "-threads", "4",
            "-g", str(fps), "-bf", "0", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, x):
        self.proc.stdin.write(x.tobytes())
        self.frames += len(x)

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"Video encoding failed: {self.path}")


def convert(args):
    archive = args.source.resolve()
    source = extract(archive) if archive.suffix == ".tar" else archive
    subset = source.parents[2]
    rel = str(source.relative_to(subset))
    root_meta = json.loads((source / "zarr.json").read_text())
    if root_meta.get("zarr_format") != 3:
        raise ValueError("Only Zarr v3 is supported")
    group_meta = json.loads((source / "data/zarr.json").read_text())
    if group_meta.get("node_type") != "group":
        raise ValueError("Missing data group")
    arrays = {p.name: Array(p) for p in sorted((source / "data").iterdir()) if p.is_dir()}
    if set(root_meta["attributes"]["modalities"]) != set(arrays):
        raise ValueError("Modality manifest does not match arrays")
    if "head_rgb" not in arrays:
        raise ValueError("head_rgb required")
    index = pq.read_table(subset / "index/clips.parquet")
    clips = index.filter(pc.equal(index["rel_path"], rel)).to_pylist()
    clips.sort(key=lambda r: r["start_idx"])
    if not clips or len({r['clip_id'] for r in clips}) != len(clips):
        raise ValueError("Missing or duplicate clip index")
    annotations = {r['clip_id']: r['task_summary_en'] for r in
                   pq.read_table(subset / "annotations/clip_annotations.parquet").to_pylist()}
    tasks = {r['task_id']: r['task_name'] for r in json.loads((subset / "task.json").read_text())}
    out = args.output.resolve()
    stage = out.with_name(out.name + ".inprogress")
    if out.exists() or stage.exists():
        raise FileExistsError(f"Refusing to overwrite {out} or {stage}")
    stage.mkdir(parents=True)
    if getattr(args, 'reuse_qc_report', None):
        previous = json.loads(args.reuse_qc_report.read_text())
        if previous['source'] != str(source) or not previous['checks']['all_chunks_decoded']:
            raise ValueError('QC report source mismatch or incomplete scan')
        qc = previous['modalities']
        if set(qc) != set(arrays):
            raise ValueError('QC modalities changed')
        bad = np.zeros(arrays['head_rgb'].shape[0],dtype=bool)
        for name,a in arrays.items():
            if tuple(qc[name]['shape']) != a.shape or qc[name]['dtype'] != a.dtype.name:
                raise ValueError('QC schema changed')
            bad[qc[name]['nonfinite_frames']] = True
            for error in qc[name]['unreadable_chunks']:
                s=error['chunk']*a.chunks[0]
                bad[s:min(s+a.chunks[0],len(bad))]=True
        log('Reusing completed QC report for unchanged source')
    else:
        bad, qc = inspect(arrays)
    n_source = len(bad)
    accepted, rejected = [], []
    for r in clips:
        s,e,n = int(r['start_idx']), int(r['end_idx']), int(r['num_frames'])
        reason = None
        if e-s+1 != n or not 0 <= s <= e < n_source:
            reason = "invalid_inclusive_clip_bounds"
        elif r['task_key'] not in tasks or not annotations.get(r['clip_id']):
            reason = "missing_task_or_summary"
        elif bad[s:e+1].any():
            reason = "unreadable_or_nonfinite_required_modality"
        if reason:
            rejected.append({**r, "reason": reason})
        else:
            accepted.append(r)
    report = {"source": str(source), "source_frames": n_source, "fps": args.fps,
              "timestamp_source": "frame_index / fps; original capture timestamps unavailable",
              "modalities": qc, "input_clips": len(clips), "accepted_clips": len(accepted),
              "rejected_clips": rejected, "checks": {"all_chunks_decoded": True,
              "original_timestamp_anomalies": "not_testable_no_original_timestamps"}}
    dump(stage / "qc/report.json", report)
    if not accepted:
        raise ValueError("No valid clips; see QC report")
    features = features_for(arrays)
    schema = hf_schema(features)
    task_texts = list(dict.fromkeys(tasks[r['task_key']] for r in accepted))
    (stage/'meta').mkdir(exist_ok=True)
    pd.DataFrame({'task_index': range(len(task_texts))}, index=pd.Index(task_texts, name='task')).to_parquet(stage/'meta/tasks.parquet')
    data_writer = video_writer = meta_writer = None
    data_file = video_file = 0
    global_index = 0
    stats = Stats()
    for ep,r in enumerate(accepted):
        start,end = r['start_idx'],r['end_idx']+1
        n = end-start
        if data_writer is None:
            data_path = stage/f"data/chunk-{data_file//1000:03d}/file-{data_file%1000:03d}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_writer = pq.ParquetWriter(data_path, schema, compression='zstd', compression_level=3)
        if video_writer is None:
            video_path = stage/f"videos/{VIDEO}/chunk-{video_file//1000:03d}/file-{video_file%1000:03d}.mp4"
            video_writer = VideoWriter(video_path, arrays['head_rgb'].shape[1:], args.fps)
        video_start = video_writer.frames / args.fps
        epstats = Stats()
        task_idx = task_texts.index(tasks[r['task_key']])
        for s in range(start,end,args.batch_frames):
            e = min(s+args.batch_frames,end)
            length = e-s
            values = {'timestamp': np.arange(s-start,e-start,dtype=np.float32)/args.fps,
                      'frame_index': np.arange(s-start,e-start,dtype=np.int64),
                      'episode_index': np.full(length,ep,dtype=np.int64),
                      'index': np.arange(global_index+s-start,global_index+e-start,dtype=np.int64),
                      'task_index': np.full(length,task_idx,dtype=np.int64),
                      'source_frame_index': np.arange(s,e,dtype=np.int64)}
            for name,a in arrays.items():
                x = a.read(s,e)
                key = VIDEO if name == 'head_rgb' else 'observation.'+name
                if name == 'head_rgb':
                    video_writer.write(x)
                    epstats.update(key,x,video=True)
                else:
                    values[key] = x
            for key,x in values.items():
                epstats.update(key,x)
            table = pa.Table.from_arrays([arrow_array(values[f.name],f.type) for f in schema],schema=schema)
            data_writer.write_table(table, row_group_size=args.batch_frames)
        stats.merge(epstats)
        metadata = {'episode_index': ep, 'tasks': [tasks[r['task_key']]], 'length': n,
                    'dataset_from_index': global_index, 'dataset_to_index': global_index+n,
                    'data/chunk_index': data_file//1000, 'data/file_index': data_file%1000,
                    'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0,
                    f'videos/{VIDEO}/chunk_index': video_file//1000,
                    f'videos/{VIDEO}/file_index': video_file%1000,
                    f'videos/{VIDEO}/from_timestamp': video_start,
                    f'videos/{VIDEO}/to_timestamp': video_writer.frames/args.fps,
                    'source_clip_id': r['clip_id'], 'source_rel_path': rel,
                    'source_subset': subset.name, 'source_task_key': r['task_key'],
                    'source_start_idx': start, 'source_end_idx_inclusive': end-1,
                    'task_summary_en': annotations[r['clip_id']]}
        for key,st in epstats.export(n).items():
            for stat,value in st.items():
                metadata[f'stats/{key}/{stat}'] = value
        mt = pa.Table.from_pylist([metadata])
        if meta_writer is None:
            mp = stage/'meta/episodes/chunk-000/file-000.parquet'
            mp.parent.mkdir(parents=True,exist_ok=True)
            meta_writer = pq.ParquetWriter(mp,mt.schema,compression='zstd')
        meta_writer.write_table(mt)
        global_index += n
        log(f"Converted episode {ep+1}/{len(accepted)}: {r['clip_id']} ({n} frames), total={global_index}")
        if data_path.stat().st_size >= args.data_mb*1024**2:
            data_writer.close(); data_writer=None; data_file += 1
        if video_path.exists() and video_path.stat().st_size >= args.video_mb*1024**2:
            video_writer.close(); video_writer=None; video_file += 1
    if data_writer is not None:
        data_writer.close()
    if video_writer is not None:
        video_writer.close()
    meta_writer.close()
    first_video = next((stage/'videos').rglob('*.mp4'))
    with av.open(str(first_video)) as c:
        v = c.streams.video[0]
        features[VIDEO]['info'] = {'video.height':v.height,'video.width':v.width,'video.codec':v.codec_context.name,
            'video.pix_fmt':v.pix_fmt,'video.fps':args.fps,'video.channels':3,'video.is_depth_map':False,'has_audio':False}
    info = {'codebase_version':'v3.0','robot_type':None,'fps':args.fps,'total_episodes':len(accepted),
            'total_frames':global_index,'total_tasks':len(task_texts),'chunks_size':1000,
            'data_files_size_in_mb':args.data_mb,'video_files_size_in_mb':args.video_mb,
            'splits':{'train':f'0:{len(accepted)}'},'features':features,
            'data_path':'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path':'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    dump(stage/'meta/info.json',info)
    dump(stage/'meta/stats.json',stats.export(global_index))
    dump(stage/'meta/source.json',{'dataset':'PsiBotAI/SynData','license':'CC-BY-4.0',
        'url':'https://huggingface.co/datasets/PsiBotAI/SynData','subset':subset.name,'volume':str(source),
        'fps':args.fps,'end_idx_convention':'inclusive','action_policy':'none; observed states unchanged',
        'depth_unit':'mm','depth_invalid_value':0,'pose_position_unit':'not confirmed',
        'pose_coordinate_frame':'not confirmed','qpos_unit':'not confirmed',
        'rgb_encoding':'lossless libx264rgb CRF 0; original RGB preserved',
        'statistics':'population min/max/mean/std; all frames, depth zeros included; no quantiles',
        'created_at':datetime.now(timezone.utc).isoformat(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if getattr(args,'verification','standard') == 'full':
        log('Verifying all Parquet values and all decoded RGB frames against source...')
        verify(stage,source)
    else:
        verify_standard(stage)
    stage.rename(out)
    log(f'DONE: {out}')


def verify_standard(root):
    """Check schema, every scalar index/timestamp, episode offsets and video headers.

    Does not reread all depth pixels or decode all RGB frames. Official-reader
    spot checks are provided separately by validate_lerobot_v3_reader.py.
    """
    info = json.loads((root/'meta/info.json').read_text())
    assert info['codebase_version']=='v3.0' and info['fps']>0
    assert 'action' not in info['features']
    expected = hf_schema(info['features'])
    columns = ['episode_index','length','dataset_from_index','dataset_to_index',
        'data/chunk_index','data/file_index','source_start_idx','source_end_idx_inclusive',
        f'videos/{VIDEO}/chunk_index',f'videos/{VIDEO}/file_index',
        f'videos/{VIDEO}/from_timestamp',f'videos/{VIDEO}/to_timestamp']
    eps = pq.read_table(root/'meta/episodes',columns=columns).to_pylist()
    assert len(eps)==info['total_episodes']
    cursor=0
    for i,ep in enumerate(eps):
        assert ep['episode_index']==i and ep['dataset_from_index']==cursor
        assert ep['length']>0
        assert ep['dataset_to_index']==cursor+ep['length']
        assert ep['source_end_idx_inclusive']-ep['source_start_idx']+1==ep['length']
        cursor += ep['length']
    assert cursor==info['total_frames']
    task_table=pd.read_parquet(root/'meta/tasks.parquet')
    task_ids=set(task_table['task_index'].tolist())
    assert len(task_ids)==info['total_tasks']
    cursor=0
    for path in sorted((root/'data').rglob('*.parquet')):
        pf=pq.ParquetFile(path)
        assert pf.schema_arrow.equals(expected,check_metadata=False),f'Schema mismatch: {path}'
        t=pf.read(columns=list(BASE)+['source_frame_index'])
        idx=t['index'].to_numpy(); epids=t['episode_index'].to_numpy()
        fi=t['frame_index'].to_numpy(); ts=t['timestamp'].to_numpy()
        assert np.array_equal(idx,np.arange(cursor,cursor+len(t)))
        assert np.isfinite(ts).all() and np.all(abs(ts-fi/info['fps'])<1e-4)
        assert set(t['task_index'].to_pylist()).issubset(task_ids)
        ci=int(path.parent.name.split('-')[1]); fidx=int(path.stem.split('-')[1])
        for ep_id in np.unique(epids):
            ep=eps[int(ep_id)]; mask=epids==ep_id
            assert ep['data/chunk_index']==ci and ep['data/file_index']==fidx
            assert mask.sum()==ep['length']
            assert np.array_equal(fi[mask],np.arange(ep['length']))
            assert np.array_equal(idx[mask],np.arange(ep['dataset_from_index'],ep['dataset_to_index']))
            assert np.array_equal(t['source_frame_index'].to_numpy()[mask],
                                  np.arange(ep['source_start_idx'],ep['source_end_idx_inclusive']+1))
        cursor += len(t)
    assert cursor==info['total_frames']
    frames=0
    for path in sorted((root/'videos').rglob('*.mp4')):
        ci=int(path.parent.name.split('-')[1]); fi=int(path.stem.split('-')[1])
        rows=[e for e in eps if e[f'videos/{VIDEO}/chunk_index']==ci and e[f'videos/{VIDEO}/file_index']==fi]
        n=sum(e['length'] for e in rows)
        assert n>0
        offset=0
        for ep in rows:
            assert abs(ep[f'videos/{VIDEO}/from_timestamp']-offset/info['fps'])<1e-6
            offset+=ep['length']
            assert abs(ep[f'videos/{VIDEO}/to_timestamp']-offset/info['fps'])<1e-6
        with av.open(str(path)) as c:
            v=c.streams.video[0]
            assert v.frames==n and float(v.average_rate)==info['fps']
            assert [v.height,v.width,3]==info['features'][VIDEO]['shape']
            assert abs(float(v.duration*v.time_base)-n/info['fps'])<1e-4
        frames+=n
    assert frames==cursor
    dump(root/'qc/verification.json',{'status':'passed','mode':'standard','frames':cursor,'episodes':len(eps),
        'schema_checked':True,'all_scalar_indices_and_timestamps_checked':True,
        'episode_and_video_offsets_checked':True,'video_headers_frame_counts_and_durations_checked':True,
        'full_output_value_comparison':'not performed; omitted at user request',
        'action_present':False})
    log(f'Standard validation passed: {len(eps)} episodes, {cursor} frames')


def verify(root, source):
    info = json.loads((root/'meta/info.json').read_text())
    arrays = {p.name:Array(p) for p in sorted((source/'data').iterdir()) if p.is_dir()}
    meta_columns = ['episode_index','length','dataset_from_index','dataset_to_index','data/chunk_index','data/file_index',
                    'source_start_idx','source_end_idx_inclusive',f'videos/{VIDEO}/chunk_index',
                    f'videos/{VIDEO}/file_index',f'videos/{VIDEO}/from_timestamp',f'videos/{VIDEO}/to_timestamp']
    episodes = pq.read_table(root/'meta/episodes',columns=meta_columns).to_pylist()
    total = 0
    for path in sorted((root/'data').rglob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            table = pa.Table.from_batches([batch])
            ids = table['source_frame_index'].to_numpy()
            for field in table.column_names:
                if field.startswith('observation.'):
                    name = field.removeprefix('observation.')
                    arr = table[field].combine_chunks()
                    if isinstance(arr.type, pa.ExtensionType):
                        arr = arr.storage
                    while pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
                        arr = arr.flatten()
                    actual = arr.to_numpy(zero_copy_only=False).reshape((len(ids),)+arrays[name].shape[1:])
                    expected = np.stack([arrays[name].read(int(i),int(i)+1)[0] for i in ids])
                    if not np.array_equal(actual,expected):
                        raise AssertionError(f'Value mismatch: {path}, {field}, source {ids[0]}')
            for row in table.select(['index','episode_index','frame_index','timestamp','source_frame_index']).to_pylist():
                ep = episodes[row['episode_index']]
                assert row['index'] == total
                assert row['frame_index'] == total-ep['dataset_from_index']
                assert row['source_frame_index'] == ep['source_start_idx']+row['frame_index']
                assert abs(row['timestamp']-row['frame_index']/info['fps']) < 1e-4
                total += 1
        log(f'Verified Parquet: {path.name}, cumulative frames={total}')
    assert total == info['total_frames'] == sum(e['length'] for e in episodes)
    total_rgb = 0
    for path in sorted((root/'videos').rglob('*.mp4')):
        chunk = int(path.parent.name.split('-')[1]); file = int(path.stem.split('-')[1])
        eps = [e for e in episodes if e[f'videos/{VIDEO}/chunk_index']==chunk and e[f'videos/{VIDEO}/file_index']==file]
        source_ids = [i for e in eps for i in range(e['source_start_idx'],e['source_end_idx_inclusive']+1)]
        count = 0
        with av.open(str(path)) as c:
            for frame in c.decode(video=0):
                assert count < len(source_ids)
                assert abs(float(frame.pts*frame.time_base)-count/info['fps']) < 1e-4
                expected = arrays['head_rgb'].read(source_ids[count],source_ids[count]+1)[0]
                assert np.array_equal(frame.to_ndarray(format='rgb24'),expected), f'RGB mismatch {path} {count}'
                count += 1
        assert count == len(source_ids)
        total_rgb += count
        log(f'Verified lossless video: {path.name}, frames={count}')
    assert total_rgb == total
    dump(root/'qc/verification.json',{'status':'passed','frames':total,'episodes':len(episodes),
        'all_numeric_values_exact':True,'all_rgb_frames_exact':True,'all_video_pts_checked':True,
        'frame_indices_and_synthetic_timestamps_checked':True,'action_present':False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=int,default=10)
    p.add_argument('--batch-frames',type=int,default=8)
    p.add_argument('--data-mb',type=int,default=100)
    p.add_argument('--video-mb',type=int,default=200)
    p.add_argument('--reuse-qc-report',type=Path,help='Reuse completed QC only when source files are unchanged')
    p.add_argument('--verification',choices=['standard','full'],default='standard')
    p.add_argument('--finalize-existing',action='store_true',help='Validate completed .inprogress output and publish it without reconverting')
    args = p.parse_args()
    if min(args.fps,args.batch_frames,args.data_mb,args.video_mb) <= 0:
        p.error('FPS, batch size and shard limits must be positive')
    if args.finalize_existing:
        output=args.output.resolve()
        stage=output.with_name(output.name+'.inprogress')
        if output.exists():
            raise FileExistsError(output)
        if args.verification=='full':
            source=args.source.resolve()
            verify(stage,source.with_suffix('') if source.suffix=='.tar' else source)
        else:
            verify_standard(stage)
        stage.rename(output)
        log(f'DONE: {output}')
    else:
        convert(args)
