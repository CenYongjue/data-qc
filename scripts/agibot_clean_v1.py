#!/usr/bin/env python3
"""AgiBot清洗第一版：双路抖动判定、首尾切分、AV1码流复制；源数据不修改。

复用已验证的基础质检算子。仅PASS导出训练数据，REVIEW保留报告待人工复核。
滤波只产生评分，不修改任何状态/动作。默认固定种子抽10条，CPU四进程。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import html
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import agibot_quality_with_gripper as core

core.LABELS['high_frequency_jitter']='高频抖动超限（滤波分析）'
MONITOR=[core.JOINT,core.END,'actions.joint.position','actions.end.position']


def high_frequency_features(values,scale,fps,config):
    """反射补边的频域低通：独立于原7帧移动平均残差法，分析持续高频能量。"""
    x=np.asarray(values,dtype=float).reshape(len(values),-1)/scale
    n=len(x);window=max(3,round(config['filter_window_seconds']*fps))
    if not np.isfinite(x).all() or n<max(window*2,8): return None
    pad=min(n-1,max(window,round(config['filter_padding_seconds']*fps)))
    padded=np.pad(x,((pad,pad),(0,0)),mode='reflect')
    spectrum=np.fft.rfft(padded,axis=0)
    # 平滑的Butterworth形状频率响应；离线对称滤波，不引入相位延迟。
    freq=np.fft.rfftfreq(len(padded),d=1/fps)
    response=1/np.sqrt(1+(freq/config['filter_cutoff_hz'])**(2*config['filter_order']))
    low=np.fft.irfft(spectrum*response[:,None],n=len(padded),axis=0)[pad:pad+n]
    high=x-low
    def average(a):
        return np.stack([np.convolve(np.pad(col,(window//2,window-1-window//2),mode='edge'),
                        np.ones(window)/window,mode='valid') for col in a.T],axis=1)
    high_power=average(high**2)
    variance=np.maximum(average(x*x)-average(x)**2,1e-12)
    rms=np.sqrt(np.maximum(high_power,0))
    ratio=high_power/np.maximum(variance,high_power+1e-12)
    # 补边不参与剔除判定；异常数值由基础算子单独报告。
    valid=np.ones(n,bool);border=min(n//3,max(window,round(config['filter_padding_seconds']*fps)))
    valid[:border]=False;valid[-border:]=False
    return {'rms':rms,'ratio':ratio,'valid':valid,'high_peak':np.max(abs(high),axis=0)}


def calibrate(numeric,episodes,fps,config):
    scales={};limits={}
    for key in set(MONITOR+[core.GRIP]):
        x=numeric[key].reshape(len(numeric[key]),-1).astype(float)
        x[~np.isfinite(x)]=np.nan
        lo,hi,width=core.robust_range(x);scales[key]=width
        alpha=config['quantile_margin_fraction']
        if isinstance(alpha,dict): alpha=alpha.get(key,alpha.get('default',.1))
        alpha=np.broadcast_to(np.asarray(alpha,dtype=float),lo.shape)
        if np.any(alpha<0):raise ValueError('Relative margin coefficients must be nonnegative')
        t=alpha*(hi-lo)  # 阈值严格采用实际分位差，不混入归一化防除零epsilon。
        limits[key]={'p01':lo.tolist(),'p99':hi.tolist(),'scale':width.tolist(),
                     'alpha':alpha.tolist(),'margin':t.tolist(),'lower':(lo-t).tolist(),'upper':(hi+t).tolist()}
    records={};jitters=[];populations={k:[] for k in MONITOR}
    for ep in episodes:
        a,b=ep['dataset_from_index'],ep['dataset_to_index'];x={k:v[a:b] for k,v in numeric.items()}
        jitter=core.jitter_features(x[core.END],scales[core.END],fps,config['smoothing_window_frames'])
        jitters.append(jitter)
        filtered={k:high_frequency_features(x[k],scales[k],fps,config) for k in MONITOR}
        records[ep['episode_index']]=(x,jitter,filtered)
        for key,result in filtered.items():
            if result is not None:populations[key].append(result['rms'][result['valid']])
    j=np.concatenate(jitters);median=np.nanmedian(j,axis=0)
    jthreshold=median+config['jitter_mad_multiplier']*np.maximum(1.4826*np.nanmedian(abs(j-median),axis=0),1e-6)
    filter_thresholds={}
    for key,arrays in populations.items():
        all_rms=np.concatenate(arrays);median=np.median(all_rms,axis=0)
        threshold=median+config['filter_mad_multiplier']*1.4826*np.median(abs(all_rms-median),axis=0)
        filter_thresholds[key]=np.maximum(threshold,config['filter_rms_floor']).tolist()
    return scales,limits,jthreshold,filter_thresholds,records


def hist_stats(hist,sample_count):
    levels=np.arange(256,dtype=float)/255;n=hist.sum(axis=1)
    mean=(hist*levels).sum(axis=1)/n
    std=np.sqrt((hist*(levels[None,:]-mean[:,None])**2).sum(axis=1)/n)
    out={'mean':mean,'std':std,'min':np.array([levels[np.flatnonzero(h)[0]] for h in hist]),
         'max':np.array([levels[np.flatnonzero(h)[-1]] for h in hist])}
    return {**{k:v.reshape(3,1,1).tolist() for k,v in out.items()},'count':[sample_count]}


def analyze(job):
    base,filtered,thresholds=job
    root,info,ep,x,jit,scales,limits,jthreshold,config,out,args=base
    trim,_=core.propose_trim(x,ep,scales,config,info['fps'])
    hist={k:np.zeros((3,256),np.int64) for k in core.CAMERAS};counts={k:0 for k in core.CAMERAS}
    paths={str(core.video_path(root,info,ep,k)):k for k in core.CAMERAS}
    original_reader=core.Reader
    class CapturingReader(original_reader):
        def frame(self,path,seconds,tolerance):
            image,pts=super().frame(path,seconds,tolerance)
            key=paths[str(path)];local=seconds-ep[f'videos/{key}/from_timestamp']
            if trim['start_frame']/info['fps']-1e-6<=local<(trim['end_frame_exclusive']-1)/info['fps']+1e-6:
                rgb=cv2.resize(image,(320,240))[:,:,::-1].reshape(-1,3)
                hist[key]+=np.stack([np.bincount(rgb[:,c],minlength=256) for c in range(3)])
                counts[key]+=1
            return image,pts
    core.Reader=CapturingReader
    try:r,samples,signature,hashed,timing=core.analyze_episode(base)
    finally:core.Reader=original_reader
    for reason in r['reasons']:
        if reason['code'] in ('jitter','no_motion'):reason['severity']='FAIL'
    metrics={}
    for key,result in filtered.items():
        if result is None:
            metrics[key]={'status':'unscorable'}
            core.add_reason(r,'high_frequency_jitter','REVIEW',f'{key}: 过短或非有限数值，无法分析')
            continue
        mask=((result['rms']>thresholds[key])&(result['ratio']>=config['filter_energy_ratio'])).any(axis=1)&result['valid']
        longest=core.max_run(mask,info['fps'])
        metrics[key]={'rms_thresholds':thresholds[key],'peak_rms_by_dimension':result['rms'][result['valid']].max(axis=0).tolist(),
                      'high_peak_by_dimension':result['high_peak'].tolist(),'bad_fraction':float(mask.mean()),
                      'longest_seconds':longest,'intervals':core.runs(mask,info['fps'])}
        if longest>=config['filter_run_seconds']:
            core.add_reason(r,'high_frequency_jitter','FAIL',f'{key}: 高频RMS与能量占比同时超限，最长{longest:.3f}秒',
                            np.flatnonzero(mask)[0]/info['fps'],intervals=core.runs(mask,info['fps']))
    r['metrics']['filter']=metrics;r['jitter_combination']='any';r['gripper_detection_enabled']=False
    r['image_statistics']={k:hist_stats(hist[k],counts[k]) for k in core.CAMERAS if counts[k]}
    return r,samples,signature,hashed,timing


def remux_segment(source,destination,start,end,fps):
    """复制AV1压缩包；保留必要关键帧前导，精确可见边界写入LeRobot时间偏移。"""
    destination.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(source)) as src, av.open(str(destination),'w') as dst:
        stream=src.streams.video[0];target=dst.add_stream_from_template(stream,opaque=True)
        src.seek(max(0,int(start/stream.time_base)),stream=stream,backward=True)
        origin=None;copied=0;last=None
        for packet in src.demux(stream):
            if packet.dts is None or packet.pts is None:continue
            if origin is None:
                if not packet.is_keyframe:continue
                origin=packet.dts
            # 当前AgiBot AV1数据无PTS/DTS重排。遇到重排拒绝导出，避免截断依赖帧。
            if packet.pts!=packet.dts:raise ValueError('Reordered AV1 packets require a different boundary exporter')
            if float(packet.pts*packet.time_base)>=end-1e-7:break
            last=float(packet.pts*packet.time_base)
            packet.pts-=origin;packet.dts-=origin;packet.stream=target;dst.mux(packet);copied+=1
        if not copied:raise ValueError('No video packets copied')
        offset=float(origin*stream.time_base)
    if abs((last+1/fps)-end)>1.1/fps:raise ValueError('Copied video end does not match requested interval')
    return {'from_timestamp':start-offset,'to_timestamp':end-offset,'copied_packets':copied,
            'source_packet_origin_seconds':offset,'preroll_seconds':start-offset}


def basic_stats(values):
    x=np.asarray(values,dtype=float)
    if x.ndim==1:x=x[:,None]
    return {**{k:v.tolist() for k,v in {'min':x.min(axis=0),'max':x.max(axis=0),
             'mean':x.mean(axis=0),'std':x.std(axis=0)}.items()},'count':[len(x)]}


def export_episode(root,info,ep,r,out,new_id,global_start):
    start=r['trim_proposal']['start_frame'];end=r['trim_proposal']['end_frame_exclusive'];n=end-start
    table=pq.read_table(root/'data').slice(ep['dataset_from_index']+start,n)
    scalar={'frame_index':np.arange(n,dtype=np.int64),'timestamp':np.arange(n,dtype=np.float32)/info['fps'],
            'episode_index':np.full(n,new_id,np.int64),'index':np.arange(global_start,global_start+n,dtype=np.int64)}
    for key,value in scalar.items():
        index=table.schema.get_field_index(key)
        table=table.set_column(index,table.schema.field(index),pa.array(value,type=table.schema.field(index).type))
    target=out/f'data/chunk-000/file-{new_id:03d}.parquet';target.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(table,target,compression='zstd')
    stats={key:basic_stats(core.numpy_column(table,key,f['shape'])) for key,f in info['features'].items() if f['dtype']!='video'}
    stats.update(r['image_statistics'])
    metadata={'episode_index':new_id,'length':n,'tasks':ep['tasks'],'dataset_from_index':global_start,
              'dataset_to_index':global_start+n,'data/chunk_index':0,'data/file_index':new_id,
              'meta/episodes/chunk_index':0,'meta/episodes/file_index':0,'source_episode_index':ep['episode_index'],
              'source_start_frame':start,'source_end_frame_exclusive':end,'action_config':[]}
    # AgiBot步骤结束边界语义尚未明确；沿用保守的inclusive解释，多保护一帧。
    for step in ep.get('action_config') or []:
        a=max(start,int(step['start_frame']));b=min(end,int(step['end_frame'])+1)
        if b>a:metadata['action_config'].append({**step,'start_frame':a-start,'end_frame':b-start-1})
    mapping={'source_episode_index':ep['episode_index'],'episode_index':new_id,'source_start_frame':start,
             'source_end_frame_exclusive':end,'retained_frames':n,'videos':{}}
    for key in core.CAMERAS:
        offset=ep[f'videos/{key}/from_timestamp'];path=out/f'videos/{key}/chunk-000/file-{new_id:03d}.mp4'
        video=remux_segment(core.video_path(root,info,ep,key),path,offset+start/info['fps'],offset+end/info['fps'],info['fps'])
        for name in ('from_timestamp','to_timestamp'):metadata[f'videos/{key}/{name}']=video[name]
        metadata[f'videos/{key}/chunk_index']=0;metadata[f'videos/{key}/file_index']=new_id
        mapping['videos'][key]=video
    for key,items in stats.items():
        for name,value in items.items():metadata[f'stats/{key}/{name}']=value
    return metadata,stats,mapping


def compare_trim(root,info,ep,r,out,config):
    """原速、同源帧边界对照：两条独立MP4不做拉伸或变速。"""
    result=[];fps=info['fps'];trim=r['trim_proposal'];n=ep['length'];reader=core.Reader()
    duration=round(config['comparison_seconds']*fps)
    boundaries=[]
    if trim['start_frame']>0:boundaries.append(('head',trim['start_frame']))
    if trim['end_frame_exclusive']<n:boundaries.append(('tail',trim['end_frame_exclusive']))
    try:
        for side,boundary in boundaries:
            if side=='head':
                before=(0,min(n,boundary+duration));after=(boundary,min(trim['end_frame_exclusive'],boundary+duration))
            else:
                before=(max(0,boundary-duration),n);after=(max(trim['start_frame'],boundary-duration),boundary)
            entry={'source_episode_index':ep['episode_index'],'side':side,'boundary_frame':boundary}
            for label,(a,b) in [('before',before),('after',after)]:
                relative=f'trim_comparisons/episode_{ep["episode_index"]:06d}_{side}_{label}.mp4'
                movie=core.Movie(out/relative,fps)
                try:
                    for fi in range(a,b):
                        frames=[reader.frame(core.video_path(root,info,ep,k),ep[f'videos/{k}/from_timestamp']+fi/fps,.5/fps)[0] for k in core.CAMERAS]
                        caption=f'{label.upper()} / {side} / source frame={fi} / boundary={boundary}'
                        movie.write(core.montage(frames,ep['episode_index'],fi/fps,caption))
                finally:movie.close()
                entry[label]={'path':relative,'source_start_frame':a,'source_end_frame_exclusive':b,'playback_fps':fps}
            result.append(entry)
    finally:reader.close()
    return result


def finalize(job):
    root,info,ep,r,report_out,dataset_out,config,new_id,global_start,comparison=job
    cv2.setNumThreads(1);reader=core.Reader()
    try:core.evidence(root,info,ep,r,report_out,reader,config)
    finally:reader.close()
    exported=export_episode(root,info,ep,r,dataset_out,new_id,global_start) if new_id is not None else None
    comparisons=compare_trim(root,info,ep,r,report_out,config) if comparison else []
    r['exported_episode_index']=new_id;r['trim_comparisons']=comparisons
    core.dump(report_out/f'episodes/episode_{r["episode_index"]:06d}.json',r)
    return r,exported,comparisons


def aggregate_stats(stats_list):
    result={}
    for key in stats_list[0]:
        values=[s[key] for s in stats_list];weights=np.array([s['count'][0] for s in values],dtype=float)
        shape=(len(weights),)+(1,)*(np.asarray(values[0]['mean']).ndim)
        w=weights.reshape(shape);means=np.array([s['mean'] for s in values]);mean=(means*w).sum(axis=0)/weights.sum()
        variance=(w*(np.array([s['std'] for s in values])**2+(means-mean)**2)).sum(axis=0)/weights.sum()
        result[key]={'mean':mean.tolist(),'std':np.sqrt(variance).tolist(),'min':np.min([s['min'] for s in values],axis=0).tolist(),
                     'max':np.max([s['max'] for s in values],axis=0).tolist(),'count':[int(weights.sum())]}
    return result


def validate_output(root,source,original_eps,mappings):
    if not mappings:return {'status':'empty','reason':'No PASS episodes; no training dataset exported'}
    info=json.loads((root/'meta/info.json').read_text());table=pq.read_table(root/'data')
    eps=pq.read_table(root/'meta/episodes').to_pylist()
    assert len(table)==info['total_frames'] and len(eps)==info['total_episodes']
    assert np.array_equal(table['index'].to_numpy(),np.arange(len(table)))
    for ep in eps:
        a,b=ep['dataset_from_index'],ep['dataset_to_index'];rows=table.slice(a,b-a)
        assert np.array_equal(rows['frame_index'].to_numpy(),np.arange(b-a))
        assert np.all(abs(rows['timestamp'].to_numpy()-np.arange(b-a)/info['fps'])<1e-4)
        assert (rows['episode_index'].to_numpy()==ep['episode_index']).all()
        for key in core.CAMERAS:
            path=core.video_path(root,info,ep,key)
            with av.open(str(path)) as c:
                assert c.streams.video[0].codec_context.name in ('libdav1d','av1')
    # 首末保留轨迹的首尾帧与源视频逐像素一致，验证无损码流复制及前导帧偏移。
    reader=core.Reader();original_info=json.loads((source/'meta/info.json').read_text());checked=0
    try:
        for ep in [eps[0],eps[-1]]:
            old=original_eps[ep['source_episode_index']]
            for fi in {0,ep['length']-1}:
                for key in core.CAMERAS:
                    actual,_=reader.frame(core.video_path(root,info,ep,key),ep[f'videos/{key}/from_timestamp']+fi/info['fps'],.5/info['fps'])
                    expected,_=reader.frame(core.video_path(source,original_info,old,key),old[f'videos/{key}/from_timestamp']+(ep['source_start_frame']+fi)/info['fps'],.5/info['fps'])
                    assert np.array_equal(actual,expected),(ep['episode_index'],key,fi)
                    checked+=1
    finally:reader.close()
    return {'status':'passed','source_output_pixel_matches':checked,'numeric_indices_timestamps':'passed','video_encoding':'original AV1 stream copy'}


def main():
    started=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('lerobotv3_output/agibot'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('agibot_clean_v1_config.json'))
    p.add_argument('--count',type=int,default=10);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--workers',type=int,default=4,choices=[1,2,4]);p.add_argument('--episodes',help='显式指定ID，覆盖随机抽样')
    args=p.parse_args();root=args.dataset.resolve();out=args.output.resolve();report_out=out/'report';dataset_out=out/'dataset'
    if out.exists():raise FileExistsError(out)
    config=json.loads(args.config.read_text());info=json.loads((root/'meta/info.json').read_text());fps=info['fps']
    if info['codebase_version']!='v3.0':raise ValueError('Expected v3.0')
    if not 0<config['filter_cutoff_hz']<fps/2:raise ValueError('Filter cutoff must be below Nyquist frequency')
    report_out.mkdir(parents=True);core.dump(report_out/'config.json',config)
    all_eps=pq.read_table(root/'meta/episodes').to_pylist();by_id={ep['episode_index']:ep for ep in all_eps}
    chosen=sorted(int(v) for v in args.episodes.split(',')) if args.episodes else sorted(np.random.default_rng(args.seed).choice(sorted(by_id),args.count,replace=False).tolist())
    if len(chosen)!=len(set(chosen)) or not set(chosen)<=set(by_id):raise ValueError('Invalid episode selection')
    episodes=[by_id[i] for i in chosen];core.dump(out/'selection.json',{'seed':args.seed,'episode_ids':chosen})
    table=pq.read_table(root/'data');numeric={k:core.numpy_column(table,k,f['shape']) for k,f in info['features'].items() if f['dtype']!='video'}
    scales,limits,jthreshold,filter_thresholds,records=calibrate(numeric,all_eps,fps,config)
    core.dump(report_out/'calibration.json',{'calibration_episodes':len(all_eps),'joint_ranges':limits,
        'residual_thresholds':jthreshold.tolist(),'filter_rms_thresholds':filter_thresholds,'filter_config':{k:v for k,v in config.items() if k.startswith('filter_')},'jitter_combination':'any'})
    print(f'Selected {chosen}; calibration complete; CPU workers={args.workers}',flush=True)
    setup_done=time.perf_counter();jobs=[]
    for ep in episodes:
        x,jitter,filtered=records[ep['episode_index']]
        base=(root,info,ep,x,jitter,scales,limits,jthreshold,config,report_out,SimpleNamespace(absence_seconds=3.,confidence=.25))
        jobs.append((base,filtered,filter_thresholds))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
        analyzed=list(pool.map(analyze,jobs));analysis_done=time.perf_counter()
        print('Quality analysis complete; exporting PASS episodes and review media...',flush=True)
        reports=[v[0] for v in analyzed];samples=[sample for v in analyzed for sample in v[1]]
        for i,r in enumerate(reports):
            for j in range(i):
                if analyzed[i][3]==analyzed[j][3]:core.add_reason(r,'exact_duplicate','FAIL',f'与episode {reports[j]["episode_index"]}完全重复')
                elif np.sqrt(np.mean((analyzed[i][2]-analyzed[j][2])**2))<config['near_duplicate_distance']:
                    core.add_reason(r,'near_duplicate','REVIEW',f'与episode {reports[j]["episode_index"]}状态动作近重复')
            severity=[reason['severity'] for reason in r['reasons']]
            r['status']='FAIL' if 'FAIL' in severity else 'REVIEW' if 'REVIEW' in severity else 'PASS'
            r['suggested_operation']='exclude_whole_episode' if r['status']=='FAIL' else 'hold_for_review' if r['status']=='REVIEW' else 'trim_and_export' if r['trim_proposal']['changed'] else 'export'
        accepted=[r for r in reports if r['status']=='PASS'];assignments={};offset=0
        for new_id,r in enumerate(accepted):assignments[r['episode_index']]=(new_id,offset);offset+=r['trim_proposal']['end_frame_exclusive']-r['trim_proposal']['start_frame']
        compare_ids={r['episode_index'] for r in accepted if r['trim_proposal']['changed']}
        compare_ids=set(sorted(compare_ids)[:config['comparison_max_episodes']])
        finish_jobs=[(root,info,ep,r,report_out,dataset_out,config,*assignments.get(ep['episode_index'],(None,None)),ep['episode_index'] in compare_ids) for ep,r in zip(episodes,reports)]
        finalized=list(pool.map(finalize,finish_jobs))
    media_done=time.perf_counter();reports=[v[0] for v in finalized];exports=[v[1] for v in finalized if v[1] is not None]
    comparisons=[c for v in finalized for c in v[2]];mappings=[v[2] for v in exports]
    if exports:
        target=dataset_out/'meta/episodes/chunk-000/file-000.parquet';target.parent.mkdir(parents=True,exist_ok=True)
        pq.write_table(pa.Table.from_pylist([v[0] for v in exports]),target,compression='zstd')
        shutil.copy2(root/'meta/tasks.parquet',dataset_out/'meta/tasks.parquet')
        new_info={**info,'total_episodes':len(exports),'total_frames':sum(v[0]['length'] for v in exports),
                  'splits':{'train':f'0:{len(exports)}'}}
        core.dump(dataset_out/'meta/info.json',new_info);core.dump(dataset_out/'meta/stats.json',aggregate_stats([v[1] for v in exports]))
        core.dump(dataset_out/'meta/source.json',{'source':str(root),'video_processing':'AV1 stream copy with keyframe preroll and exact episode offsets',
            'numeric_processing':'trim/reindex only; state/action values unchanged','image_statistics':'1Hz original-image samples downscaled320x240 over retained interval',
            'action_config_end_frame':'conservatively treated as inclusive; preserved in output','mapping_file':'../../episode_mapping.json'})
    core.dump(out/'episode_mapping.json',mappings)
    validation=validate_output(dataset_out,root,by_id,mappings);core.dump(out/'validation.json',validation)
    source_seconds=sum(ep['length'] for ep in episodes)/fps
    summary={'input':str(root),'episodes':len(reports),'frames':sum(ep['length'] for ep in episodes),'fps':fps,
        'selected_episode_ids':chosen,'workers':args.workers,'status_counts':{s:sum(r['status']==s for r in reports) for s in ['PASS','REVIEW','FAIL']},
        'exported_episodes':len(exports),'exported_frames':sum(v[0]['length'] for v in exports),'gripper_enabled':False,
        'source_modified':False,'trimmed_exported_episodes':sum(r['status']=='PASS' and r['trim_proposal']['changed'] for r in reports),
        'trim_comparison_episodes':sorted(compare_ids),'filter_thresholds_provisional':True,
        'review_policy':'REVIEW withheld for manual review; not labeled definitively corrupt','validation':validation}
    pd.DataFrame(samples).to_csv(report_out/'video_samples.csv',index=False)
    pd.DataFrame([{'episode_index':r['episode_index'],'status':r['status'],'exported_episode_index':r['exported_episode_index'],
        'reasons':';'.join(sorted(set(a['code'] for a in r['reasons']))),'trim_start':r['trim_proposal']['start_frame'],
        'trim_end_exclusive':r['trim_proposal']['end_frame_exclusive']} for r in reports]).to_csv(report_out/'episodes.csv',index=False)
    pd.DataFrame([{'episode_index':r['episode_index'],'human_decision':'','notes':''} for r in reports]).to_csv(report_out/'manual_review.csv',index=False)
    core.dump(report_out/'trim_comparisons.json',comparisons);core.dump(report_out/'summary.json',summary)
    core.write_html(report_out,reports,summary)
    page=report_out/'index.html'
    page.write_text(page.read_text().replace('仅建议，未裁剪。','PASS已按此边界导出；REVIEW/FAIL未导出。').replace('本轮不删除或修改数据。FAIL 为规则不合格候选，REVIEW 为待人工复核；','源数据未修改。PASS导出清洗副本；FAIL整条剔除；REVIEW暂不导出、待人工复核；'))
    comparison_html='<h1>首尾静止裁剪对照（均为原速）</h1><p>BEFORE包含被删静止段；AFTER仅含边界内保留帧。两段起点/时长可能不同，源帧编号用于核对。</p>'
    for c in comparisons:
        comparison_html+=f'<h2>Episode {c["source_episode_index"]} / {c["side"]} / boundary={c["boundary_frame"]}</h2>'
        for label in ('before','after'):
            v=c[label];comparison_html+=f'<h3>{label} [{v["source_start_frame"]}, {v["source_end_frame_exclusive"]})</h3><video controls preload="none" width="960" src="{v["path"]}"></video>'
    (report_out/'trim_comparisons.html').write_text('<!doctype html><meta charset="utf-8">'+comparison_html)
    report_done=time.perf_counter();total=report_done-started
    performance={'source_trajectory_seconds':source_seconds,'source_camera_video_seconds':source_seconds*3,
        'trajectory_hours_per_wall_hour':source_seconds/total,'camera_video_hours_per_wall_hour':source_seconds*3/total,
        'workers':args.workers,'sample_hz':config['sample_hz'],'timing':{'setup_calibration_seconds':setup_done-started,
        'quality_analysis_overview_seconds':analysis_done-setup_done,'evidence_export_comparison_seconds':media_done-analysis_done,
        'metadata_validation_reports_seconds':report_done-media_done,'total_seconds':total},
        'scope':'含双路抖动、基础质检、速览/证据、实际无转码切分导出、裁剪前后对照和结构/像素抽查；不含Python启动及顶层导入',
        'conditions':'共享服务器小样本；非空载独占，缓存/异常比例/切分数量影响速度'}
    core.dump(out/'performance.json',performance)
    text=f"# 实测效率\n\n10条抽样（实际{len(episodes)}条），轨迹{source_seconds:.2f}秒，耗时{total:.2f}秒。\n\n**轨迹口径{source_seconds/total:.2f}倍；三相机视频累计口径{source_seconds*3/total:.2f}倍。**\n\n"
    text+='\n'.join(f'- {k}: {v:.3f}秒' for k,v in performance['timing'].items())+'\n\n'+performance['scope']+'\n'+performance['conditions']+'\n'
    (out/'performance.md').write_text(text)
    (report_out/'performance.md').write_text(text)
    with (report_out/'index.html').open('a') as f:f.write('<p><a href="trim_comparisons.html">首尾裁剪前后对照</a> · <a href="../performance.md">新版完整流程速度</a></p>')
    print(json.dumps({'summary':summary,'performance':performance},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
