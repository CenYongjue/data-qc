#!/usr/bin/env python3
"""从已有清洗结果挑选问题较多的轨迹，导出指标图、三相机截图和视频复查页。

复用已有MP4，不重编码整段视频；图表读取全帧数值，截图只取异常代表时刻。
WARN（例如全数据共同的零四元数）不作为主要问题排序依据。
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/data_all/cenyj/.cache/agibot_problem_views/matplotlib')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse
import html
import json
from pathlib import Path
import time
import cv2
import av
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import agibot_clean_v1 as clean
core=clean.core


def rank(report):
    reasons=[r for r in report['reasons'] if r['severity']!='WARN']
    fields=[report['metrics'].get(k,{}) for k in (core.JOINT,'actions.joint.position')]
    return (int(report['status']=='FAIL'),int(report['status']=='REVIEW'),len({r['code'] for r in reasons}),
            max((m.get('outlier_fraction',0) for m in fields),default=0),
            max((m.get('longest_seconds',0) for m in fields),default=0))


def plot_metrics(path,ep,x,report,calibration,config,fps):
    times=np.arange(ep['length'])/fps;fig,axes=plt.subplots(5,1,figsize=(13,13),sharex=True,layout='constrained')
    joint_rows=[]
    for axis,key in zip(axes[:2],[core.JOINT,'actions.joint.position']):
        limits=calibration['joint_ranges'][key]
        lo=np.asarray(limits['p01']);hi=np.asarray(limits['p99']);width=np.maximum(hi-lo,1e-6)
        values=x[key];lower=np.asarray(limits['lower']);upper=np.asarray(limits['upper'])
        counts=((values<lower)|(values>upper)).sum(axis=0)
        selected=np.argsort(-counts,kind='stable')[:3]
        for index in selected:
            label=('left' if index<7 else 'right')+f'_arm_{index%7}'
            alpha=float(limits['alpha'][index])
            axis.plot(times,(values[:,index]-lo[index])/width[index],label=f'{label}, bad frames={counts[index]}',linewidth=.9)
            joint_rows.append({'field':key,'joint':label,'outlier_frames':int(counts[index]),
                               'min_observed':float(values[:,index].min()),'max_observed':float(values[:,index].max()),
                               'lower':float(lower[index]),'upper':float(upper[index]),'alpha':alpha})
        # 当前配置alpha=0.10；范围归一化后同为[-alpha,1+alpha]。
        alpha_values=np.asarray(limits['alpha'])[selected]
        if np.allclose(alpha_values,alpha_values[0]):
            axis.axhspan(-alpha_values[0],1+alpha_values[0],color='green',alpha=.10,label='allowed statistical interval')
            axis.axhline(-alpha_values[0],color='red',linestyle='--',linewidth=.8)
            axis.axhline(1+alpha_values[0],color='red',linestyle='--',linewidth=.8)
        axis.set_title(key);axis.set_ylabel('(q - p1) / (p99 - p1)');axis.legend(loc='upper right',fontsize=8)
    scale=np.asarray(calibration['joint_ranges'][core.END]['scale'])
    jitter=core.jitter_features(x[core.END],scale,fps,config['smoothing_window_frames'])
    threshold=np.asarray(calibration['residual_thresholds'])
    valid=np.ones(len(times),bool);border=config['smoothing_window_frames']//2+2
    valid[:border]=False;valid[-border:]=False
    for i,name in enumerate(['residual','residual velocity','residual acceleration']):
        axes[2].plot(times[valid],(jitter/threshold)[valid,i],label=name,linewidth=.8)
    axes[2].axhline(1,color='red',linestyle='--');axes[2].set_ylabel('score / threshold')
    axes[2].set_title('Method B: all three scores must exceed 1 and meet duration rule');axes[2].legend(fontsize=8)
    for key in clean.MONITOR:
        filtered=clean.high_frequency_features(x[key],np.asarray(calibration['joint_ranges'][key]['scale']),fps,config)
        if filtered is None:continue
        score=np.minimum(filtered['rms']/np.asarray(calibration['filter_rms_thresholds'][key]),
                         filtered['ratio']/config['filter_energy_ratio']).max(axis=1)
        score[~filtered['valid']]=np.nan
        axes[3].plot(times,score,label=key.replace('observation.states.','state.').replace('actions.','action.'),linewidth=.8)
    axes[3].axhline(1,color='red',linestyle='--');axes[3].set_ylabel('combined gate / threshold')
    axes[3].set_title(f'Method A: RMS and energy ratio gate; score > 1 must persist >= {config["filter_run_seconds"]} s');axes[3].legend(fontsize=8)
    scales={k:np.asarray(calibration['joint_ranges'][k]['scale']) for k in [core.JOINT,core.END,core.GRIP]}
    trim,speed=core.propose_trim(x,ep,scales,config,fps)
    axes[4].plot(times,speed,linewidth=.8,label='max normalized speed')
    axes[4].axhline(config['static_normalized_speed'],color='red',linestyle='--',label='activity threshold')
    if trim['start_frame']>0:axes[4].axvspan(0,trim['start_frame']/fps,color='orange',alpha=.25,label='proposed head trim')
    if trim['end_frame_exclusive']<ep['length']:axes[4].axvspan(trim['end_frame_exclusive']/fps,ep['length']/fps,color='orange',alpha=.25,label='proposed tail trim')
    axes[4].set_title('Static-end trim proposal (applied only to PASS exports)');axes[4].legend(fontsize=8)
    axes[4].set_xlabel('source episode time (seconds)');axes[4].set_ylabel('range / second')
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle(f'Episode {ep["episode_index"]:03d} | {report["status"]} | statistical candidates, not physical limits')
    fig.savefig(path,dpi=110);plt.close(fig)
    return joint_rows


def main():
    started=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--count',type=int,default=6)
    args=p.parse_args();root=args.result.resolve();report_root=root/'report';out=report_root/'problem_views'
    if out.exists():raise FileExistsError(out)
    out.mkdir();cv2.setNumThreads(1)
    summary=json.loads((report_root/'summary.json').read_text());source=Path(summary['input'])
    info=json.loads((source/'meta/info.json').read_text());config=json.loads((report_root/'config.json').read_text())
    calibration=json.loads((report_root/'calibration.json').read_text())
    reports=[json.loads(f.read_text()) for f in (report_root/'episodes').glob('*.json')]
    candidates=sorted([r for r in reports if any(a['severity']!='WARN' for a in r['reasons'])],key=rank,reverse=True)
    selected=candidates[:args.count]
    episodes={e['episode_index']:e for e in pq.read_table(source/'meta/episodes').to_pylist()}
    table=pq.read_table(source/'data');numeric={k:core.numpy_column(table,k,f['shape']) for k,f in info['features'].items() if f['dtype']!='video'}
    reader=core.Reader();cards=[];rankings=[]
    try:
        for report in selected:
            i=report['episode_index'];ep=episodes[i];a,b=ep['dataset_from_index'],ep['dataset_to_index']
            x={k:v[a:b] for k,v in numeric.items()};plot=f'episode_{i:06d}_metrics.png'
            details=plot_metrics(out/plot,ep,x,report,calibration,config,info['fps'])
            reasons=[reason for reason in report['reasons'] if reason['severity']!='WARN']
            representative=max(reasons,key=lambda reason:max([v.get('duration_seconds',0) for v in reason.get('intervals',[])],default=0))
            local=representative['at_seconds'];images=[]
            for camera in core.CAMERAS:
                try:images.append(reader.frame(core.video_path(source,info,ep,camera),ep[f'videos/{camera}/from_timestamp']+local,.5/info['fps'])[0])
                except (ValueError,OSError,av.error.FFmpegError):images.append(None)
            snapshot=f'episode_{i:06d}_snapshot.jpg'
            cv2.imwrite(str(out/snapshot),core.montage(images,i,local,'REPRESENTATIVE FLAG TIME / NO OBJECT DETECTOR'))
            reasons_html=''.join(f'<li>{html.escape(v["severity"]+" / "+v["label"]+": "+v["detail"])}</li>' for v in reasons)
            metrics=''.join(f'<tr><td>{html.escape(v["field"])}</td><td>{v["joint"]}</td><td>{v["outlier_frames"]}</td><td>{v["min_observed"]:.5g}–{v["max_observed"]:.5g}</td><td>{v["lower"]:.5g}–{v["upper"]:.5g}</td></tr>' for v in details)
            clips=''.join(f'<p>{html.escape(v["reason"])} / {v["start_seconds"]:.2f}–{v["end_seconds"]:.2f}s，原速</p><video controls preload="none" src="../{v["path"]}"></video>' for v in report['evidence_videos'])
            cards.append(f'<section id="episode-{i}"><h2>Episode {i} / {report["status"]}</h2><ul>{reasons_html}</ul>'
                f'<img src="{snapshot}"><h3>全帧数值与判定阈值</h3><img src="{plot}">'
                f'<table><tr><th>字段</th><th>关节</th><th>越界帧数</th><th>观测范围</th><th>统计阈值</th></tr>{metrics}</table>'
                f'<h3>原速问题片段</h3>{clips}<h3>三路速览（加速播放）</h3><video controls preload="none" src="../{report["overview_video"]}"></video>'
                f'<p><a href="../episodes/episode_{i:06d}.json">完整指标和区间</a></p></section>')
            ranking={'episode_index':i,'status':report['status'],'unique_nonwarning_issue_types':len({v['code'] for v in reasons}),
                     'reason_codes':sorted({v['code'] for v in reasons}),'max_joint_outlier_fraction':max(report['metrics'][k]['outlier_fraction'] for k in [core.JOINT,'actions.joint.position']),
                     'snapshot_seconds':local,'joint_details':details}
            rankings.append(ranking)
    finally:reader.close()
    result={'selected':rankings,'sorting':'FAIL优先，其次REVIEW、非WARN问题种类、关节越界比例、最长连续越界；不按重复WARN排序',
            'common_warning':'invalid_orientation零范数机器人姿态是全数据共同警告，不作为异常样本排序依据。',
            'conditions':'统计异常仅为候选，不能仅凭图像判断机械限位；A/B图分开显示，未超阈值不伪造为抖动。',
            'view_export_seconds':time.perf_counter()-started}
    core.dump(out/'selection.json',result)
    heading='<h1>AgiBot重点问题样本复查</h1><p>优先选择FAIL/REVIEW，再按问题种类与关节越界比例排序。共同的零范数四元数WARN不参与排序。统计越界不是物理越界，需人工确认。</p>'
    heading+='<p>图中同时展示关节统计阈值、原残差法B、高频滤波法A和静止裁剪建议，避免将未触发的指标误认为已检出问题。</p>'
    heading+='<p><a href="../index.html">全部轨迹报告</a> · <a href="../trim_comparisons.html">裁剪前后对照</a> · <a href="selection.json">选择依据和原始指标</a></p>'
    heading+='<p>样本导航：'+ ' · '.join(f'<a href="#episode-{r["episode_index"]}">{r["episode_index"]}</a>' for r in selected)+'</p>'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AgiBot问题样本</title>'
        '<style>body{max-width:1200px;margin:24px auto;font:16px sans-serif;background:#f5f6f8}section{padding:20px;margin:24px 0;background:white}img,video{max-width:100%;width:100%}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:5px}</style>'+heading+''.join(cards))
    with (report_root/'index.html').open('a') as f:f.write('<p><a href="problem_views/index.html">重点问题样本：指标图、截图及视频</a></p>')
    print(json.dumps({'selected_episode_ids':[r['episode_index'] for r in rankings],
                      'view_export_seconds':result['view_export_seconds'],'page':str(out/'index.html')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
