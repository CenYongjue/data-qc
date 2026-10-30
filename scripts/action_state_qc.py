#!/usr/bin/env python3
"""Absolute joint-target / measured-state QC. Report only; never edits input.

Run with the dedicated QC venv. All thresholds are provisional and overridable
by --config JSON. W=diag(1/range**2), so weighted distances are dimensionless.
Positive lag means state follows action. No circular shifts or future smoothing.
"""
import argparse
import json
import re
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
os.environ.setdefault('MPLCONFIGDIR', '/tmp/action_state_qc_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT = dict(max_delay_s=0.5, min_overlap_frames=30, min_joint_range_rad=0.05,
               min_xcorr=0.25, min_peak_margin=0.02, min_active_joints=2,
               command_deadband_rad=0.0001, min_window_s=0.025,
               min_target_norm=0.03, min_response_norm=0.002,
               reverse_cos=-0.3, positive_progress=0.01,
               min_assessable_fraction=0.2, min_assessable_windows=10,
               review_reverse_fraction=0.1, review_no_response_fraction=0.5,
               fail_reverse_fraction=0.6, enable_behavior_fail=False,
               plot_episodes=8)


def canonicalize_action(df, info, action_type='absolute_position'):
    """Validate explicit schema and map action names onto state joint order.

    Absolute positions in radians are confirmed by the user. State measurement
    provenance is *assumed*, not verified by a numeric similarity test.
    """
    if action_type != 'absolute_position':
        raise ValueError('This version accepts confirmed absolute-position actions only')
    sn = info['features']['observation.state']['names']
    an = info['features']['action']['names']
    def key(n):
        return re.sub(r'_(target_dof|target_q|q)_', '_', n)
    ak = [key(n) for n in an]
    if len(set(ak)) != len(ak):
        raise ValueError('Ambiguous action joint names')
    order = [ak.index(key(n)) for n in sn]
    a = np.stack(df['action']).astype(np.float64)[:, order]
    q = np.stack(df['observation.state']).astype(np.float64)
    t = df['timestamp'].to_numpy(dtype=float)
    if a.shape != q.shape or q.shape[1] != len(sn):
        raise ValueError('Action/state dimensions do not match metadata')
    return a, q, t, sn, [an[i] for i in order]


def compute_state_velocity(q, t):
    """Backward differences at interval midpoints; no acausal smoothing."""
    return np.diff(q, axis=0) / np.diff(t)[:, None]


def estimate_delay_xcorr(a, q, t, scale, cfg):
    """Pearson per active joint, median across joints at each nonwrapped lag.

    Differentiate BOTH target and measurement: comparing raw position targets
    with measured velocity would mix units and favor trend-related artifacts.
    A negative-lag search is diagnostic only; it is never used to align windows.
    """
    # 对目标和状态同时差分，避免把位置趋势误认为响应延迟。
    av = compute_state_velocity(a, t) / scale
    qv = compute_state_velocity(q, t) / scale
    active = (np.std(av, axis=0) > 0.02) & (np.std(qv, axis=0) > 0.02)
    dt = float(np.median(np.diff(t)))
    kmax = min(int(cfg['max_delay_s'] / dt + 1e-6), len(av)-cfg['min_overlap_frames'])
    curves = []
    for k in range(-max(kmax, 0), max(kmax, 0)+1):
        # qv[i+k] responds to av[i] for positive k; no np.roll wraparound.
        x, y = (av[:len(av)-k or None], qv[k:]) if k >= 0 else (av[-k:], qv[:len(qv)+k])
        rs = []
        if len(x) >= cfg['min_overlap_frames']:
            # Vectorized SciPy Pearson avoids one Python call per joint.
            ids = np.flatnonzero(active)
            ids = ids[(np.std(x[:, ids], axis=0) > 1e-8) &
                      (np.std(y[:, ids], axis=0) > 1e-8)]
            if len(ids):
                rs = pearsonr(x[:, ids], y[:, ids], axis=0).statistic.tolist()
        curves.append(dict(lag_frames=k, lag_s=k*dt, correlation=float(np.median(rs)) if rs else np.nan,
                           active_joints=len(rs), overlap_frames=len(x)))
    curve = pd.DataFrame(curves)
    pos = curve[(curve.lag_frames >= 0) & curve.correlation.notna()]
    if pos.empty:
        return dict(delay_s=0., reliable=False, peak_correlation=None, reason='INSUFFICIENT_EXCITATION'), curve
    best = pos.loc[pos.correlation.idxmax()]
    # Compare well-separated candidates; neighboring bins of a broad peak alone
    # are not evidence that latency is precisely identifiable.
    far = pos[np.abs(pos.lag_frames-best.lag_frames) >= 2]
    margin = float(best.correlation-far.correlation.max()) if len(far) else 0.
    neg = curve[(curve.lag_frames < 0) & curve.correlation.notna()]
    state_leading = bool(len(neg) and neg.correlation.max() > best.correlation + .05)
    reliable = bool(best.correlation >= cfg['min_xcorr'] and margin >= cfg['min_peak_margin']
                    and best.active_joints >= cfg['min_active_joints'] and not state_leading
                    and best.lag_frames < kmax)
    return dict(delay_s=float(best.lag_s), reliable=reliable,
                peak_correlation=float(best.correlation), peak_margin=margin,
                negative_peak_correlation=float(neg.correlation.max()) if len(neg) else None,
                state_leading_suspected=state_leading,
                reason='OK' if reliable else 'DELAY_UNCERTAIN'), curve


def segment_action_windows(a, t, delay, cfg):
    """Each changed published target owns [publish+lag, next_publish+lag).

    Identical targets are held until next meaningful update. A one-frame target
    is NOT given an arbitrarily long future horizon; later commands supersede it.
    Last target ends at recording end. Fractional endpoints are interpolated.
    """
    # 后一条目标会覆盖前一条；相同目标的重复发送合并为保持区间。
    starts = np.r_[0, np.flatnonzero(np.max(np.abs(np.diff(a, axis=0)), axis=1)
                                   > cfg['command_deadband_rad'])+1]
    for n, i in enumerate(starts):
        publish_end = float(t[starts[n+1]]) if n+1 < len(starts) else float(t[-1])
        lo, hi = float(t[i]+delay), min(publish_end+delay, float(t[-1]))
        yield int(i), float(t[i]), publish_end, lo, hi


def compute_direction_score(target, response):
    den = np.linalg.norm(target)*np.linalg.norm(response)
    return float(np.dot(target, response)/den) if den > 1e-12 else np.nan


def compute_progress_score(target, response):
    return float(np.dot(target, response)/max(np.dot(target, target), 1e-12))


def compute_response_magnitude(response):
    return float(np.linalg.norm(response))


def aggregate_episode_qc(ep, windows, delay, cfg, duration):
    valid = windows[windows.assessable]
    fraction = len(valid)/max(len(windows), 1)
    reverse = float((valid.label == 'REVERSE').mean()) if len(valid) else np.nan
    noresponse = float((valid.label == 'NO_RESPONSE').mean()) if len(valid) else np.nan
    reasons = []
    if len(valid) < cfg['min_assessable_windows'] or fraction < cfg['min_assessable_fraction']:
        status = 'UNASSESSABLE'; reasons.append('INSUFFICIENT_VALID_WINDOWS')
    elif not delay['reliable']:
        status = 'UNASSESSABLE'; reasons.append('DELAY_UNCERTAIN')
    else:
        status = 'PASS'
        if reverse > cfg['review_reverse_fraction']:
            status = 'REVIEW'; reasons.append('REVERSE_MOTION')
        if noresponse > cfg['review_no_response_fraction']:
            status = 'REVIEW'; reasons.append('NO_RESPONSE')
        if cfg['enable_behavior_fail'] and reverse > cfg['fail_reverse_fraction']:
            status = 'FAIL'
    return dict(episode_index=int(ep), status=status, reasons=';'.join(reasons),
                duration_s=duration, windows=len(windows), assessable_windows=len(valid),
                assessable_fraction=fraction, mean_direction_score=float(valid.direction_score.mean()),
                positive_progress_fraction=float((valid.progress_score > cfg['positive_progress']).mean()) if len(valid) else np.nan,
                reverse_fraction=reverse, no_response_fraction=noresponse, **delay)


def inspect_episode(ep, frame, info, scale, cfg):
    a, q, t, names, _ = canonicalize_action(frame, info)
    if len(t)<cfg['min_overlap_frames']+2 or not np.isfinite(np.c_[a,q,t]).all() or np.any(np.diff(t)<=0):
        raise ValueError('NONFINITE_NONMONOTONIC_OR_TOO_SHORT')
    delay, curve = estimate_delay_xcorr(a,q,t,scale,cfg)
    # Four independent kinematic groups prevent one large hand motion from
    # hiding reversed arm motion. Full 58-D scores are also retained.
    groups = {'all':np.arange(len(names))}
    for prefix in ('left_arm','left_hand','right_arm','right_hand'):
        groups[prefix] = np.array([i for i,n in enumerate(names) if n.startswith(prefix)])
    rows=[]
    def at(s):
        return np.array([np.interp(s,t,q[:,j]) for j in range(q.shape[1])])
    for i,pub,nxt,lo,hi in segment_action_windows(a,t,delay['delay_s'],cfg):
        duration=hi-lo
        reason = 'WINDOW_TOO_SHORT' if duration < cfg['min_window_s'] else ''
        q0,q1=at(min(lo,t[-1])),at(max(min(hi,t[-1]),min(lo,t[-1])))
        # 只看这个指令拥有的时间段，不要求机器人在短窗口内完全到位。
        target=(a[i]-q0)/scale; response=(q1-q0)/scale
        for group,idx in groups.items():
            e,d=target[idx],response[idx]
            en=float(np.linalg.norm(e)); mag=compute_response_magnitude(d)
            why=reason or ('AT_TARGET_OR_DEADBAND' if en<cfg['min_target_norm'] else '')
            assessable=not why
            direction=compute_direction_score(e,d)
            progress=compute_progress_score(e,d)
            if why: label='UNASSESSABLE'
            elif mag<cfg['min_response_norm']: label='NO_RESPONSE'
            elif direction<cfg['reverse_cos'] and progress<0: label='REVERSE'
            elif progress>cfg['positive_progress']: label='PARTIAL_OR_FORWARD'
            else: label='WEAK_PROGRESS'
            rows.append(dict(episode_index=int(ep),group=group,command_frame=i,publish_s=pub,
                             next_publish_s=nxt,start_s=lo,end_s=hi,duration_s=max(duration,0.),
                             assessable=assessable,reason=why,label=label,target_norm=en,
                             direction_score=direction,progress_score=progress,response_magnitude=mag))
    windows=pd.DataFrame(rows)
    # Group-specific suspicious fractions contribute to episode decisions.
    # 分组汇总，避免另一只手/手臂的正常运动掩盖局部反向。
    summary=aggregate_episode_qc(ep,windows[windows.group!='all'],delay,cfg,float(t[-1]-t[0]))
    summary['frames']=len(t)
    return summary,windows,curve,(a,q,t,names)


def plot_episode(path, summary, windows, curve, arrays):
    a,q,t,names=arrays
    # Representative most-active joint per arm/hand, not all 58 unreadable lines.
    fig,axs=plt.subplots(6,1,figsize=(14,15),constrained_layout=True)
    lag=summary['delay_s']
    for ax,prefix in zip(axs[:4],('left_arm','left_hand','right_arm','right_hand')):
        ids=[i for i,n in enumerate(names) if n.startswith(prefix)]
        j=max(ids,key=lambda j:np.ptp(q[:,j]))
        ax.plot(t,a[:,j],label='Published absolute action',lw=.8)
        ax.plot(t,q[:,j],label='Measured state (assumed)',lw=.8)
        ax.plot(t+lag,a[:,j],label='Action shifted by estimated lag',alpha=.6,lw=.8)
        bad=windows[(windows.group==prefix)&windows.label.isin(['REVERSE','NO_RESPONSE'])]
        # 用单个集合画全部异常区间，避免数千次单独绘图调用。
        if len(bad):
            from matplotlib.collections import PolyCollection
            verts = [[(r.start_s,0),(r.end_s,0),(r.end_s,1),(r.start_s,1)] for r in bad.itertuples()]
            ax.add_collection(PolyCollection(verts, facecolors='red', alpha=.12,
                                             edgecolors='none', transform=ax.get_xaxis_transform()))
        ax.set_ylabel(names[j]+' [rad]');ax.legend(loc='upper right',fontsize=7)
    axs[4].plot(curve.lag_s,curve.correlation,'o-',ms=3)
    axs[4].axvline(lag,color='red');axs[4].set(xlabel='Lag [s]; positive = state follows action',ylabel='Median Pearson r')
    w=windows[windows.group=='all']
    axs[5].plot(w.start_s,w.progress_score.clip(-2,2),label='Progress (display clipped ±2)')
    axs[5].plot(w.start_s,w.direction_score,label='Direction');axs[5].legend()
    axs[5].set(xlabel='Time [s]',ylabel='Window scores')
    fig.suptitle(f"Episode {summary['episode_index']} | {summary['status']} | lag={lag:.3f}s | reliable={summary['reliable']}")
    fig.savefig(path,dpi=110);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='datasets/action/file-011.parquet')
    p.add_argument('--info',default='datasets/action/info.json')
    p.add_argument('--output',default='lerobotv3_output/action_state_qc_pilot')
    p.add_argument('--config');args=p.parse_args()
    start=time.perf_counter();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cfg=DEFAULT.copy()
    # 配置文件可只写需要覆盖的阈值；所有生效值写入报告便于复现。
    if args.config: cfg.update(json.loads(Path(args.config).read_text()))
    info=json.loads(Path(args.info).read_text());df=pd.read_parquet(args.input)
    a,q,t,names,mapped=canonicalize_action(df,info)
    # Dataset-level robust ranges, not per-window ranges. Freeze these after
    # normal-sample calibration; tiny/static joints have a floor against noise.
    scale=np.maximum(np.percentile(q,99,axis=0)-np.percentile(q,1,axis=0),cfg['min_joint_range_rad'])
    metadata=dict(config=cfg,input=str(Path(args.input).resolve()),unit='rad',action_type='absolute_position',
                  state_semantics='ASSUMED_MEASURED_NOT_INDEPENDENTLY_VERIFIED',
                  timestamp_semantics='shared row timestamp assumed for action publish and state sample',
                  thresholds_calibrated=False,scale_source='input shard p99-p1; preliminary',
                  W_diagonal=(1/scale**2).tolist(),joint_names=names,mapped_action_names=mapped)
    (out/'config_and_assumptions.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    summaries=[];all_windows=[];curves=[]
    for ep,frame in df.groupby('episode_index',sort=True):
        try:
            s,w,c,_=inspect_episode(ep,frame,info,scale,cfg)
            all_windows.append(w);c['episode_index']=int(ep);curves.append(c)
        except ValueError as e:
            s=dict(episode_index=int(ep),status='FAIL',reasons=str(e),frames=len(frame))
        summaries.append(s)
    result=pd.DataFrame(summaries);result.to_csv(out/'episodes.csv',index=False)
    windows=pd.concat(all_windows,ignore_index=True) if all_windows else pd.DataFrame()
    windows.to_parquet(out/'windows.parquet',index=False)
    if curves:pd.concat(curves,ignore_index=True).to_csv(out/'xcorr.csv',index=False)
    # Mix suspicious and ordinary episodes for review; plots are outside core timing.
    core=time.perf_counter()-start
    plotdir=out/'plots';plotdir.mkdir(exist_ok=True)
    priority=result.status.map({'FAIL':0,'REVIEW':1,'UNASSESSABLE':2,'PASS':3})
    chosen=list(result.assign(priority=priority).sort_values(['priority','episode_index']).head(max(cfg['plot_episodes']//2,1)).episode_index)
    remaining=[e for e in result.episode_index if e not in chosen]
    if remaining:
        chosen += [remaining[i] for i in np.linspace(0,len(remaining)-1,min(len(remaining),cfg['plot_episodes']-len(chosen)),dtype=int)]
    for ep in chosen:
        try:
            s,w,c,arrays=inspect_episode(ep,df[df.episode_index==ep],info,scale,cfg)
            plot_episode(plotdir/f'episode_{ep:06d}.png',s,w,c,arrays)
        except ValueError:pass
    total=time.perf_counter()-start
    duration=float(result.duration_s.sum()) if 'duration_s' in result else 0.
    report=dict(episodes=len(result),frames=len(df),status_counts=result.status.value_counts().to_dict(),
                trajectory_duration_s=duration,core_wall_s=core,total_wall_s=total,
                core_speed_x=duration/core,total_speed_x=duration/total,plotted_episodes=[int(x) for x in chosen],
                note='Provisional thresholds; UNASSESSABLE is not a failed trajectory. Window rows include all and four disjoint groups.')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    links='\n'.join(f'<p>Episode {ep}<br><img style="max-width:100%" src="plots/episode_{ep:06d}.png"></p>' for ep in chosen)
    (out/'review.html').write_text('<meta charset="utf-8"><h1>Action/state QC pilot</h1><p>Red: candidate reverse/no-response windows. State semantics assumed; thresholds uncalibrated.</p>'+links)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
