#!/usr/bin/env python3
"""视频质检v2：先文件/时间轴，再画面，最后双GPU人手检测；支持全量与断点续跑。

详细中文说明见README_video_quality_pipeline_v2.md。默认不修改/删除/裁剪源数据。
所有阈值为可调初始值，候选不是已证实损坏。完整解码不等于完整画面语义判断。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('YOLO_CONFIG_DIR',os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'yolo'))
os.environ.setdefault('MPLCONFIGDIR',os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache')), 'robot-data-qc', 'matplotlib'))
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from collections import Counter
import csv
import hashlib
import html
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sqlite3
import time
import traceback
import av
import cv2
import numpy as np
from egodex_yolo_pipeline import gpu_loop
import egodex_video_qc as legacy
from video_qc_v2.worker import process_record,init_cpu,LABELS,reason


def dump(path,value):
    """先写临时文件再原子替换：运行时读取progress.json不会拿到半截JSON。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def discover(root,manifest=None):
    if manifest:
        # 多路视频必须显式分组，不能把不同回合误当成相机。共同钟由数据提供方声明。
        data=json.loads(manifest.read_text());records=data['records'] if isinstance(data,dict) else data
        for r in records:
            for camera in r['cameras'].values():camera['path']=str((manifest.parent/camera['path']).resolve())
            if r.get('hdf5'):r['hdf5']=str((manifest.parent/r['hdf5']).resolve())
        return records
    if next(root.rglob('meta/info.json'),None) is not None:
        from video_qc_v2.lerobot import discover_lerobot
        return discover_lerobot(root)
    groups={}
    for path in root.rglob('*.mp4'):
        relative=path.relative_to(root);task=path.parent.name;identifier='__'.join(relative.with_suffix('').parts)
        groups.setdefault(task,[]).append({'id':identifier,'task':task,'kind':'human','hdf5':str(path.with_suffix('.hdf5')),
            'cameras':{'ego':{'path':str(path),'start_seconds':0.,'timing_mode':'cfr'}},'shared_clock':False})
    # 轮转任务目录，初期就能覆盖不同任务，且不用一次把4.6万个future放进内存。
    for rows in groups.values():rows.sort(key=lambda r:int(Path(r['cameras']['ego']['path']).stem))
    records=[]
    for index in range(max(map(len,groups.values()),default=0)):
        for task in sorted(groups):
            if index<len(groups[task]):records.append(groups[task][index])
    return records


def fingerprint(record):
    files=[v['path'] for v in record['cameras'].values()]+([record['hdf5']] if record.get('hdf5') else [])+record.get('metadata_paths',[])
    parts=[]
    for name in files:
        p=Path(name)
        try:s=p.stat();parts.append([name,s.st_size,s.st_mtime_ns])
        except FileNotFoundError:parts.append([name,None,None])
    return hashlib.sha256(json.dumps([record,parts],sort_keys=True).encode()).hexdigest()


def export_actual_issue(record,report,out,config):
    """只输出实际REVIEW/FAIL的候选原速片段；原源文件从不修改。

    同时支持多路：每个有问题的相机各有证据，跨相机错误给每路一个代表片段。
    若源视频根本打不开，保留错误原因而不是让媒体导出异常杀掉全量任务。
    """
    views={}
    for key,camera in report['cameras'].items():
        reasons=[r for r in report['reasons'] if r['severity'] in ['FAIL','REVIEW'] and r['camera'] in (None,key)]
        if not reasons or not camera['samples']:continue
        source=record['cameras'][key]
        # 旧导出器读取整段文件的本地时间；将episode局部秒数恢复到原文件PTS坐标。
        offset=source.get('start_seconds',0.)
        samples=[{**s,'seconds':s['seconds']+offset,'hand_count':s['hand_count'] if s['hand_count'] is not None else 0} for s in camera['samples']]
        temp={'id':Path(record['id']).name+'__'+key,'video':source['path'],'fps':camera['fps'],'start_seconds':offset,
              'duration_seconds':camera['duration_seconds']+offset,'samples':samples,
              'reasons':[{**r,'at_seconds':r['at_seconds']+offset} for r in reasons]}
        # clip_id和路径由扫描/manifest产生，不允许外部ID写出输出目录。
        if any(part in temp['id'] for part in ['..','/','\\']):raise ValueError('Unsafe record ID')
        target=out/'review'/record['task'];target.mkdir(parents=True,exist_ok=True)
        adapted={**config,'problem_overviews':True,'evidence_clip_seconds':config['evidence_clip_seconds'],
                 'max_evidence_clips_per_episode':config['max_evidence_clips_per_episode']}
        original_annotate=legacy.annotate;original_movie=legacy.Movie
        checked=any(s['hand_count'] is not None for s in camera['samples'])
        def annotate(image,sample,t,caption):
            canvas=original_annotate(image,sample,t,caption)
            canvas[566:]=0
            text=f'nearest 4Hz sample={sample["seconds"]:.2f}s, hands={sample["hand_count"]}' if checked and sample else 'Hand check SKIPPED; no absence conclusion'
            cv2.putText(canvas,text,(10,582),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,220,255),1)
            return canvas
        def movie(path,fps):return original_movie(path,8 if path.parent.name=='overviews' else fps)
        legacy.annotate=annotate;legacy.Movie=movie
        try:
            views[key]=legacy.export_review((temp,str(target),adapted))
            views[key]['path_prefix']=str(target.relative_to(out))
            views[key]['note']='框为最近4Hz采样；未执行手部时框数0不代表无手'
        except Exception as exc:views[key]={'render_error':f'{type(exc).__name__}: {exc}'}
        finally:legacy.annotate=original_annotate;legacy.Movie=original_movie
    return views


def job_wrapper(job):
    record,config,work,out,finger=job
    before=time.perf_counter()
    try:
        report=process_record((record,config,work))
    except Exception as exc:
        # 算子内部故障不能伪装成源文件损坏；报告待查并继续批处理。
        report={'id':record['id'],'task':record['task'],'kind':record.get('kind','human'),'reasons':[],
                'cameras':{},'status':'REVIEW','source_modified':False,'stages':{'hands':'SKIPPED_OPERATOR_ERROR'}}
        reason(report,'operator_error','REVIEW',f'{type(exc).__name__}: {exc}');report['traceback']=traceback.format_exc()
    report['source_fingerprint']=finger
    report['task_description']=record.get('task_description',[])
    report['source_segments']={k:{name:value for name,value in camera.items() if name!='_predecoded'} for k,camera in record['cameras'].items()}
    report['review_media']=export_actual_issue(record,report,Path(out),config) if report['status']!='PASS' else {}
    duration=max((c.get('duration_seconds',0) for c in report['cameras'].values()),default=0.)
    report['source_seconds']=duration;report['complete_worker_seconds']=time.perf_counter()-before
    # 全帧运动参数仅对真正抖动/中央集中候选保存为压缩NPZ，普通低纹理不存海量轨迹。
    for key,camera in report['cameras'].items():
        motion_rows=camera.pop('_motion_rows',None)
        if motion_rows and any(v['code'] in ('shake','motion_uncertain') and v['camera']==key for v in report['reasons']):
            path=Path(out)/'traces'/record['task']/f'{record["id"]}__{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,seconds=[v['seconds'] for v in motion_rows],reliable=[v['reliable'] for v in motion_rows],
                x=[v.get('x',np.nan) for v in motion_rows],y=[v.get('y',np.nan) for v in motion_rows],angle=[v.get('angle',np.nan) for v in motion_rows],
                coverage=[v.get('coverage',0) for v in motion_rows],central_fraction=[v.get('central_fraction',0) for v in motion_rows])
            camera['motion_trace']=str(path.relative_to(out))
    path=Path(out)/'episodes'/record['task']/f'{record["id"]}.json';dump(path,report)
    return {'id':record['id'],'task':record['task'],'fingerprint':finger,'status':report['status'],
            'seconds':duration,'camera_seconds':sum(c.get('duration_seconds',0) for c in report['cameras'].values()),'report':str(path.relative_to(out)),'reasons':report['reasons'],
            'media':report['review_media'],'worker_seconds':report['complete_worker_seconds'],
            'hands_samples':sum(sum(s['hand_count'] is not None for s in c['samples']) for c in report['cameras'].values()),
            'timing':{k:sum(c['timing_seconds'].get(k,0) for c in report['cameras'].values()) for k in ['decoder_setup','decode','gray','temporal','lk','signalstats','blockdetect','cache','hands_submit_wait']}}


def dashboard(out,db,total,started,session_completed,gpu_loading,final=False,budget_exhausted=False,cpu_workers=4):
    now=time.time();rows=db.execute('SELECT payload FROM records').fetchall();items=[json.loads(v[0]) for v in rows]
    status=Counter(v['status'] for v in items);by_task={}
    source=sum(v['seconds'] for v in items);elapsed=time.perf_counter()-started
    current=[v for v in items if v.get('session')==session_completed]
    current_seconds=sum(v['seconds'] for v in current)
    current_camera_seconds=sum(v.get('camera_seconds',v['seconds']) for v in current)
    progress={'state':'budget_complete' if budget_exhausted else 'complete' if final and len(items)==total else 'running','total_records':total,'completed_records':len(items),
              'status_counts':dict(status),'remaining_records':total-len(items),'last_updated_epoch':now,
              'session_wall_seconds':elapsed,'session_completed':len(current),'session_source_seconds':current_seconds,
              'session_realtime_factor_including_startup':current_seconds/elapsed if elapsed else None,
              'session_camera_seconds':current_camera_seconds,'session_camera_realtime_factor':current_camera_seconds/elapsed if elapsed else None,
              'processed_source_hours_all_sessions':source/3600,'cpu_workers':cpu_workers,'sample_hz':4,'gpu_loading':gpu_loading,
              'source_modified':False,'timing_note':'含文件检查、完整解码、画面算子、手部推理、告警片段与报告；不是GPU纯推理速度。'}
    dump(out/'progress.json',progress)
    template='<!doctype html><meta charset="utf-8"><style>body{max-width:1150px;margin:25px auto;font:16px sans-serif}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}video,img{max-width:100%}pre{white-space:pre-wrap}</style>'
    links=[]
    for item in items:by_task.setdefault(item['task'],[]).append(item)
    for task,values in sorted(by_task.items()):
        counts=Counter(v['status'] for v in values);cards=[]
        for item in sorted(values,key=lambda v:(v['status']=='PASS',v['id'])):
            reasons=''.join(f'<li>{html.escape(r["severity"]+" / "+r["label"]+": "+r["detail"])}</li>' for r in item['reasons'] if r['severity']!='WARN')
            media=''
            for camera,view in item['media'].items():
                prefix=view.get('path_prefix','')
                for clip in view.get('videos',[]):
                    path=f'../{prefix}/{clip["path"]}'
                    media+=f'<p>{html.escape(camera)} / {html.escape(clip["reason"])} / 原速</p><video controls preload="none" src="{path}"></video>'
                if view.get('overview'):
                    media+=f'<p>4Hz速览，2倍速</p><video controls preload="none" src="../{prefix}/{view["overview"]}"></video>'
                if view.get('render_error'):media+=f'<p>源视频无法生成片段：{html.escape(view["render_error"])}</p>'
            if item['status']!='PASS':cards.append(f'<section><h2>{html.escape(item["id"])} / {item["status"]}</h2><ul>{reasons}</ul>{media}<p><a href="../{item["report"]}">指标及区间JSON</a></p></section>')
        path=out/'tasks'/f'{task}.html';path.parent.mkdir(exist_ok=True)
        path.write_text(template+f'<h1>{html.escape(task)}</h1><p>{html.escape(str(dict(counts)))}</p><p><a href="../index.html">返回进度</a></p>'+''.join(cards)+
            '<h2>全部回合</h2>'+''.join(f'<p><a href="../{v["report"]}">{html.escape(v["id"])}</a> {v["status"]}</p>' for v in values))
        links.append(f'<tr><td><a href="tasks/{task}.html">{html.escape(task)}</a></td><td>{len(values)}</td><td>{counts["PASS"]}</td><td>{counts["REVIEW"]}</td><td>{counts["FAIL"]}</td></tr>')
    (out/'index.html').write_text(template+('<meta http-equiv="refresh" content="30">' if not final else '')+
        '<h1>视频质检v2：全量进度与问题片段</h1><p><a href="review.html">按检出类别筛选问题视频</a></p><p>4Hz画质/人手采样；完整性、冻结、闪烁和镜头运动使用完整时间轴。人手检测仅用于人类数据，机器人/UMI跳过；不改源数据。多相机所有相机均合格才通过。</p>'+
        '<p>REVIEW是候选，不等于确认损坏。WARN包含抖动不可可靠评分，不能将这些算子视为通过。此页每30秒刷新，按任务打开原速问题视频。</p><pre>'+html.escape(json.dumps(progress,ensure_ascii=False,indent=2))+'</pre>'+
        '<table><tr><th>任务</th><th>完成</th><th>PASS</th><th>REVIEW</th><th>FAIL</th></tr>'+''.join(links)+'</table><p><a href="progress.json">机器可读进度</a> · <a href="config.json">全部阈值</a></p>')
    with (out/'episodes.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['id','task','status','seconds','reason_codes','report'])
        writer.writerows([v['id'],v['task'],v['status'],v['seconds'],';'.join(sorted({r['code'] for r in v['reasons']})),v['report']] for v in items)
    return progress


def main():
    started=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('datasets/egoDex'))
    p.add_argument('--manifest',type=Path,help='可选多相机JSON；否则递归扫描EgoDex MP4')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('video_qc_v2_config.json'))
    p.add_argument('--weights',type=Path,default=Path('models/hand_presence/hand_yolov8n.pt'))
    p.add_argument('--gpus',default='0,1');p.add_argument('--limit',type=int,help='仅用于必要的小样本验证，不传则全量')
    p.add_argument('--time-budget-seconds',type=float,help='实际运行时间预算；到时不再派发新回合，完成在途回合后退出，保留断点')
    p.add_argument('--cpu-workers',type=int,default=4,help='CPU回合工作进程数；每个解码器仍为2线程')
    args=p.parse_args();root=args.input.resolve();out=args.output.resolve();c=json.loads(args.config.read_text())
    if args.time_budget_seconds is not None and args.time_budget_seconds<=0:raise ValueError('时间预算必须大于零')
    if args.cpu_workers<=0:raise ValueError('CPU工作进程数必须大于零')
    if out.exists() and not args.resume:raise FileExistsError('Output exists; use --resume')
    out.mkdir(parents=True,exist_ok=True);work=out/'.cache';work.mkdir(exist_ok=True)
    config_hash=hashlib.sha256(json.dumps(c,sort_keys=True).encode()+args.weights.read_bytes()).hexdigest()
    old=out/'run.json'
    if old.exists() and json.loads(old.read_text())['config_hash']!=config_hash:raise ValueError('配置/模型变化，不能复用旧结果，请使用新目录')
    records=discover(root,args.manifest.resolve() if args.manifest else None)
    if args.limit:records=records[:args.limit]
    if not records:raise ValueError('No records')
    for record in records:
        for name in [record['id'],record['task'],*record['cameras']]:
            if any(v in name for v in ['..','/','\\']):raise ValueError('Unsafe ID/task/camera')
    db=sqlite3.connect(out/'results.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY,fingerprint TEXT,payload TEXT)');db.commit()
    done={v[0]:v[1] for v in db.execute('SELECT id,fingerprint FROM records')}
    jobs=[]
    for record in records:
        finger=fingerprint(record)
        if done.get(record['id'])!=finger:jobs.append((record,c,str(work),str(out),finger))
    session=f'{os.getpid()}-{time.time_ns()}'
    dump(out/'run.json',{'input':str(root),'records':len(records),'config_hash':config_hash,'session':session,'cpu_workers':args.cpu_workers,'source_modified':False})
    dump(out/'config.json',c)
    from video_qc_v2.review_page import write_review_page
    write_review_page(out,LABELS)
    (out/'RUNNING.pid').write_text(str(os.getpid()))
    # 每个CPU进程独享回复队列，GPU仍为共享输入队列；避免扩并行后回复送错进程。
    ctx=mp.get_context('spawn');requests=ctx.Queue(maxsize=64);replies=[ctx.Queue(maxsize=32) for _ in range(args.cpu_workers)];slots=ctx.Queue();status=ctx.Queue()
    for slot in range(args.cpu_workers):slots.put(slot)
    # 全机器人任务不加载GPU模型，释放GPU给服务器其他任务；混合任务才启动人手检测器。
    devices=[int(v) for v in args.gpus.split(',')] if any(r.get('kind')!='robot' for r in records) else []
    if len(devices)!=len(set(devices)):raise ValueError('Duplicate GPUs')
    processes=[ctx.Process(target=gpu_loop,args=(gpu,str(args.weights.resolve()),requests,replies,status,c['gpu_batch_size'],c['gpu_flush_ms']/1000,c['hand_imgsz'],c['hand_confidence'])) for gpu in devices]
    loading=[];gpu_stats=[];last_page=0.;budget_exhausted=False
    print(f'Discovered {len(records)} records; pending {len(jobs)}; CPU={args.cpu_workers}; GPU={devices}; sample_hz=4',flush=True)
    try:
        for process in processes:process.start()
        deadline=time.perf_counter()+120
        while len(loading)<len(processes):
            try:message=status.get(timeout=1)
            except Empty:
                if any(v.exitcode not in (None,0) for v in processes) or time.perf_counter()>deadline:raise RuntimeError('GPU启动失败/超时')
                continue
            if message['type']=='error':raise RuntimeError(message['traceback'])
            loading.append(message)
        dump(out/'gpu_startup.json',loading)
        dashboard(out,db,len(records),started,session,loading,cpu_workers=args.cpu_workers)
        with ProcessPoolExecutor(max_workers=args.cpu_workers,mp_context=ctx,initializer=init_cpu,initargs=(slots,requests,replies,c['hand_imgsz'])) as pool:
            iterator=iter(jobs);pending={}
            def submit():
                # 定时测速每个进程仅保留一条在途，截止后自然收尾，不杀掉正在写结果的进程。
                nonlocal budget_exhausted
                if args.time_budget_seconds is not None and time.perf_counter()-started>=args.time_budget_seconds:
                    budget_exhausted=True;return False
                try:job=next(iterator)
                except StopIteration:return False
                future=pool.submit(job_wrapper,job);pending[future]=job[0]['id'];return True
            for _ in range(args.cpu_workers if args.time_budget_seconds is not None else args.cpu_workers*2):
                if not submit():break
            while pending:
                finished,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in finished:
                    identifier=pending.pop(future);item=future.result();item['session']=session
                    db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)',(item['id'],item['fingerprint'],json.dumps(item,ensure_ascii=False)));db.commit()
                    print(f'{item["id"]} {item["status"]} {item["seconds"]:.2f}s / worker {item["worker_seconds"]:.2f}s',flush=True)
                    submit()
                # 不逐回合重写整站，默认30秒一次；进度与已完成JSON始终可查看。
                if time.perf_counter()-last_page>30:
                    progress=dashboard(out,db,len(records),started,session,loading,cpu_workers=args.cpu_workers);last_page=time.perf_counter()
                    print('PROGRESS '+json.dumps(progress,ensure_ascii=False),flush=True)
                for process in processes:
                    if process.exitcode not in (None,0):raise RuntimeError('GPU进程异常退出；已完成结果可resume')
        for _ in processes:requests.put(None)
        while len(gpu_stats)<len(processes):
            message=status.get(timeout=60)
            if message['type']=='error':raise RuntimeError(message['traceback'])
            if message['type']=='done':gpu_stats.append(message)
        for process in processes:process.join(timeout=30)
        progress=dashboard(out,db,len(records),started,session,loading,final=True,budget_exhausted=budget_exhausted,cpu_workers=args.cpu_workers)
        dump(out/'performance.json',{'progress':progress,'gpu_batches':gpu_stats,'requested_time_budget_seconds':args.time_budget_seconds,'timing_scope':'包含新4Hz检测、前置全帧检查和真实告警片段；主进程顶层导入不计入；预算到时完成在途回合，因此可略超时'})
        print('DONE '+json.dumps(progress,ensure_ascii=False),flush=True)
    except BaseException:
        dump(out/'last_error.json',{'traceback':traceback.format_exc(),'completed_records':db.execute('SELECT count(*) FROM records').fetchone()[0]})
        raise
    finally:
        for process in processes:
            if process.is_alive():process.terminate();process.join(timeout=5)
        db.close()
        (out/'RUNNING.pid').unlink(missing_ok=True)


if __name__=='__main__':main()
