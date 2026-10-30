#!/usr/bin/env python3
"""四卡持久NVDEC生产者 + 24个CPU质检进程，默认五分钟自动测试。

--detach创建独立会话、日志和PID；退出SSH或Codex后，已启动任务仍可执行。
完成后自动生成conclusion.md、benchmark_summary.json和分类复核网页。
四个GPU进程各取连续四分之一回合，最多32条在途，CPU按既有规则独立检查每条轨迹。
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sqlite3
import subprocess
import sys
import time
import traceback
from video_quality_pipeline_v2 import discover,fingerprint,dump,dashboard
from video_qc_v2.worker import LABELS
from video_qc_v2.review_page import write_review_page
from video_qc_v2.persistent_decode import producer,cpu_init,cpu_job,cleanup
from summarize_video_qc_run import summarize


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.detach:
        argv=[v for v in sys.argv[1:] if v!='--detach']
        with (out/'background.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),*argv],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        (out/'background.pid').write_text(str(child.pid))
        print(json.dumps({'background_pid':child.pid,'log':str(out/'background.log'),'conclusion':str(out/'conclusion.md')},ensure_ascii=False));return
    if (out/'results.sqlite').exists():raise FileExistsError('本实验请使用新的输出目录，避免混合不同测试结果')
    started=time.perf_counter();deadline=started+args.seconds
    config=json.loads(Path(__file__).with_name('video_qc_v2_config.json').read_text())
    devices=[int(v) for v in args.gpus.split(',')]
    if not devices or len(set(devices))!=len(devices):raise ValueError('GPU列表必须非空且不重复')
    gpu_pixels=getattr(args,'gpu_pixels',False)
    config.update(block_detection=False,decode_backend='persistent_cuda_gpu_pixels' if gpu_pixels else 'persistent_cuda',decode_gpus=devices)
    records=discover(args.input.resolve())
    if any(r.get('kind')!='robot' for r in records):raise ValueError('本实验仅适用于当前无块检测、无人手模型的机器人数据')
    hashes={r['id']:fingerprint(r) for r in records};session=f'{os.getpid()}-{time.time_ns()}'
    dump(out/'config.json',config);dump(out/'run.json',{'input':str(args.input.resolve()),'records':len(records),'session':session,
         'config_hash':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),'cpu_workers':args.cpu_workers,
         'gpu_processes':len(devices),'source_modified':False,'partition':'每张GPU一段连续回合；回合分布不同于原顺序CPU测试'})
    (out/'RUNNING.pid').write_text(str(os.getpid()));work=out/'.cache';work.mkdir(exist_ok=True);write_review_page(out,LABELS)
    db=sqlite3.connect(out/'results.sqlite');db.execute('PRAGMA journal_mode=WAL');db.execute('CREATE TABLE records (id TEXT PRIMARY KEY,fingerprint TEXT,payload TEXT)');db.commit()
    ctx=mp.get_context('spawn');ready=ctx.Queue(maxsize=8);status=ctx.Queue();credits=ctx.BoundedSemaphore(args.cpu_workers+8);stop=ctx.Event()
    producers=[];step=(len(records)+len(devices)-1)//len(devices)
    decoder_class=None
    if gpu_pixels:
        from video_qc_v2.gpu_quality import GPUDecoder
        decoder_class=GPUDecoder
    for i,gpu in enumerate(devices):
        part=records[i*step:(i+1)*step]
        producers.append(ctx.Process(target=producer,args=(gpu,part,config,ready,status,credits,stop,deadline,decoder_class)))
    pending={};done=[];received=0;completed=0;last_page=0.;loading=[{'device':gpu,'role':'persistent_NVDEC_producer'} for gpu in devices]
    pool=ProcessPoolExecutor(max_workers=args.cpu_workers,mp_context=ctx,initializer=cpu_init)
    print(f'START persistent GPU={devices}, CPU={args.cpu_workers}, budget={args.seconds}s',flush=True)
    try:
        for process in producers:process.start()
        dashboard(out,db,len(records),started,session,loading,cpu_workers=args.cpu_workers)
        while True:
            if time.perf_counter()>=deadline:stop.set()
            while True:
                try:message=status.get_nowait()
                except Empty:break
                if message['type']=='error':raise RuntimeError(message['traceback'])
                done.append(message['stats'])
            while len(pending)<args.cpu_workers:
                try:r=ready.get(timeout=.1)
                except Empty:break
                future=pool.submit(cpu_job,(r,config,str(work),str(out),hashes[r['id']]))
                pending[future]=r;received+=1
            finished,_=wait(pending,timeout=.1,return_when=FIRST_COMPLETED) if pending else ([],[])
            for future in finished:
                record=pending.pop(future)
                try:
                    item=future.result()
                    if any(r['code']=='operator_error' for r in item['reasons']):raise RuntimeError('质检算子异常，不能作为有效测速: '+json.dumps(item['reasons'],ensure_ascii=False))
                    item['session']=session
                    db.execute('INSERT INTO records VALUES (?,?,?)',(item['id'],item['fingerprint'],json.dumps(item,ensure_ascii=False)));db.commit();completed+=1
                    print(f'RESULT {item["id"]} {item["status"]} source={item["seconds"]:.2f}s',flush=True)
                finally:cleanup(record);credits.release()
            if time.perf_counter()-last_page>30:
                p=dashboard(out,db,len(records),started,session,loading,cpu_workers=args.cpu_workers);last_page=time.perf_counter()
                print('PROGRESS '+json.dumps(p),flush=True)
            for process in producers:
                if process.exitcode not in (None,0):raise RuntimeError(f'GPU生产者退出: {process.exitcode}')
            # 两个队列独立传输，done可能先于最后一个描述符到达；必须按总发出数对账。
            if len(done)==len(producers) and received==sum(s['episodes'] for s in done) and not pending:break
        pool.shutdown(wait=True)
        for process in producers:process.join(timeout=10)
        p=dashboard(out,db,len(records),started,session,loading,final=True,budget_exhausted=completed<len(records),cpu_workers=args.cpu_workers)
        dump(out/'performance.json',{'progress':p,'gpu_batches':[],'persistent_decoders':done,
             'requested_time_budget_seconds':args.seconds,'timing_scope':'包含GPU初始化、预处理、共享内存、CPU质检、复核导出和收尾；CPU阶段耗时不含GPU独立进程工作时间'})
        result=summarize(out)
        text=f'''# 四卡持久GPU解码{'与像素算子' if gpu_pixels else ''}测试结论

状态：{p['state']}，任务已经退出。

- 实际总耗时：{result['wall_minutes']:.2f}分钟（包括启动和在途回合收尾）。
- 已处理：{completed}条回合，轨迹时长{result['source_minutes']:.2f}分钟。
- 轨迹时长口径：{result['measured_realtime_factor_including_export']:.2f}倍实时。
- 六路视频合计口径：{result['measured_camera_realtime_factor_including_export']:.2f}倍实时。
- CPU无块检测历史测试：15.21倍；逐回合创建GPU解码器历史测试：5.17倍。
- 持久解码器共建立{sum(s['decoder_opens'] for s in done)}个，seek共{sum(s['seeks'] for s in done)}次。
- GPU像素算子：{'启用；CPU只接收头部灰度图和各路指标' if gpu_pixels else '未启用'}。
- CPU累计阶段占比不包含GPU进程的独立工作时间，不能直接当成整个系统的耗时占比。
- 本轮四张GPU读取四段连续回合，样本分布和机器负载可能与历史测试不同，不是严格控制条件的扩展比。

本测试关闭块检测及依赖块分数的色度花屏分支，只在头部运行LK，保留问题片段与速览。
结果与各GPU工作时间见benchmark_summary.json、performance.json；分类复核见review.html。
'''
        (out/'conclusion.md').write_text(text);print('DONE '+json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException:
        dump(out/'FAILED.json',{'traceback':traceback.format_exc(),'completed':completed})
        try:
            p=json.loads((out/'progress.json').read_text());p['state']='error';dump(out/'progress.json',p)
        except Exception:pass
        raise
    finally:
        stop.set();pool.shutdown(wait=True,cancel_futures=True)
        for record in pending.values():cleanup(record)
        for process in producers:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join(timeout=5)
        # 生产者若在Queue.put中崩溃，队列可能留下不完整消息；get_nowait仍可能阻塞。
        # 异常退出时交由resource_tracker回收，不再读取已损坏的队列。
        if all(process.exitcode==0 for process in producers):
            while True:
                try:cleanup(ready.get_nowait())
                except Empty:break
        db.close();(out/'RUNNING.pid').unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('datasets/hifi_umi'))
    p.add_argument('--output',type=Path,default=Path('lerobotv3_output/hifi_umi_persistent_nvdec_4gpu_5min'))
    p.add_argument('--gpus',default='2,3,4,5');p.add_argument('--cpu-workers',type=int,default=24)
    p.add_argument('--seconds',type=float,default=300);p.add_argument('--detach',action='store_true');p.add_argument('--gpu-pixels',action='store_true')
    args=p.parse_args()
    if args.seconds<=0 or args.cpu_workers<=0:p.error('时间和进程数必须为正数')
    main(args)
