#!/usr/bin/env python3
"""从已完成PASS报告抽样，导出原速、原尺寸六路拼接视频供人工查漏。

只读取现有指标，不重新执行质检。固定随机种子，保存抽样依据和原回合索引。
六个视角各保留640x512像素；仅增加标签条，再以CRF16编码复核副本。
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import html
import json
from pathlib import Path
import random
import sqlite3
import subprocess
import numpy as np

VIEWS=['head_main','head_main_stereo_right','left_hand_down','left_hand_up','right_hand_down','right_hand_up']


def window_stat(values,mode):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):return None
    # 用约两秒窗口均值，避免只因单个瞬时帧就入选边界抽样。
    n=min(9,len(values));means=np.convolve(values,np.ones(n)/n,mode='valid')
    return float(np.min(means) if mode=='min' else np.max(means))


def metrics(report):
    out={}
    for name,c in report['cameras'].items():
        samples=c['samples']
        for metric,field,mode in [('sharpness','sharpness','min'),('darkness','brightness','min'),('brightness','brightness','max'),('block_area','block_area_fraction','max')]:
            value=window_stat([s.get(field) if s.get(field) is not None else float('nan') for s in samples],mode)
            if value is not None:out[metric]=min(out.get(metric,value),value) if mode=='min' else max(out.get(metric,value),value)
        if samples:out['low_texture']=max(out.get('low_texture',0),np.mean([s.get('low_texture',False) for s in samples]))
        if name.startswith('head_'):out['head_motion']=max(out.get('head_motion',0),c.get('motion',{}).get('peak_translation_residual_diagonal',0))
    return out


def export_one(item,output):
    r=item['report'];sources=r['source_segments'];relative=Path('pass_samples')/(r['id']+'.mp4');target=output/relative
    target.parent.mkdir(exist_ok=True)
    duration=min(s['end_seconds']-s['start_seconds'] for s in sources.values())
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-filter_complex_threads','1']
    for name in VIEWS:
        s=sources[name]
        command+=['-threads','1','-ss',f'{s["start_seconds"]:.9f}','-t',f'{duration:.9f}','-i',s['path']]
    filters=[]
    for i,name in enumerate(VIEWS):
        filters.append(f'[{i}:v]setpts=PTS-STARTPTS,pad=iw:ih+24:0:24:black,drawtext=text={name}:fontcolor=white:fontsize=17:x=8:y=3[v{i}]')
    filters.append(''.join(f'[v{i}]' for i in range(6))+'xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0:shortest=1[v]')
    command+=['-filter_complex',';'.join(filters),'-map','[v]','-an','-t',f'{duration:.9f}',
              '-c:v','libx264','-preset','veryfast','-crf','16','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(f'{r["id"]}: {result.stderr[-1500:]}')
    return {'id':r['id'],'task':item['task'],'selection':item['selection'],'metric':item.get('metric'),
            'metric_value':item.get('metric_value'),'task_description':r.get('task_description',[]),
            'duration_seconds':duration,'video':str(relative),'report':item['report_path'],
            'source_segments':sources,'metrics':item['metrics']}


def page(output,items):
    def esc(v):return html.escape(str(v),quote=True)
    cards=[]
    for i,r in enumerate(items,1):
        metric=f'；指标 {r["metric"]}={r["metric_value"]:.4g}' if r['metric'] else ''
        cards.append(f'<section><h2>{i}. {esc(r["id"])}</h2><p>{esc(r["selection"]+metric)}</p><p>{esc(" / ".join(r["task_description"]))}</p><p>原速，整回合 {r["duration_seconds"]:.2f} 秒；上排头部双目、左夹爪下；下排左夹爪上、右夹爪下、右夹爪上。各视角保留原尺寸。</p><video controls preload="none" src="{esc(r["video"])}"></video><p><a href="{esc(r["report"])}">原质检指标JSON</a></p></section>')
    weak=sum(r['selection'].startswith('指标偏弱抽样') for r in items)
    random_count=sum(r['selection'].startswith('跨任务随机抽样') for r in items)
    intro=f'{weak}条指标排序样本＋{random_count}条跨任务随机样本＋{len(items)-weak-random_count}条补充样本。指标排序只表示相对位置，不等于触发异常；全体同分或全零时不标记“偏高/偏低”。'
    (output/'pass_samples.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>通过样本人工查漏</title><style>body{max-width:1600px;margin:24px auto;padding:12px;font:16px sans-serif;line-height:1.6}section{border-top:1px solid #ccc;padding:16px 0}video{width:100%;max-width:1920px}a{margin-right:16px}</style><h1>通过样本人工查漏：12条</h1><p><a href="review.html">返回分类复核</a><a href="index.html">全量结果</a><a href="pass_samples/manifest.json">抽样清单与源区间</a></p><p>'+intro+'固定随机种子20261029。该样本不是无偏统计验证集，不能直接用于估算整体漏检率。视频原速、整回合、六路原尺寸拼接，CRF16重编码，未重新执行检测。</p>'+''.join(cards)+'</html>')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args();out=args.output.resolve()
    db=sqlite3.connect(out/'results.sqlite');rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM records')];db.close()
    passed=[r for r in rows if r['status']=='PASS'];candidates=[]
    for row in passed:
        report=json.loads((out/row['report']).read_text())
        candidates.append({'task':row['task'],'report_path':row['report'],'report':report,'metrics':metrics(report)})
    chosen=[];ids=set();tasks=set()
    for metric,reverse,label in [('sharpness',False,'清晰度指标偏低'),('darkness',False,'亮度偏低'),('brightness',True,'亮度偏高'),('block_area',True,'块状异常覆盖率偏高'),('head_motion',True,'头部运动残差偏高'),('low_texture',True,'低纹理比例偏高')]:
        ranking=sorted([x for x in candidates if metric in x['metrics'] and x['report']['id'] not in ids],key=lambda x:x['metrics'][metric],reverse=reverse)
        pick=next((x for x in ranking if x['task'] not in tasks),ranking[0])
        values=[x['metrics'][metric] for x in ranking]
        # 排名第一也可能是全体同为零，不能据此声称指标偏高。
        selection=('补充通过样本：该指标全体同分，未发现相对偏弱证据' if max(values)-min(values)<1e-12 else '指标偏弱抽样：'+label)
        chosen.append({**pick,'selection':selection,'metric':metric,'metric_value':float(pick['metrics'][metric])});ids.add(pick['report']['id']);tasks.add(pick['task'])
    rng=random.Random(20261029);remaining=[x for x in candidates if x['task'] not in tasks];rng.shuffle(remaining)
    for x in remaining:
        if x['task'] in tasks:continue
        chosen.append({**x,'selection':'跨任务随机抽样'});tasks.add(x['task'])
        if len(chosen)==12:break
    # 只对12条人工查漏样本编码；最多4个FFmpeg并发，每个编码器2线程。
    results={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(export_one,x,out) for x in chosen]
        for future in as_completed(futures):
            r=future.result();results[r['id']]=r;print('EXPORTED '+r['id'],flush=True)
    items=[results[x['report']['id']] for x in chosen]
    (out/'pass_samples/manifest.json').write_text(json.dumps({'seed':20261029,'items':items},ensure_ascii=False,indent=2))
    page(out,items)
    for name in ['index.html','review.html']:
        path=out/name
        if path.exists():
            text=path.read_text();link='<p><a href="pass_samples.html">通过样本人工查漏：12条六路视频</a></p>'
            if link not in text:
                pos=text.index('</h1>')+5;text=text[:pos]+link+text[pos:];path.write_text(text)
    print('DONE '+str(out/'pass_samples.html'),flush=True)


if __name__=='__main__':main()
