#!/usr/bin/env python3
"""选择实际裁剪时长最大的PASS回合，导出原速独立视频和同步并排对照。

并排视频按源时间对齐；被删区间在右侧变黑并注明CUT，不冻结画面假装仍在播放。
右侧保留画面直接读取已导出的LeRobot数据，展示真实切分结果。
"""
import argparse
import html
import json
from pathlib import Path
import subprocess
import time
import av
import cv2
import numpy as np
import pyarrow.parquet as pq
import agibot_quality_with_gripper as core


class WideMovie:
    def __init__(self,path,fps):
        self.process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-n',
            '-f','rawvideo','-pix_fmt','bgr24','-s','1920x330','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',
            '-threads','2','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def write(self,image):self.process.stdin.write(image.tobytes())
    def close(self):
        self.process.stdin.close()
        if self.process.wait():raise RuntimeError('Video encoder failed')


def export_one(source,source_info,ep,cleaned,clean_info,new_ep,report,out,context):
    fps=source_info['fps'];n=ep['length'];trim=report['trim_proposal'];start=trim['start_frame'];end=trim['end_frame_exclusive']
    pad=round(context*fps);boundaries=[]
    if start>0:boundaries.append(('head',0,min(end,start+pad),start))
    if end<n:boundaries.append(('tail',max(start,end-pad),n,end))
    results=[];reader=core.Reader()
    try:
        for side,a,b,boundary in boundaries:
            stem=f'episode_{ep["episode_index"]:06d}_{side}'
            names={key:f'{stem}_{key}.mp4' for key in ['before','after','side_by_side']}
            before=core.Movie(out/names['before'],fps);after=core.Movie(out/names['after'],fps)
            combined=WideMovie(out/names['side_by_side'],fps)
            try:
                for fi in range(a,b):
                    t=fi/fps;kept=start<=fi<end
                    originals=[reader.frame(core.video_path(source,source_info,ep,k),ep[f'videos/{k}/from_timestamp']+t,.5/fps)[0] for k in core.CAMERAS]
                    left=core.montage(originals,ep['episode_index'],t,f'BEFORE / source frame={fi} / boundary={boundary}')
                    before.write(left)
                    if kept:
                        new_fi=fi-start
                        frames=[reader.frame(core.video_path(cleaned,clean_info,new_ep,k),new_ep[f'videos/{k}/from_timestamp']+new_fi/fps,.5/fps)[0] for k in core.CAMERAS]
                        right=core.montage(frames,ep['episode_index'],t,f'AFTER / source frame={fi} / new frame={new_fi}')
                        after.write(right)
                    else:
                        right=np.zeros((300,960,3),np.uint8)
                        cv2.putText(right,'CUT: removed from cleaned episode',(35,115),cv2.FONT_HERSHEY_SIMPLEX,1,(0,210,255),2)
                        cv2.putText(right,f'source frame={fi} / source time={t:.3f}s',(35,170),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                    canvas=np.zeros((330,1920,3),np.uint8);canvas[30:,:960]=left;canvas[30:,960:]=right
                    cv2.putText(canvas,'BEFORE / ORIGINAL SPEED',(15,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                    cv2.putText(canvas,'AFTER / ALIGNED BY SOURCE TIME',(975,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
                    combined.write(canvas)
            finally:
                before.close();after.close();combined.close()
            results.append({'source_episode_index':ep['episode_index'],'cleaned_episode_index':new_ep['episode_index'],
                'side':side,'boundary_frame':boundary,'source_context_frames':[a,b],
                'kept_source_frames':[max(a,start),min(b,end)],'removed_frames':start if side=='head' else n-end,
                'removed_seconds':(start if side=='head' else n-end)/fps,'source_episode_frames':n,
                'paths':names,'fps':fps,'alignment':'source time; deleted regions are black on AFTER, not held frames'})
    finally:reader.close()
    return results


def main():
    started=time.perf_counter();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--count',type=int,default=4)
    p.add_argument('--context-seconds',type=float,default=4.)
    args=p.parse_args();root=args.result.resolve();out=root/'report/large_trim_comparisons'
    if out.exists():raise FileExistsError(out)
    out.mkdir();cv2.setNumThreads(1)
    summary=json.loads((root/'report/summary.json').read_text());source=Path(summary['input']);cleaned=root/'dataset'
    source_info=json.loads((source/'meta/info.json').read_text());clean_info=json.loads((cleaned/'meta/info.json').read_text())
    source_eps={ep['episode_index']:ep for ep in pq.read_table(source/'meta/episodes').to_pylist()}
    new_eps={ep['source_episode_index']:ep for ep in pq.read_table(cleaned/'meta/episodes').to_pylist()}
    reports=[json.loads(f.read_text()) for f in (root/'report/episodes').glob('*.json')]
    candidates=[r for r in reports if r.get('exported_episode_index') is not None and r['trim_proposal']['changed']]
    selected=sorted(candidates,key=lambda r:(-r['trim_proposal']['removed_seconds'],r['episode_index']))[:args.count]
    results=[]
    for r in selected:
        i=r['episode_index'];results.extend(export_one(source,source_info,source_eps[i],cleaned,clean_info,new_eps[i],r,out,args.context_seconds))
        print(f'Exported episode {i}: removed {r["trim_proposal"]["removed_seconds"]:.3f}s',flush=True)
    cards=[]
    for r in results:
        i=r['source_episode_index'];a,b=r['source_context_frames'];ka,kb=r['kept_source_frames']
        cards.append(f'<section><h2>源回合 {i} / {"裁头" if r["side"]=="head" else "裁尾"} {r["removed_seconds"]:.3f} 秒（{r["removed_frames"]} 帧）</h2>'
            f'<p>原回合 {r["source_episode_frames"]} 帧；边界为源帧 {r["boundary_frame"]}。展示源区间 [{a}, {b})，保留区间 [{ka}, {kb})。</p>'
            f'<h3>同步并排对比：左原始，右清洗后</h3><video controls preload="none" src="{r["paths"]["side_by_side"]}"></video>'
            f'<p><a href="{r["paths"]["side_by_side"]}" download>下载并排MP4</a> · <a href="{r["paths"]["before"]}" download>下载裁剪前片段</a> · <a href="{r["paths"]["after"]}" download>下载裁剪后片段</a></p>'
            f'<details><summary>分别播放前后片段</summary><h3>裁剪前（含删除区间）</h3><video controls preload="none" src="{r["paths"]["before"]}"></video>'
            f'<h3>裁剪后（只播放保留内容）</h3><video controls preload="none" src="{r["paths"]["after"]}"></video></details></section>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AgiBot较大幅度边界裁剪对照</title>'
        '<style>body{max-width:1600px;margin:24px auto;font:16px sans-serif;background:#f5f6f8}section{background:white;padding:20px;margin:24px 0}video{width:100%}</style>'
        '<h1>实际裁剪时长最大的回合</h1><p>从已导出的PASS回合按裁剪总秒数降序选取。视频均为原速，各侧包含三路相机。</p>'
        '<p>并排视频按源时间对齐；右侧黑屏CUT表示该时间已从清洗回合中删除，并非视频损坏或冻结。右侧有效画面直接读取清洗数据集。</p>'
        '<p><a href="../index.html">全部质检报告</a> · <a href="../trim_comparisons.html">其他裁剪对照（包含裁头）</a></p>'+''.join(cards))
    result={'sorting':'actual removed seconds descending among exported PASS episodes','samples':results,'seconds':time.perf_counter()-started}
    core.dump(out/'selection.json',result)
    with (root/'report/index.html').open('a') as f:f.write('<p><a href="large_trim_comparisons/index.html">裁剪幅度最大的回合：原速同步并排对照</a></p>')
    print(f'DONE: {out}/index.html; {len(results)} comparisons; {result["seconds"]:.2f}s',flush=True)


if __name__=='__main__':main()
