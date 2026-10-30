"""视频质检算子。所有分数都是检测依据，不修改图像或训练数据。

设计原则：便宜的时域分析复用320宽灰度图；blockdetect只在4Hz原分辨率Y平面运行；
运动估计可靠性与“运动过于抖动”分别报告，避免把低纹理或操作前景当相机故障。
"""
from pathlib import Path
import math
import time
import av
import cv2
import numpy as np


def intervals(times,mask,min_duration=0.,max_gap=None):
    """将实际时间戳上的连续证据合并；不跨缺失帧/分析中断拼接异常。"""
    out=[];start=None;last=None
    for index in range(len(times)+1):
        yes=index<len(times) and bool(mask[index])
        gap=yes and last is not None and max_gap is not None and times[index]-last>max_gap
        if start is not None and (not yes or gap):
            if last-start+1e-6>=min_duration:out.append({'start_seconds':float(start),'end_seconds':float(last),'duration_seconds':float(last-start)})
            start=last=None
        if yes:
            if start is None:start=times[index]
            last=times[index]
    return out


class MetadataFilter:
    """用PyAV运行现成FFmpeg滤镜；metadata滤镜把分数写到小临时文件。

    PyAV15未公开AVFrame.metadata，因此使用FFmpeg自身的metadata=print接口。
    同一解码帧直接送入滤镜图，没有另开FFmpeg重新解码源视频。direct=1便于及时读取。
    """
    def __init__(self,frame,name,options,path):
        self.path=Path(path);self.graph=av.filter.Graph()
        self.source=self.graph.add_buffer(template=frame)
        filt=self.graph.add(name,options)
        metadata=self.graph.add('metadata',f'mode=print:file={self.path}:direct=1')
        self.sink=self.graph.add('buffersink')
        if name=='signalstats':
            scene=self.graph.add('scdet','threshold=15')
            self.graph.link_nodes(self.source,filt,scene,metadata,self.sink)
        else:self.graph.link_nodes(self.source,filt,metadata,self.sink)
        self.graph.configure()
    def push(self,frame):
        self.source.push(frame)
        while True:
            try:self.sink.pull()
            except (av.error.BlockingIOError,av.error.EOFError):break
    def finish(self):
        self.push(None)
        rows=[];current=None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.startswith('frame:'):
                    if current is not None:rows.append(current)
                    current={}
                    for token in line.split():
                        if token.startswith('pts_time:'):current['seconds']=float(token.split(':',1)[1])
                elif '=' in line and current is not None:
                    key,value=line.split('=',1)
                    try:current[key]=float(value)
                    except ValueError:pass
            if current is not None:rows.append(current)
        return rows


def spatial_block_scores(frame,grid,period_min,period_max,path):
    """原分辨率分区调用FFmpeg blockdetect，不把全局分数误当作异常面积。

    各区独立评分，保留整幅画面坐标下的分区顺序。逐帧独立构建滤镜，避免
    blockdetect内部梯度工作缓冲跨帧影响分数；每次只保存一个很小的元数据文件。
    不缩放像素，否则原始压缩块周期会改变。这里增加的开销单独计入blockdetect。
    """
    gray=frame.reformat(format='gray').to_ndarray();h,w=gray.shape;values=[]
    for gy in range(grid):
        for gx in range(grid):
            tile=np.ascontiguousarray(gray[gy*h//grid:(gy+1)*h//grid,gx*w//grid:(gx+1)*w//grid])
            local=av.VideoFrame.from_ndarray(tile,format='gray');local.pts=frame.pts;local.time_base=frame.time_base
            filt=MetadataFilter(local,'blockdetect',f'planes=1:period_min={period_min}:period_max={period_max}',path)
            filt.push(local);rows=filt.finish()
            if len(rows)!=1 or 'lavfi.block' not in rows[0]:
                raise RuntimeError('分区blockdetect未返回分数：算子故障，不代表视频损坏')
            score=rows[0]['lavfi.block']
            # 纯色/规则区域的非边界梯度分母可能为0，FFmpeg返回inf/nan。
            # 此时只能记为不可评分，不能报解码损坏，也不能直接算成异常区域。
            values.append(score if math.isfinite(score) else None)
    return values


class CameraMotion:
    """低分辨率LK + 前后向检查 + RANSAC + 内点加权相似变换。

    变换只有[a,-b,tx; b,a,ty]四个参数，不用高自由度单应矩阵追随操作物体。
    RANSAC之后才做权重拟合；OpenCV的estimateAffinePartial2D本身没有点权重参数。
    """
    def __init__(self,config):
        self.c=config;self.previous=None;self.points=None;self.previous_t=None;self.last_reseed=-1e9
        self.last_orb=-1e9;self.orb=cv2.ORB_create(nfeatures=config['max_points']*2)
        self.rows=[];self.segment=0;self.global_matrix=np.eye(3);self.segment_start=0.
        self.stats={'orb_calls':0,'reseed_calls':0,'identical_pairs_skipped':0}
    def seed(self,gray):
        # 网格配额防止中央角点数量过多；只在补点时调用，不每帧重建描述子。
        h,w=gray.shape;points=[];grid=self.c['grid'];limit=math.ceil(self.c['max_points']/(grid*grid))
        margin=self.c['border_pixels']
        for gy in range(grid):
            for gx in range(grid):
                x0=max(margin,gx*w//grid);x1=min(w-margin,(gx+1)*w//grid)
                y0=max(margin,gy*h//grid);y1=min(h-margin,(gy+1)*h//grid)
                if x1<=x0 or y1<=y0:continue
                p=cv2.goodFeaturesToTrack(gray[y0:y1,x0:x1],maxCorners=limit,qualityLevel=.015,minDistance=5,blockSize=5)
                if p is not None:points.extend((p.reshape(-1,2)+[x0,y0]).tolist())
        self.stats['reseed_calls']+=1
        return np.asarray(points,np.float32).reshape(-1,1,2)
    def orb_matches(self,previous,current):
        p,d1=self.orb.detectAndCompute(previous,None);q,d2=self.orb.detectAndCompute(current,None)
        if d1 is None or d2 is None:return None,None
        matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
        good=[v[0] for v in matches if len(v)==2 and v[0].distance<.7*v[1].distance]
        if len(good)<self.c['min_inliers']:return None,None
        return np.float32([p[v.queryIdx].pt for v in good]),np.float32([q[v.trainIdx].pt for v in good])
    @staticmethod
    def weighted_similarity(p,q,weights):
        # 四参数加权最小二乘，规模至多120点，不引入重型优化器。
        x,y=p[:,0],p[:,1];A=np.zeros((len(p)*2,4));b=q.reshape(-1)
        A[0::2]=np.c_[x,-y,np.ones(len(p)),np.zeros(len(p))]
        A[1::2]=np.c_[y,x,np.zeros(len(p)),np.ones(len(p))]
        weight=np.repeat(np.sqrt(weights),2)
        a,beta,tx,ty=np.linalg.lstsq(A*weight[:,None],b*weight,rcond=None)[0]
        return np.array([[a,-beta,tx],[beta,a,ty]],float)
    def push(self,gray,t):
        h,w=gray.shape;c=self.c
        if self.previous is None:
            self.previous=gray.copy();self.previous_t=t;self.points=self.seed(gray);self.last_reseed=t;return
        dt=t-self.previous_t
        valid_dt=dt>0 and dt<c['max_frame_gap_seconds']
        row={'seconds':float(t),'dt':float(dt),'reliable':False,'segment':self.segment,'confidence':0.,'reason':'tracking_failed'}
        p=q=None
        same=valid_dt and np.array_equal(gray,self.previous)
        if same:
            self.stats['identical_pairs_skipped']+=1
            # 数字图像完全相同，观测运动为0；不反复跑LK，但冻结由单独算子判断。
            p=self.points.reshape(-1,2) if self.points is not None else None
            q=p.copy() if p is not None else None
        elif valid_dt and self.points is not None and len(self.points)>=c['min_inliers']:
            next_points,ok,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None,winSize=(21,21),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,15,.03))
            back,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,next_points,None,winSize=(21,21),maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,15,.03))
            keep=(ok.ravel()>0)&(back_ok.ravel()>0)&(np.linalg.norm(back-self.points,axis=2).ravel()<c['fb_error_pixels'])
            a=self.points.reshape(-1,2);b=next_points.reshape(-1,2)
            keep&=(b[:,0]>=c['border_pixels'])&(b[:,0]<w-c['border_pixels'])&(b[:,1]>=c['border_pixels'])&(b[:,1]<h-c['border_pixels'])
            p=a[keep];q=b[keep]
        # ORB只在图像有纹理但LK失效时限频调用；低纹理下不做无效重复恢复。
        if valid_dt and (p is None or len(p)<c['min_inliers']) and t-self.last_orb>=c['orb_cooldown_seconds']:
            if float(cv2.Laplacian(gray,cv2.CV_32F).var())>c['orb_min_texture']:
                self.last_orb=t;self.stats['orb_calls']+=1;p,q=self.orb_matches(self.previous,gray)
        if valid_dt and p is not None and len(p)>=c['min_inliers']:
            transform,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=c['ransac_pixels'],maxIters=500,confidence=.99,refineIters=5)
            if transform is not None:
                keep=inliers.ravel()>0;pi=p[keep];qi=q[keep];n=len(pi)
                central=((pi[:,0]>.25*w)&(pi[:,0]<.75*w)&(pi[:,1]>.25*h)&(pi[:,1]<.75*h)).mean() if n else 1.
                occupied=len({(min(c['grid']-1,int(x/w*c['grid'])),min(c['grid']-1,int(y/h*c['grid']))) for x,y in pi})
                area=float(cv2.contourArea(cv2.convexHull(pi)))/(w*h) if n>=3 else 0.
                error=np.linalg.norm(pi@transform[:,:2].T+transform[:,2]-qi,axis=1)
                median_error=float(np.median(error)) if n else 1e6;ratio=n/len(p)
                reliable=n>=c['min_inliers'] and ratio>=c['min_inlier_ratio'] and occupied>=c['min_grid_cells'] and area>=c['min_coverage'] and central<=c['max_central_fraction'] and median_error<c['max_fit_error_pixels']
                row.update(inliers=n,inlier_ratio=float(ratio),grid_cells=occupied,coverage=area,central_fraction=float(central),fit_error_pixels=median_error)
                if reliable:
                    radial=np.maximum(abs(pi[:,0]/w-.5)*2,abs(pi[:,1]/h-.5)*2)
                    weights=(1+c['peripheral_weight_boost']*radial)/(1+(error/c['ransac_pixels'])**2)
                    transform=self.weighted_similarity(pi,qi,weights)
                    matrix=np.vstack([transform,[0,0,1]])
                    # 局部窗口重置累积漂移；不把重置造成的边界当突变。
                    if t-self.segment_start>c['trajectory_window_seconds']:
                        self.segment+=1;self.global_matrix=np.eye(3);self.segment_start=t
                    self.global_matrix=matrix@self.global_matrix
                    center=np.array([w/2,h/2,1.]);position=self.global_matrix@center
                    diag=math.hypot(w,h)
                    scale=math.hypot(self.global_matrix[0,0],self.global_matrix[1,0])
                    row.update(reliable=True,reason='ok',segment=self.segment,confidence=float(min(1.,ratio)*min(1.,area/.3)),
                        x=float((position[0]-w/2)/diag),y=float((position[1]-h/2)/diag),
                        angle=float(math.atan2(self.global_matrix[1,0],self.global_matrix[0,0])),scale=float(scale))
                else:row['reason']='insufficient_spatial_support'
                self.points=qi.reshape(-1,1,2).astype(np.float32)
        if not row['reliable']:
            self.segment+=1;self.global_matrix=np.eye(3);self.segment_start=t
        self.rows.append(row)
        if self.points is None or len(self.points)<c['max_points']//2 or t-self.last_reseed>=c['reseed_seconds']:
            self.points=self.seed(gray);self.last_reseed=t
        self.previous=gray.copy();self.previous_t=t
    def summarize(self):
        rows=self.rows;c=self.c
        if not rows:return {'scorable':False,'unreliable_intervals':[],'shake_intervals':[]}
        times=np.array([r['seconds'] for r in rows]);reliable=np.array([r['reliable'] for r in rows]);flags=np.zeros(len(rows),bool)
        residual_peak=0.;groups={}
        for i,row in enumerate(rows):
            if row['reliable']:groups.setdefault(row['segment'],[]).append(i)
        for indices in groups.values():
            if len(indices)<8:continue
            ts=times[indices];dt=np.median(np.diff(ts))
            if dt<=0:continue
            values=np.array([[rows[i]['x'],rows[i]['y'],rows[i]['angle']] for i in indices]);values[:,2]=np.unwrap(values[:,2])
            window=max(3,round(c['smoothing_seconds']/dt));window+=1-window%2
            if len(values)<window+2:continue
            smooth=np.stack([np.convolve(np.pad(col,(window//2,window//2),mode='edge'),np.ones(window)/window,mode='valid') for col in values.T],axis=1)
            residual=values-smooth;translation=np.linalg.norm(residual[:,:2],axis=1);rotation=abs(residual[:,2])
            residual_peak=max(residual_peak,float(translation.max()))
            amplitudes=(translation>c['translation_residual_diagonal'])|(rotation>math.radians(c['rotation_residual_degrees']))
            velocity=np.diff(values,axis=0)/np.diff(ts)[:,None]
            # 忽略几乎为零的变化，避免估计噪声带来大量虚假换向。
            useful=np.linalg.norm(velocity[:,:2],axis=1)>c['min_velocity_diagonal_per_second']
            reversals=np.zeros(len(values),bool)
            for j in range(2,len(values)):
                a,b=velocity[j-2],velocity[j-1]
                translation_reverse=useful[j-2] and useful[j-1] and np.dot(a[:2],b[:2])<0
                rotation_reverse=abs(a[2])>c['min_rotation_velocity'] and abs(b[2])>c['min_rotation_velocity'] and a[2]*b[2]<0
                reversals[j]=translation_reverse or rotation_reverse
            width=max(3,round(c['decision_window_seconds']/dt))
            counts=np.convolve(np.pad(reversals.astype(int),(width//2,width-1-width//2)),np.ones(width),mode='valid')
            bad=amplitudes&(counts>=c['min_reversals'])
            bad[:window//2]=False;bad[-(window//2):]=False;flags[np.asarray(indices)]=bad
        return {'scorable':bool(reliable.any()),'reliable_fraction':float(reliable.mean()),'statistics':self.stats,
                'peak_translation_residual_diagonal':residual_peak,
                'unreliable_intervals':intervals(times,~reliable,c['unreliable_run_seconds'],c['max_frame_gap_seconds']),
                'shake_intervals':intervals(times,flags,c['shake_run_seconds'],c['max_frame_gap_seconds'])}
