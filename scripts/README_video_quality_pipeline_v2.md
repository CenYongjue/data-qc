# EgoDex 视频质检 v2

## HiFi-UMI / LeRobot v3

入口现在会自动识别 `meta/info.json`，逐个读取 `meta/episodes/**/*.parquet` 的回合边界，
按每路视频的 from/to_timestamp 对共享MP4分段检查。支持根目录下多个LeRobot分片。
不解压、不重新转换格式、不改源数据。各相机均合格才算回合通过。

```bash
.venvs/agibot_gripper_qc/bin/python -u scripts/video_quality_pipeline_v2.py \
  --input datasets/hifi_umi --output lerobotv3_output/hifi_umi_video_qc_v2 \
  --cpu-workers 24 --resume
```

HiFi-UMI按带状态/动作的夹爪具身数据处理，跳过人手可见性和轨迹运动合理性，
全机器人任务不加载YOLO，也不占用GPU。
HiFi-UMI镜头抖动仅检查head_main和head_main_stereo_right；四路夹爪相机跳过LK、
抖动及运动置信度判定，报告SKIPPED_CAMERA_ROLE，其他画质与完整性检查照常运行。
按此新规则全量重跑的输出目录是 `lerobotv3_output/hifi_umi_video_qc_v2_head_only`。
`review.html` 支持按检出类别、状态筛选，显示每类回合数量并分页播放原速片段/速览。
每30秒读取已有CSV/JSON更新，不重复执行检测，也不重新编码。播放视频时不自动重绘。
原先在 `hifi_umi_video_qc_v2` 启动的9967服务可直接打开 `/head_only/review.html`，
此路径链接到新规则结果，旧结果网页仍保留。

文件级额外检查：Parquet可读、必要列、回合长度、全局/回合索引、timestamp、task_index，
以及显式valid.frame标记。标记为false的帧按导出说明属于无效源帧的黑帧占位，记录FAIL。
相机间检查只验证导出后的共同时间轴覆盖，不能证明原始采集硬件同步或发现补帧掩盖的丢帧。
逐进程缓存轻量标量表，避免每回合重复扫描全部数据；不缓存机器人YOLO图像。

`session_realtime_factor_including_startup`以回合最长相机时长为分子；
`session_camera_realtime_factor`以所有相机时长总和为分子。六路数据的两种倍速相差约六倍。
复核片段限制在对应回合区间内，不跨越拼接视频中的回合边界。

## 五分钟关闭压缩失真的对照入口

```bash
.venvs/agibot_gripper_qc/bin/python -u scripts/hifi_umi_qc_no_block.py
```

默认输入HiFi-UMI，24进程、300秒实际时间预算；结果独立保存到
`lerobotv3_output/hifi_umi_no_block_5min`，完成在途回合后自动汇总。
独立配置中设置 `block_detection=false`，正式配置及默认行为不改变。
关闭块检测时不创建区域滤镜、不计算块分数、不做块判定，报告SKIPPED_DISABLED。
依赖块分数佐证的色度花屏分支报告SKIPPED_MISSING_BLOCK_EVIDENCE；其他检查保留。

## 五分钟NVDEC实验入口

```bash
.venvs/agibot_gripper_qc/bin/python -u scripts/hifi_umi_qc_gpu_decode.py --gpus 2,3
```

默认结果为 `lerobotv3_output/hifi_umi_nvdec_5min`。继承关闭块检测的24进程配置，
用PyAV15的HWAccel(cuda)直接替换解码入口；GPU解码后回CPU运行既有算子。
软件回退被禁止，各相机报告decoder.hardware_accelerated和设备编号。
启动时对每张GPU抽查8帧PTS和320宽灰度差异，写入nvdec_preflight.json；不据此推算性能。
decoder_setup单独记录解码器建立时间，decode包含PyAV硬解和返回CPU帧的开销。
该入口用于实测直接替换的整链路收益，不代表已实现持久化GPU解码服务或GPU图像算子。

## 四卡持久解码后台实验

```bash
.venvs/agibot_gripper_qc/bin/python -u scripts/hifi_umi_qc_persistent_gpu.py --gpus 2,3,4,5 --detach
```

每卡一个持久进程，分别顺序读取连续四分之一回合，复用六路解码器并保留跨回合的下一帧。
24个CPU质检进程通过共享内存读取全帧320宽灰度及4Hz YUV444；队列只传描述符，
最多32条回合在途，CPU完成后清理共享内存。按回合重新初始化LK/曝光等判定状态。
启动后不依赖SSH终端，默认五分钟停止生产新回合，完成在途任务后写入conclusion.md。
默认目录 `lerobotv3_output/hifi_umi_persistent_nvdec_4gpu_5min`，后台日志background.log。
performance.json分别记录GPU解码/准备/初始化和CPU检查用时；整体倍速包括两者。
只用于无块检测、无人手模型的机器人流水线。样本分布与原顺序测试不同，结果需谨慎对比。

## GPU像素算子实验

```bash
.venvs/agibot_gripper_qc/bin/python -u scripts/hifi_umi_qc_gpu_pixels.py --gpus 2,3,4,5 --detach
```

独立输出 `lerobotv3_output/hifi_umi_gpu_pixels_4gpu_5min`，不替换CPU默认流程。
质检专用虚拟环境新增PyNvVideoCodec 2.0.2，没有升级其他依赖或系统驱动。
NV12显存帧通过DLPack交给Torch，批量执行缩放/范围转换、Sobel/Laplacian、曝光、
帧差和亮度统计；仿照FFmpeg参考帧规则产生冻结元数据，CPU仍执行同一套时序阈值。
头部灰度图通过共享内存回CPU做LK；夹爪只回传数值，不回传全帧。
解码器帧缓冲可能复用，批次积攒前先显存内clone，避免像素与PTS错配。
GPU缩放与swscale不保证逐像素一致，首帧验证记录灰度/边缘/清晰度差异；
同图GPU Sobel/Laplacian必须与OpenCV计算一致，否则终止实验。
实验不产生未使用的YLOW/YHIGH/UAVG/VAVG；这些辅助字段为null，块分数花屏分支仍跳过。
PyNvVideoCodec接口不提供逐帧is_corrupt标志，报告明确标记不可用；解码异常会中止测速，
表完整性、显式valid.frame、帧数和时间戳检查保留。这版并不证明损坏视频检出能力与CPU完全等价。

入口为 `video_quality_pipeline_v2.py`，算子在 `video_qc_v2/`，阈值在
`video_qc_v2_config.json`。默认只生成质量报告和复核素材，不修改、裁剪或删除源文件。

## 运行

在 `/data_all/cenyj` 执行：

```bash
.venvs/agibot_gripper_qc/bin/python -u scripts/video_quality_pipeline_v2.py \
  --input datasets/egoDex --output lerobotv3_output/egodex_video_qc_v2_all \
  --gpus 0,1 --resume
```

不传 `--limit` 就递归处理全部已解压 MP4 和同名 HDF5；ZIP 不重复处理。
使用 `--time-budget-seconds 900` 可做约十五分钟的实际运行测速，到时停止派发新回合，
完成在途回合后正常退出。此时状态是 `budget_complete`，不代表整个数据集完成。
`--cpu-workers 24` 可设置24个CPU回合工作进程，默认仍为4；每个解码器仍为2线程，
GPU输入队列有界，每个CPU进程独立接收检测结果。时间预算测试最多保留每进程一条在途回合。
24进程测速目录为 `lerobotv3_output/egodex_video_qc_v2_cpu24_15min`。
按任务轮转，四个 CPU 工作进程，每个解码器两个线程；双 A100 FP16 批量人手检测。
`--resume` 复用配置、权重及源文件大小和修改时间均未改变的已完成记录。
同一输出目录不要同时运行两个任务。配置或模型改变时使用新的输出目录。

## 检查顺序

1. 文件存在、HDF5 可读、必要字段和长度；完整视频解码、PTS 单调性、时间间隔、帧数一致性。
2. 全时间轴低分辨率冻结、亮度跳变及闪烁、LK 全局相机运动分析。
   冻结结束后的跳变用于提高疑似卡顿的复核优先级，正常停顿仅 WARN。
3. 4Hz 的曝光、边缘区域清晰度和原分辨率 FFmpeg blockdetect。
   曝光结合全局亮度、极暗/极亮像素比例与空间覆盖；低纹理不直接算模糊。
   当前模糊阈值由原来的8降低为4，降低敏感度：分数低于阈值且持续两秒才触发候选。
   blockdetect改为4×4等面积区域分别运行，每区以该回合的中位数+6倍稳健MAD
   （且至少2.5）作阈值；至少50%的区域异常持续一秒才触发。
   每帧独立滤镜工作缓冲避免历史帧干扰；不再重复计算旧版全图blockdetect。
   覆盖率是异常分区面积近似，不能等同精确的异常像素比例。
   相对基线能抑制固定规则纹理，但整段恒定存在的压缩失真可能漏检，需结合人工校准。
4. 前置检查未明确 FAIL 时，4Hz YOLO 人手检测；持续至少三秒完全没有手触发复核。
   只要有一只手就不满足“双手均不可见”的条件。机器人数据跳过人手检测。

LK 使用网格分布特征、前后向校验、RANSAC 相似变换，再给予外围内点有限的较高权重。
根据可靠运动轨迹的高频残差和往返运动标记抖动候选；中心点过于集中时降置信度。
无法可靠估计运动保留 WARN，不当作已经确认通过。正常头部运动也可能产生候选，需要人工复核。

多路丢帧检查需要通过 `--manifest` 显式声明同一回合各相机和共同时间基准。
EgoDex 当前单路，因此多相机检查标记 NOT_APPLICABLE，不能据此证明多路同步。
“花屏”当前是解码错误及色度突变伴块状异常的启发式检查，不能覆盖所有语义花屏。
所有画面阈值均为第一版可调初值，REVIEW 表示候选，不能直接当作已确认坏数据。

## 输出与效率

- `index.html`：全量进度与任务入口，运行时每约三十秒更新。
- `tasks/*.html`：各任务问题片段、速览及每条报告链接。
- `episodes/<任务>/*.json`：逐回合原因、异常时间区间、采样指标与各环节耗时。
- `review/<任务>/`：问题原速 MP4、4Hz 采样的两倍速速览、截图。
- `traces/`：仅保存需要复核的相机运动轨迹。
- `episodes.csv`、`progress.json`：汇总与当前效率；`results.sqlite` 支持断点续跑。
- `performance.json`：本轮结束后的效率及 GPU 批次统计；运行期间以 progress.json 为准。

只为 REVIEW/FAIL 输出复核视频，每路最多两个六秒问题片段，避免给全部通过样本编码。
不能解码的源文件可能无法输出视频，这时保留错误原因和导出错误。
效率为本次会话处理的视频时长除以总墙钟时间，包含加载、检查、推理、复核视频编码和报告。
启动初期及极小样本的倍速不代表稳定吞吐。
十五分钟调整版结果输出在 `lerobotv3_output/egodex_video_qc_v2_area_blur4_15min`，
旧版暂停结果仍在 `lerobotv3_output/egodex_video_qc_v2_all`；两者阈值和blockdetect评分方式不同。
`egodex_video_qc_v2_area_15min` 是误将模糊敏感度提高后的中止运行，不用于最终测速结论。
完成后可用 `python scripts/summarize_video_qc_run.py 输出目录` 生成 `benchmark_summary.json`，
其中包含实测倍速、问题分布和各环节累计工作时间；去掉导出的倍速明确标记为估算。

服务器查看：

```bash
cd /data_all/cenyj/lerobotv3_output/egodex_video_qc_v2_all
python3 -m http.server 9966 --bind 127.0.0.1
```

本地 SSH 转发 `ssh -L 9966:127.0.0.1:9966 用户名@服务器`，浏览器打开
`http://127.0.0.1:9966/`。
