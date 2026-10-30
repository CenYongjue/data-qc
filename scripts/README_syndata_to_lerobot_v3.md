# SynData → LeRobot v3.0

入口：`syndata_to_lerobot_v3.py`。适配 SynData 的 Zarr v3（bytes + zstd、完整空间分块），不是任意数据格式的通用转换器。不支持的布局明确报错，不静默补零。

```bash
cd /data_all/cenyj
source conda.sh
conda activate intern_pipeline
python -u scripts/syndata_to_lerobot_v3.py \
  --source syndataset/glove-rep-bg/tasks/task_0007/000001.zarr.tar \
  --output lerobotv3_output/syndata \
  --fps 10
```

依赖：numpy、numcodecs、pyarrow、pandas、datasets、av 和系统 ffmpeg（libx264rgb）。不需要转换环境安装 PyTorch/LeRobot。

## 输入

- `--source` 可以是 `.zarr.tar` 或已经解压的 `.zarr` 目录。压缩包先安全解压到同目录；已有目录不覆盖，随后完整检查各分块。
- 输入应位于 `<subset>/tasks/<task_key>/<volume>.zarr[.tar]`。
- 必需元数据：`<subset>/index/clips.parquet`、`annotations/clip_annotations.parquet`、`task.json`。无需步骤标注。
- `start_idx`、`end_idx` 是包含两端的源帧范围，必须满足 `num_frames=end_idx-start_idx+1`。

## 转换与质检

- 每条有效 clip 对应一个 episode；10 fps，根据帧序号生成 episode 内时间。不声称恢复或检查原始采集时钟。
- 完整检查所有分块的存在性、解压、长度、NaN/Inf，报告零值比例、负值和四元数范数异常。深度零值保留，不视为损坏。
- 无法读取或包含非有限数值的必需模态使受影响 clip 整条被剔除，原因写入 `qc/report.json`。未受影响 clip 保留；不修改源文件。
- 原始数值、形状和 float32 类型保持不变；不推测单位或坐标系，不做归一化、重采样、动作构造。
- `observation.<原始字段名>` 保存数值；RGB 为 `observation.images.head_rgb`。
- 深度采用 Hugging Face Array2D float32 写入 Parquet，零值与小数精度保留。
- RGB 使用 MP4 / lossless libx264rgb / CRF 0，帧尺寸不变，较有损编码占用更大。
- 任务名称对应 `task_index`；英文摘要、clip ID、子集、任务号、来源路径及源帧范围放在 episode 元数据；逐帧保存 `source_frame_index`。
- `meta/stats.json` 与 episode 元数据包含全量 min/max/mean/std/count，深度按像素统计并包含零值，RGB 为 [0,1] 每通道统计。不计算分位数。
- `--batch-frames` 默认 8；Parquet / 视频分别以 100 / 200 MiB 为软分片阈值，在 episode 结束后轮转，单条 episode 不拆到不同文件。
- 结果先写 `<output>.inprogress`，默认通过必要结构验证后才改名为 `<output>`：检查全部标量索引和时间戳、字段 schema、episode 边界、视频头信息中的帧数和时长。不重复比较所有深度像素或完整解码全部视频。
- 可选 `--verification full` 才执行数值逐值比较及视频完整解码逐帧比较。本次按用户要求采用 standard 验证。
- 已有输出或暂存目录均拒绝覆盖。失败时保留暂存目录供诊断。`--finalize-existing` 仅对已经全部写完的暂存输出执行验证和最终改名，不重新转换；不能恢复未写完的数据。
- `--reuse-qc-report <report.json>` 可复用已完成的质检报告，仅限确认源文件未改变时使用；检查路径、模态和形状，但不对源文件内容重新做校验。它只省略输入质检，不省略所选输出验证，也不续写部分转换文件。

## 输出

标准 `meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`meta/episodes/...`、`data/...` 和 `videos/...`。
附加 `meta/source.json`、`qc/report.json`、`qc/verification.json`。
输出没有 `action`，也没有强行拼接的 `observation.state`；用于后续清洗及特征构造。

可选数值与视频全量核验内置在脚本中。集成测试覆盖末帧边界、缺失块导致的 clip 剔除、非连续源帧映射、统计计算及拒绝覆盖：

```bash
python scripts/test_syndata_to_lerobot_v3.py
```

用已有的官方 LeRobot 环境抽查第一个、中间、最后一个 episode 的首、中、尾帧（默认共 9 帧；可选 `--all-episodes` 扩大范围）：

```bash
HF_HOME=/data_all/cenyj/.cache/lerobot_validation \
  /data_all/cenyj/tools/miniforge3/envs/lerobot/bin/python \
  scripts/validate_lerobot_v3_reader.py lerobotv3_output/syndata
```

官方读取验证需要 LeRobot 及其数据依赖。本次使用本地 LeRobot 源码、datasets 4.8.5、PyAV 15.1.0；报告保存为 `qc/official_reader_verification.json`。

来源：[PsiBotAI/SynData](https://huggingface.co/datasets/PsiBotAI/SynData)，CC-BY-4.0。
