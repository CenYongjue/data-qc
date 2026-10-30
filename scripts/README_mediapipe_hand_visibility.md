# MediaPipe 真人手出镜验证

独立环境：`/data_all/cenyj/.venvs/mediapipe_hand_qc`，Python 3.11，`include-system-site-packages=false`。未向 intern_pipeline 或系统 Python 安装包。依赖版本保存在 `requirements_mediapipe_hand_qc.txt`。

官方模型存放于 `models/mediapipe/hand_landmarker.task`，来自：
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

```bash
cd /data_all/cenyj
MPLCONFIGDIR=/data_all/cenyj/.cache/mediapipe_matplotlib \
  .venvs/mediapipe_hand_qc/bin/python -u scripts/mediapipe_hand_visibility.py \
  --video sample_video/video_20261026_192204.mp4 \
  --output sample_video/mediapipe_hand_qc
```

已有输出目录不覆盖；重跑时指定新目录。默认 CPU 推理、10 Hz 抽样，检测最多两只手，检测/presence/跟踪阈值均为官方默认的 0.5。使用 VIDEO 模式和真实 PTS 时间戳，推理图像 960x540。

输出：

- `index.html`：复查入口。
- `annotated.mp4`：10 fps 抽样标注预览，包含关键点、框、模型左右手分类及时间；无音频，细微帧率变化以画面 PTS 为准。
- `samples.csv`：每个抽样点的源时间、帧号和手数量。
- `landmarks.jsonl`：关键点与左右手分类结果。handedness_score 是左右手类别置信度，不是手部检测或存在置信度。
- `summary.json`：连续未检出区间、至少 2 秒的待复核区间、检测占比、耗时、模型校验值和参数。
- `contact_sheet.jpg`：约每秒一张标注缩略图。

本轮不实际删除视频。未检出区间包含真实未出镜、遮挡、模糊、仅部分手指可见等模型漏检情况；没有人工逐帧标签，不能给出准确率和召回率。真人手模型的验证不能代替 AgiBot 机器人夹爪检测。

服务器上查看：在输出目录运行 `python -m http.server 8000 --bind 127.0.0.1`，本地执行 SSH `-L 8000:127.0.0.1:8000` 转发后打开 `http://127.0.0.1:8000/index.html`。
