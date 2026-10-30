# EgoDex视频/受损文件/双手出镜样本试验

运行环境复用`tools/miniforge3/envs/vjepa2-312`已有的h5py/av/opencv/numpy，本轮未安装或修改环境。手掌模型存放在项目models/hand_presence目录。

```bash
cd /data_all/cenyj
PYTHONDONTWRITEBYTECODE=1 tools/miniforge3/envs/vjepa2-312/bin/python scripts/egodex_video_qc.py \
  --dataset datasets/egoDex/part1 --output lerobotv3_output/egodex_qc_2hz_pilot \
  --count 20 --workers 4 --sample-hz 2 --missing-seconds 3
```

输出目录必须不存在；改目录可复现。20条为4个定向场景加固定种子跨任务抽样，不是全数据质量分布的无偏估计，不包含全量清洗。

## 当前执行范围

- 视频全程CPU解码，启用err_detect=explode，检查可解码性、坏帧标记、PTS递增/帧间隔、解码帧数与视频头/HDF5一致性。
- HDF5检查必要组/字段、内参形状、逐字段序列长度；读取所有dataset数据块以发现不可读/压缩块损坏；数值非有限值单独记录。不检查运动合理性。
- 质量与手掌检测2Hz，补采最后一帧。复用当前模糊/曝光/低纹理规则，阈值来自AgiBot初始配置，尚未专门为EgoDex人工标定。
- 轻量OpenCV DNN MediaPipe Palm Detector（192x192）；仅使用框和分数，不运行手部21点姿态网络。模型附带的粗关键点输出不使用、不保存。
- 任意一只手掌检出即不满足“双手不可见”；连续未检出任何手的采样证据跨度>=3秒记REVIEW，不自动删除。单手不可见不触发。
- HDF5手部追踪置信度只是辅助列，绝不代替画面检测；没有手部左右分类需求，因为存在任意一只即可。
- 完全相同画面仅记冻结候选REVIEW，不将正常操作停顿判为损坏。
- 只导出报告及最多6个问题样本的速览/原速证据，不生成正常样本整段标注视频或逐帧JPEG。
- 框来自最近2Hz抽样，原速片段字幕明确标注，不声称逐帧检测。速览2Hz抽样、4fps播放（2倍速）。
- 不执行运动质检、轨迹数值修复、首尾裁剪、LeRobot转换或源数据删除。本轮是检测验证。

## 实测与复核

20条视频425.73秒，共12772帧全程解码，876个质量/手掌抽样点。CPU四进程总55.89秒，**7.62倍**；其中完整性/质量/模型阶段44.69秒，复查媒体10.23秒。包含HDF5读取、全视频解码、2Hz检测、问题MP4、速览及报告，不含下载、Python启动和顶层导入。进程内累计细项不能当墙钟阶段直接相加。

17PASS、3REVIEW、0FAIL。本批未发现损坏、帧数/PTS不一致、持续模糊或曝光问题。三个REVIEW均为双手未检出候选：

- braid_unbraid/1035：0.0–3.5秒。
- fry_egg/0：4.0–7.0秒及32.5–36.0秒。
- add_remove_lid/1748：0.0–3.0秒。

**三个候选代表截图中均能看到手，存在明确模型漏检，不能据此自动剔除。**只做了代表截图复核，未对完整区间逐帧标注，不能计算准确率/召回率。手背、握持物体、遮挡和局部手指是需要后续验证的场景。结果网页已醒目标注，人工记录为manual_spot_review.json。

入口：`lerobotv3_output/egodex_qc_2hz_pilot/index.html`。6个输出MP4均可读取；临时故意损坏的MP4/HDF5验证能记录FAIL且不改源文件。完整性检查与画面是否正确是不同概念，不用一次抽样通过声称所有视频内容正确。

## 模型来源

- OpenCV官方模型与算法：https://github.com/opencv/opencv_zoo/tree/main/models/palm_detection_mediapipe
- OpenCV官方HF权重镜像：https://huggingface.co/opencv/palm_detection_mediapipe/resolve/main/palm_detection_mediapipe_2023feb.onnx
- SHA256：78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c

实现按官方预处理与anchor布局解码框，NMS使用xywh坐标。模型/阈值当前不满足可靠的自动双手缺失剔除需求，后续优先验证更适合第一视角的人手权重。
