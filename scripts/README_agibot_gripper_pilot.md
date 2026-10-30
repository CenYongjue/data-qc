# AgiBot 夹爪存在检测：YOLOE nano 小样本试验

```bash
cd /data_all/cenyj
.venvs/agibot_gripper_qc/bin/python -u scripts/agibot_gripper_pilot.py
```

脚本拒绝覆盖已完成的 summary.json。复现时复制 frames/ 和 sample_manifest.json 到新目录，通过 `--samples 新目录` 指定；首次帧提取清单已经保存在输出目录。

环境 `.venvs/agibot_gripper_qc` 只读复用本用户 lerobot conda 环境中的 PyTorch，其余包安装到虚拟环境。基础 conda 环境未安装 Ultralytics；系统及其他人的环境不修改。CPU 运行，Torch 2 线程、OpenCV 1 线程。

## 模型及范围

- Ultralytics YOLOE-26n，使用视觉参考框，无需下载文本编码器，不做真人手关键点/位姿预测。
- 从分割权重构造 detection-only 版本，移除分割分支；参考图编码阶段仍使用原模型。
- 公开镜像：`https://huggingface.co/openvision/yoloe26-n-seg/resolve/main/model.pt`。
- 本地权重：`models/agibot_gripper/yoloe-26n-seg.hf.pt`。
- SHA256：`f432978449803654a11b386754a7edc96187dca39ea622925b781d3d36a975b8`，已核对 Hugging Face LFS 校验值。
- 官方 GitHub 下载中断的文件未用于推理。
- 检测输入 320，主判定置信度 0.25；保存低至 0.01 的原始候选用于分析，没有通过降低阈值迎合测试结果。

## 测试设计与结果

输出目录：`lerobotv3_output/agibot_gripper_pilot/`。

- 参考图取自 episode 0 的 2 秒位置，三路分别标框；腕部相机将可见夹指作为同一类外观示例。
- 测试取 episode 16、31，各自 2/15/30 秒，共 18 张独立图像，不使用参考 episode 作为测试。
- 另取 3 张中心背景裁剪图作为合成负对照；它们不是自然完整负样本。

| 相机 | 实际测试图像的检出数 |
|---|---:|
| head | 0/6 |
| hand_left | 5/6 |
| hand_right | 5/6 |

3 张合成负对照均未触发 0.25 阈值。人工抽看测试框主要落在腕部视角的夹指上；头部视角的夹爪明显可见，但没有检出。因此当前模型/参考组合**不能用于三相机自动剔除**。这些数字只是检出次数，不是严谨准确率或召回率。

缓存图像上的 CPU 预测（含预后处理、不含文件读取和视频解码）约 26–30 ms/张；这不是端到端清洗吞吐。1 Hz 抽样时单路推理有较大加速空间，但三路串行与解码会增加时间，不能据此宣布达到 15 倍轨迹实时。

## 结果文件

- `index.html` / `contact_sheet.jpg`：框图复查。
- `summary.json`：相机结果、参考编码及预测耗时、模型版本及校验值。
- `samples.csv` / `raw_detections.json`：逐图结果及全部低阈值候选。
- `reference_boxes/`：视觉提示参考框。
- `negative_controls.jpg` / `.json`：背景对照及构造方式。

## 兼容性处理

镜像权重的 Conv/BN 已融合。先将目标检测结构同样融合再载入权重，检查参数覆盖率，低于 99% 时中止；本次 257/257 张量完整迁移。禁止把随机初始化导致的零检出当作模型泛化结果。

下一步优先改善头部视角参考或用少量标注微调专用 nano 夹爪检测器，并增加真实负样本。未修改 AgiBot 数据和批量清洗规则，没有生成完整标注视频。

官方接口参考：https://docs.ultralytics.com/models/yoloe/
