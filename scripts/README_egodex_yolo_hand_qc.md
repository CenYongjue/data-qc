# EgoDex YOLO人手检测同批对照

使用发布者Bingsu/adetailer的`hand_yolov8n.pt`，单类别`hand`，纯目标检测网络、约301万参数。它是人手专用训练权重，与此前192输入MediaPipe Palm和参考图提示的YOLOE夹爪试验不同。

来源：https://huggingface.co/Bingsu/adetailer

复用`.venvs/agibot_gripper_qc`隔离环境，补装h5py3.16.0仅写入该venv。未修改系统/其他用户或基础conda环境。模型存入models/hand_presence。

```bash
cd /data_all/cenyj
PYTHONDONTWRITEBYTECODE=1 .venvs/agibot_gripper_qc/bin/python scripts/egodex_yolo_hand_qc.py \
  --baseline lerobotv3_output/egodex_qc_2hz_pilot \
  --output lerobotv3_output/egodex_yolo_hand_2hz_pilot \
  --workers 4 --imgsz 640 --confidence .25
```

输出目录必须不存在。读取之前保存的selection，严格同20条样本、876个2Hz+尾帧点、同3秒持续阈值，保留全视频解码、HDF5完整读取和图像指标，不做运动检查、裁剪或删除源数据。YOLO置信度为0.25，旧Palm为0.5，两种分数不直接等价；权重、输入和阈值均变化，不把改善归因为单个因素。

每个CPU工作进程只加载和热身一次。四进程就绪后才启动质检，日志保留每个模型加载/热身起止时间。排除加载后的速度采用这些区间的并集作为墙钟时间扣除，不将四进程累计时间直接扣除。`worker_startup_imports_models_seconds`还包含子进程启动和导入，不等于纯模型加载。

之前Palm的3条候选即使YOLO判PASS，也导出相同时刻对照片段与速览，并明确标记“旧Palm候选对照”，不伪造为YOLO告警。正常其余样本只保存CSV/JSON。

结果入口index.html，逐样本比较comparison.json，计时performance.json/md。是否有手检测点增加不等于准确率提升，真正无手误检仍需人工真值验证。

## 之前Palm排除加载时间的估算

原完整耗时55.8859秒，425.7333秒视频，7.6179倍。模型初始化/热身四进程累计1.6731秒，原日志没有加载起止时间，无法精确得到墙钟占用。简单按四进程均摊估计加载占0.4183秒，对应7.6754倍（约7.7倍）。即使按不适用的最乐观“累计时间全扣”也只有7.8530倍；不能把这个值称为实测去加载倍速。此次YOLO计时已补齐并行墙钟口径。

本轮已完成，不追加模型或测试：同20条425.73秒，CPU完整68.42秒/6.22倍，排除加载热身墙钟3.34秒后65.09秒/6.54倍。旧3个持续缺手候选均不再触发，但编绳/杯盖的代表帧仍漏检，杯盖检出点减少，不能称为全面改善。未做人工逐帧真值评估。

## A100 FP16基线（2026-10-28）

已新增`--device 0 --half`。这里CLI的half表示FP16，实际通过当前Ultralytics的quantize=16接口传入，避免旧half参数反复弃用警告。四个CPU工作进程及每视频解码2线程完全不变；本初版每个工作进程持有一个GPU模型副本，同用A100 0。尚未集中GPU批推理、安装TensorRT、做INT8或优化CPU解码。

同批20条425.73秒：总46.55秒/9.15倍；模型加载及热身的并行墙钟4.52秒，排除后42.03秒/10.13倍。完整性/质检/手部阶段25.50秒，复查视频及进程退出11.78秒；20PASS，未新增连续缺手告警。FP16后端已核实为cuda:0 / torch.float16；Ultralytics会deepcopy推理模型，不能读取外层model的dtype作为推理精度。

入口：lerobotv3_output/egodex_yolo_hand_a100_fp16/index.html。CPU解码并行提升仅估算在acceleration_notes.json，未实测修改进程数。后续优先保持640输入，集中批推理、异步CPU/GPU流水、TensorRT FP16；INT8使用EgoDex代表图像校准，不默认使用COCO校准。
