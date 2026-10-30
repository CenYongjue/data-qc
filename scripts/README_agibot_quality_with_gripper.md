# AgiBot 质量检查与夹爪检测整合版

脚本 `agibot_quality_with_gripper.py` 独立于原版，源数据保持不变。默认选 episode 0、3、16、31 做效率试验；数值阈值仍用全部32条轨迹校准，视频只检查选定轨迹。近重复检查范围是选定轨迹。

```bash
cd /data_all/cenyj
.venvs/agibot_gripper_qc/bin/python scripts/agibot_quality_with_gripper.py \
  --dataset lerobotv3_output/agibot \
  --output lerobotv3_output/agibot_qc_gripper_benchmark_fast \
  --episodes 0,3,16,31
```

输出目录必须不存在。依赖复用已有隔离环境、模型权重及 `agibot_gripper_pilot` 的三路视觉提示，不需要重新安装或下载。

- 原有数值、时间戳、运动异常、静止首尾建议、图像质量、冻结和去重规则保留。
- 三路相机各1Hz，补采最后一帧；同一次解码结果用于图像质量、夹爪检测和速览。稀疏抽样按关键帧seek，连续读取复用迭代器，问题片段不再逐帧重复seek。
- YOLOE-26n纯检测，320输入，CPU两线程，置信度0.25；连续未检出采样点跨度达到3秒标记REVIEW。头部模型已知明显漏检，不能据此自动剔除，也不能把未检出当作真实未出镜。
- 每条保留三路拼接速览（含检测框，5fps播放）。最多生成两个6秒原速问题片段；这些片段展示原图，不逐帧重新跑检测。若只有裁剪建议则展示裁剪边界。
- 不输出逐帧JPEG、全视频逐帧标注、分割掩膜或位姿。CSV记录抽样检测框、分数和图像指标；每轨迹JSON、HTML保留原因和视频链接。
- 默认仅报告，不删除或裁剪。三路分别判断，一路待复查则整条待复查。

`performance.json` / `performance.md` 给出实测三相机共享轨迹倍速和相机累计倍速。`total_seconds` 包括模型加载、数值校准、解码、检测、视频及报告，不含Python解释器启动和顶层依赖导入。解码/检测/图像指标/编码为阶段内明细，与analysis/evidence时间重叠，不要重复相加。编码明细是管道写入和等待时间，编码进程与主进程可能重叠。小样本性能受缓存、并发负载和异常片段数影响。

本次结果：4条轨迹5068帧（168.93秒），525个三路抽样点，4个速览、7个问题片段，整体54.85秒，轨迹口径3.08倍/三路视频累计9.24倍。头部仅1/175点检出，不能用未检出直接剔除。结果为共享服务器负载下实测，运行条件见报告目录run_conditions.json。

关闭模型进行对照实测（不导入或加载Torch/YOLOE，保留原质检、速览和按实际问题选取的证据片段）：

```bash
.venvs/agibot_gripper_qc/bin/python scripts/agibot_quality_with_gripper.py \
  --dataset lerobotv3_output/agibot \
  --output lerobotv3_output/agibot_qc_no_gripper_benchmark \
  --episodes 0,3,16,31 --no-gripper
```

关闭模型后不会产生夹爪未检出的原因，因此问题片段数量可与启用模型时不同；这是两种实际流程吞吐的比较，不是固定全部媒体输出后的纯模型耗时比较。

无模型版实测：同样168.93秒轨迹耗时16.53秒，10.22倍轨迹实时速度；4个速览、3个问题片段。525个抽样点及图像质量规则结果与模型版一致，输出视频均可打开。

CPU有限并行：增加`--workers 2`或`--workers 4`，当前仅支持`--no-gripper`。按整条轨迹分配独立工作进程，各自持有解码器/编码器；单容器解码2线程、编码2线程，OpenCV1线程。全数据阈值只在主进程校准一次，跨轨迹去重仍统一执行。进程间不共享视频容器。

```bash
.venvs/agibot_gripper_qc/bin/python scripts/agibot_quality_with_gripper.py \
  --dataset lerobotv3_output/agibot --output lerobotv3_output/agibot_qc_parallel_run \
  --episodes 0,3,16,31 --no-gripper --workers 4
```

自动对比1/2/4工作进程（输出目录必须是新的）：

```bash
.venvs/agibot_gripper_qc/bin/python scripts/benchmark_agibot_cpu_parallel.py \
  --output lerobotv3_output/agibot_qc_cpu_comparison_new
```

每组按顺序执行，保留相同速览及按规则生成的问题片段。对比脚本自动核对逐轨迹报告、阈值和逐抽样CSV完全一致。并行时`decode_seconds`等为各工作进程累计用时，不能与总墙钟时间计算占比；吞吐始终用总墙钟时间（含子进程启动退出）计算。

有限并行本次实测：1/2/4进程分别16.45/11.02/6.84秒，轨迹口径10.27/15.33/24.71倍。4进程含解释器启动和导入为7.89秒，即21.40倍。结果、条件和对应网页见`lerobotv3_output/agibot_qc_cpu_parallel/comparison.md`。当前可先用4进程。
