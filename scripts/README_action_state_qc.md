# Action–State 一致性质检（第一版）

运行：

```bash
cd /data_all/cenyj
.venvs/agibot_gripper_qc/bin/python scripts/action_state_qc.py \
  --input datasets/action/file-011.parquet \
  --info datasets/action/info.json \
  --output lerobotv3_output/action_state_qc_pilot
```

只读输入，不删除记录。通过 `--config your_config.json` 覆盖脚本 DEFAULT 中任意参数。
例如 `{"max_delay_s": 0.5, "min_window_s": 0.025, "plot_episodes": 8}`。

## 数据与假设

用户已确认 Action 为绝对关节位置、单位弧度。按 info.json 的关节名称映射，四组为左右臂各7维、左右手各22维。State 暂按实测状态处理，但字段名称本身不能证明来源；共享 timestamp 暂作为指令发布与状态采样时钟。当前不验证指令来源、关节物理参数。

W 对角元素为 `1 / max(p99-p1, 0.05 rad)^2`，分位数取本输入文件全部 State。W 不是物理限位。第一版阈值未经正常样本校准；后续应冻结经人工确认的正常样本统计量，不能在每个新批次上自动漂移阈值。

## 顺序与判断

1. 检查维度、映射、有限值与时间单调性。基础非法记录的 Episode 标 FAIL。
2. 在相邻采样区间计算目标变化速度与实测速度。不对原始位置与速度直接互相关；采用同量纲导数，逐关节 Pearson 后取中位数。搜索 ±0.5 秒、至少30帧重叠，不循环移位。非负相关最大值是候选延迟，负延迟只作时序诊断。激励不足、峰值低/不明显、峰值落在搜索上界或负延迟明显更强时，延迟不可信。当前为整回合共享延迟；不同控制器可能需要后续按组估计。
3. 每次目标更新建立窗口 `[本次发布+延迟, 下一次发布+延迟)`，结束不能越过记录。目标相同或小于死区的更新视为持续保持。窗口不足25ms或已经在目标附近标 UNASSESSABLE；不会为了评价旧动作而借用下一条动作的执行时间。延迟无法确认的 Episode 也标 UNASSESSABLE，但仍输出探索性指标。
4. 在窗口两端按时间线性插值 State。设 `e=W^(1/2)(target-q_start)`、`d=W^(1/2)(q_end-q_start)`：Direction 为二者余弦，Progress 为 `dot(e,d)/dot(e,e)`，Response 为 `norm(d)`。进度可以小于0或大于1，不强迫达到目标。低响应、反向、部分正向、微弱进展分别记标签；响应接近0时方向无定义，保留空值。
5. 所有58维与四个独立组分别保留窗口行。Episode 汇总使用四个组的窗口，避免全维向量掩盖某一侧反向。可评价组窗口少于10或比例低于20%为 UNASSESSABLE；其余在可信延迟下，反向比例>10%或无响应比例>50%为 REVIEW。默认不因行为指标判 FAIL，待正常样本校准后可启用 `enable_behavior_fail`。

这些阈值是跑通用的保守初值。33ms 的指令窗口可能很难从惯性与测量噪声中区分单条动作效果。无响应只表示候选问题，不能排除接触、急停等合理原因。部分执行不算异常，未到位本身也不算异常。当前范围仅为动作—状态一致性，没有包含另一套运动抖动/限位检测。

## 输出

- `config_and_assumptions.json`：完整参数、单位、映射、W、未确认假设。
- `windows.parquet`：每动作、每组的时间边界、得分、可评价性、原因与标签；`group=all` 为全维结果，其余四组用于汇总。
- `episodes.csv`：每回合 PASS / REVIEW / FAIL / UNASSESSABLE 和统计。
- `xcorr.csv`：每回合全部延迟候选及相关系数。
- `plots/*.png`、`review.html`：8个代表回合，展示每组最活跃关节的原始 Action/State、延迟校正曲线、相关曲线、异常窗口。红色是候选，不是人工确认错误。
- `summary.json`：数量、含/不含图表导出的墙钟耗时与轨迹时长倍速。按时间戳首尾差计轨迹时长。

网页可在输出目录执行 `python3 -m http.server 9966 --bind 127.0.0.1`，通过 SSH 转发访问 `review.html`。
