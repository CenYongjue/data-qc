# AgiBot清洗新版（agibot_clean_v1.py）

专用于已转换LeRobot v3的AgiBot机器人数据。没有夹爪存在模型，没有AV1→H.264预转码，没有轨迹数值平滑修复。输入数据保持不变。

## 运行

```bash
cd /data_all/cenyj
.venvs/agibot_gripper_qc/bin/python scripts/agibot_clean_v1.py \
  --output lerobotv3_output/agibot_clean_v1_10 \
  --episodes 5,9,12,13,16,23,25,26,28,30 --workers 4
```

输出目录必须不存在。省略`--episodes`时按固定种子`20261027`随机抽10条（`--count`、`--seed`可调）。本次为展示裁头与裁尾，将随机样本的0替换为9，实际ID以`selection.json`为准。依赖已有隔离环境，原来的基础脚本保持可用。

## 质检

复用`agibot_quality_with_gripper.py`的数值有效性、时间戳、关节统计越界、末端突变、静止、图像质量、冻结及去重算子；视频质量三路各1Hz。数值/动作检查使用全帧数值，不受视频抽样频率限制。

- 统计范围严格为`[p1-alpha*(p99-p1), p99+alpha*(p99-p1)]`；系数0.10，支持标量或按字段及关节配置。归一化epsilon不会混入统计余量。
- 方法A：反射补边的频域低通，Butterworth形状响应，默认3Hz、4阶；高频残差0.5秒窗口RMS和能量占比同时超限，连续至少0.3秒。监测状态/动作的关节与末端位置。
- 方法A每个维度RMS阈值为全32条校准的`median + 8*1.4826*MAD`，并设最低归一化RMS阈值0.005；高频能量占比至少0.2。边缘1秒不参与判定。参数均在`agibot_clean_v1_config.json`，只是初始参数，尚未用人工真值标定。
- 方法B：原7帧移动平均残差及其一、二阶导数，三者同时超稳健阈值且持续达标。保留原末端位置监测逻辑。
- A或B任一达到剔除条件则FAIL整条剔除。全静止无有效运动也标FAIL。滤波结果不替换原状态/动作。
- PASS导出训练副本；REVIEW（例如统计关节越界、近重复）暂存报告等待复核，不当作已证明损坏，但不混入本次训练输出。
- 未识别物理限位、真实跨模态时钟偏差、任务成功或正在运动的复位阶段；只剪开头/结尾静止，不删除中间静止。

## 实际切分与无转码输出

保留当前首尾速度判定及任务标注保护，默认边界padding0.5秒、最小裁剪0.5秒、最短保留2秒。

`dataset/`是真正的LeRobot v3副本：按保留区间切片全部数值字段，重建episode/frame/global索引、时间戳、数据偏移、视频偏移和统计。原始状态/动作值不修改，步骤标注裁剪并平移；现有action_config结束边界尚未确认，保守按inclusive保护并处理。

视频逐压缩包复制原AV1，实际输出保留区间所在的视频片段。若起点不是关键帧，保留解码必需的关键帧前导，并用`videos/.../from_timestamp`表示精确起点，不把前导帧加入数值记录。当前源的PTS/DTS无重排，代码遇到重排会拒绝导出，避免不完整依赖。源帧与输出帧像素抽查相同。保留帧内数值/视频同步，不把整个原视频简单附加到新数据假装切分。

每个保留episode独立视频/Parquet文件，符合v3索引约定；默认样本量小，尚未实现批量文件容量滚动。图像统计基于保留区间的原视频1Hz抽样缩略图，数值统计为保留全帧的min/max/mean/std；没有伪造全帧图像统计或分位数。

## 复查与计时

- `report/index.html`：每条质量原因、速览、原速问题片段。
- `report/trim_comparisons.html`：最多3条已裁剪PASS样本的前后视频。BEFORE含删去的静止部分，AFTER仅保留边界内的片段，均原速显示源帧号。两段起点/时长不同，不声称逐时刻同步并排。
- 这些复查MP4延续原流程编码方式；“不转码”指原数据和训练输出保留AV1，不影响复查材料生成。
- `episode_mapping.json`：新旧episode、源帧区间及每路视频前导偏移。
- `validation.json`：结构及首末输出轨迹的边界帧像素核对。
- `performance.json/md`：计入数值校准、两套抖动、视频检查、速览/证据、训练数据实际导出、前后对照和验证的总墙钟时间。主计时不含解释器启动及顶层导入。

校准使用全部32条数值轨迹，视频质检仅针对选定10条。四进程统一阈值，跨轨迹去重在所选10条中统一执行。源数据不变，REVIEW/FAIL及原因可追溯。

本次实测：10条12304帧/410.13秒，四进程完整流水线20.93秒，轨迹口径19.60倍；PASS8、REVIEW2、FAIL0。导出8条9448帧，6条裁剪，共删首尾静止帧225帧/7.50秒。前后对照：9裁头0.50秒，12裁尾0.80秒，23裁尾1.73秒。结构、原帧像素及官方LeRobot读取均通过；附加官方读取检查单独记录，不包含在20.93秒中。

## 全部32条结果与重点视图

全部处理命令：

```bash
.venvs/agibot_gripper_qc/bin/python scripts/agibot_clean_v1.py \
  --output lerobotv3_output/agibot_clean_v1_all --count 32 --workers 4
.venvs/agibot_gripper_qc/bin/python scripts/export_agibot_problem_views.py \
  --result lerobotv3_output/agibot_clean_v1_all --count 6
```

全量实测44.60秒，1313.83秒轨迹，29.46倍。24PASS导出28848帧，8REVIEW待人工复核，0FAIL；14条实际裁剪。重点视图另7.37秒，六条为3/5/7/4/18/0，输出全帧关节曲线和阈值、抖动A/B评分、静止建议、代表时刻三路截图及已有原速问题片段/速览入口，不重新渲染完整标注视频。统一警告不参与问题排序。两阶段相加51.97秒，25.28倍。官方读取抽查另记。

重点入口：`lerobotv3_output/agibot_clean_v1_all/report/problem_views/index.html`。

按实际裁剪时长选择最大的回合，导出原速并排及独立前后片段：

```bash
.venvs/agibot_gripper_qc/bin/python scripts/export_agibot_large_trim_comparisons.py \
  --result lerobotv3_output/agibot_clean_v1_all --count 4
```

当前选择28/22/29/21，分别裁尾2.067/2.033/1.933/1.867秒。各样本导出before、after、side_by_side三份MP4，网页入口为report/large_trim_comparisons/index.html。并排按源时间对齐，左原始、右实际清洗数据；被删除的右侧时段标黑屏CUT，独立after文件只播放实际保留的内容。12个MP4的帧数、帧率和尺寸验证通过。
