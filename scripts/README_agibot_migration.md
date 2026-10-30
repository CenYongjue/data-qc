# AgiBot LeRobot v2.1 → v3.0

```bash
cd /data_all/cenyj
HF_HOME=/data_all/cenyj/.cache/lerobot_validation \
  tools/miniforge3/envs/lerobot/bin/python -u scripts/migrate_lerobot_v21_to_v3.py \
  --source datasets/agibot/task_327 \
  --output lerobotv3_output/agibot
```

使用已有 `lerobot` conda 环境及官方 `convert_dataset_v21_to_v30` 中的 info、tasks、data、video、episode metadata 迁移函数。无需上传或下载 Hub 数据。

官方顶层入口会改名输入目录，因此这里直接调用其迁移函数，将输出写入独立的 `.inprogress` 目录；完成必要验证后改为最终目录。原始数据不改动，已有输出拒绝覆盖。

唯一兼容性调整：当前 Pandas 3 与 Hugging Face Array2D 扩展类型的拼接不兼容，因此将官方 `concat_data_files` 调用临时替换为 PyArrow 拼接，保持原始 schema 和 float32 多维数组；分片规则和偏移计算仍使用官方实现。没有修改 LeRobot 仓库源码。

保留 30 fps、全部 `observation.states.*` 和 `actions.*` 字段，以及 episode 的 `action_config` 分步骤标注。视频使用官方码流复制拼接，保留 AV1，不重新编码。

验证范围：版本和字段定义、全部标量索引/时间戳、episode 长度与范围、任务表及分步骤标注一致性、视频头中的帧数与帧率，以及官方读取器对首/中/末 episode 各 3 帧的抽查。不全量解码视频，不重复计算原有统计。

报告：`<output>/qc/migration_report.json`。
