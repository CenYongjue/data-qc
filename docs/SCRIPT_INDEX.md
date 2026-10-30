# 脚本索引

为保留原来的同级 import，暂不移动脚本。日常入口只需看 README 和 pipeline.py。部分历史文件仍被当前入口复用，不应按文件名随意删除。

| 文件 | 定位 |
|---|---|
| [action_state_qc.py](../scripts/action_state_qc.py) | 现行入口/辅助工具 |
| [agibot_clean_v1.py](../scripts/agibot_clean_v1.py) | 现行入口/辅助工具 |
| [agibot_gripper_pilot.py](../scripts/agibot_gripper_pilot.py) | 依赖模块（也保留历史CLI） |
| [agibot_quality_report.py](../scripts/agibot_quality_report.py) | 依赖模块（也保留历史CLI） |
| [agibot_quality_with_gripper.py](../scripts/agibot_quality_with_gripper.py) | 依赖模块（也保留历史CLI） |
| [benchmark_agibot_cpu_parallel.py](../scripts/benchmark_agibot_cpu_parallel.py) | 历史试验/导出工具 |
| [check_image_quality.py](../scripts/check_image_quality.py) | 现行入口/辅助工具 |
| [egodex_video_qc.py](../scripts/egodex_video_qc.py) | 依赖模块（也保留历史CLI） |
| [egodex_yolo_hand_qc.py](../scripts/egodex_yolo_hand_qc.py) | 依赖模块（也保留历史CLI） |
| [egodex_yolo_pipeline.py](../scripts/egodex_yolo_pipeline.py) | 依赖模块（也保留历史CLI） |
| [export_agibot_large_trim_comparisons.py](../scripts/export_agibot_large_trim_comparisons.py) | 历史试验/导出工具 |
| [export_agibot_problem_views.py](../scripts/export_agibot_problem_views.py) | 历史试验/导出工具 |
| [export_pass_review_samples.py](../scripts/export_pass_review_samples.py) | 现行入口/辅助工具 |
| [hifi_umi_qc_gpu_decode.py](../scripts/hifi_umi_qc_gpu_decode.py) | GPU性能实验（非默认） |
| [hifi_umi_qc_gpu_pixels.py](../scripts/hifi_umi_qc_gpu_pixels.py) | GPU性能实验（非默认） |
| [hifi_umi_qc_no_block.py](../scripts/hifi_umi_qc_no_block.py) | 历史试验/导出工具 |
| [hifi_umi_qc_persistent_gpu.py](../scripts/hifi_umi_qc_persistent_gpu.py) | GPU性能实验（非默认） |
| [mediapipe_hand_visibility.py](../scripts/mediapipe_hand_visibility.py) | 历史试验/导出工具 |
| [migrate_lerobot_v21_to_v3.py](../scripts/migrate_lerobot_v21_to_v3.py) | 现行入口/辅助工具 |
| [probe_agibot_gpu_decode.py](../scripts/probe_agibot_gpu_decode.py) | GPU性能实验（非默认） |
| [probe_gpu_quality.py](../scripts/probe_gpu_quality.py) | GPU性能实验（非默认） |
| [summarize_video_qc_run.py](../scripts/summarize_video_qc_run.py) | 现行入口/辅助工具 |
| [syndata_to_lerobot_v3.py](../scripts/syndata_to_lerobot_v3.py) | 现行入口/辅助工具 |
| [test_agibot_quality_report.py](../scripts/test_agibot_quality_report.py) | 已有测试 |
| [test_syndata_to_lerobot_v3.py](../scripts/test_syndata_to_lerobot_v3.py) | 已有测试 |
| [validate_lerobot_v3_reader.py](../scripts/validate_lerobot_v3_reader.py) | 现行入口/辅助工具 |
| [video_quality_pipeline_v2.py](../scripts/video_quality_pipeline_v2.py) | 现行入口/辅助工具 |

`scripts/video_qc_v2/`：`lerobot.py`输入适配、`worker.py`回合检查、`operators.py`画质/LK、`review_page.py`分类网页；`persistent_decode.py`与`gpu_quality.py`是实验代码。

详细运行参数见各脚本 `--help`。历史 README 保留原始环境/路径，当前项目运行方式以根 README 为准。
