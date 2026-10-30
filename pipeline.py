#!/usr/bin/env python3
"""统一入口：保留各阶段原有参数，不隐式转换、删除或串联修改数据。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'video': 'video_quality_pipeline_v2.py',
    'action': 'action_state_qc.py',
    'robot-motion': 'agibot_clean_v1.py',
    'convert-syndata': 'syndata_to_lerobot_v3.py',
    'migrate-v21': 'migrate_lerobot_v21_to_v3.py',
    'summary': 'summarize_video_qc_run.py',
    'pass-samples': 'export_pass_review_samples.py',
    'image': 'check_image_quality.py',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python pipeline.py <command> [original script arguments]')
        for name, script in COMMANDS.items():
            print(f'  {name:18} scripts/{script}')
        print('\nvideo defaults to configs/video_fast.json (block detection disabled).')
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f'Unknown command: {command}; use --help')
    args = sys.argv[2:]
    if command == 'video' and not any(x == '--config' or x.startswith('--config=') for x in args):
        args = ['--config', str(ROOT / 'configs/video_fast.json'), *args]
    raise SystemExit(subprocess.call([sys.executable, str(ROOT / 'scripts' / COMMANDS[command]), *args]))


if __name__ == '__main__':
    main()
