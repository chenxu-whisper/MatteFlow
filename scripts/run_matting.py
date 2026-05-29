#!/usr/bin/env python3
"""
MatteFlow CLI - 命令行抠图工具

用法:
    python scripts/run_matting.py --input assets/video/test_green_1.mp4
    python scripts/run_matting.py --input assets/image/rabbit.png
    python scripts/run_matting.py --input assets/frames/rabbit_seq --output output/frames --mode green
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from matteflow.cli_app import _configure_logging, _resolve_output_dir, build_parser, main


__all__ = ["build_parser", "main", "_configure_logging", "_resolve_output_dir"]


if __name__ == "__main__":
    raise SystemExit(main())
