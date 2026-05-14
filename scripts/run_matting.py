#!/usr/bin/env python3
"""
MatteFlow CLI - 命令行抠图工具

用法:
    python scripts/run_matting.py --input assets/video/test_green_1.mp4
    python scripts/run_matting.py --input assets/video/test_black_1.mp4 --output output/black --mode black
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from matteflow import MattingPipeline, MattingConfig, QualityMode, BackgroundMode
from matteflow.utils.output_paths import resolve_project_output_dir

logger = logging.getLogger(__name__)


def _resolve_output_dir(
    input_path: Path,
    output_arg: str | None,
    project_root: Path = project_root,
) -> Path:
    if output_arg:
        return Path(output_arg)
    return resolve_project_output_dir(input_path, project_root=project_root)


def _configure_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def main():
    parser = argparse.ArgumentParser(description="MatteFlow - High-quality video matting")
    parser.add_argument("--input", "-i", required=True, help="输入视频或序列帧目录")
    parser.add_argument("--output", "-o", help="输出目录，默认使用 temp/output/<输入文件名>/")
    parser.add_argument("--mode", choices=["auto", "green", "black"], default="auto",
                       help="背景模式 (default: auto)")
    parser.add_argument("--quality", choices=["fast", "standard", "high"], default="standard",
                       help="质量模式 (default: standard)")
    parser.add_argument("--mask", action="store_true", help="输出黑白遮罩")
    parser.add_argument("--debug", action="store_true", help="输出调试信息")
    
    args = parser.parse_args()
    _configure_logging(args.debug)
    
    # 构建配置
    config = MattingConfig()
    
    if args.mode == "green":
        config.background_mode = BackgroundMode.GREEN_SCREEN
    elif args.mode == "black":
        config.background_mode = BackgroundMode.BLACK_BACKGROUND
    else:
        config.background_mode = BackgroundMode.AUTO
    
    if args.quality == "fast":
        config.quality_mode = QualityMode.FAST
    elif args.quality == "high":
        config.quality_mode = QualityMode.HIGH
    else:
        config.quality_mode = QualityMode.STANDARD
    
    config.output_mask = args.mask
    config.output_debug = args.debug
    
    # 创建 pipeline
    pipeline = MattingPipeline(config)
    
    # 进度回调
    def on_progress(current, total, stage):
        pct = int(current / total * 100)
        print(f"\r[MatteFlow] {stage}: {pct}%", end="", flush=True)
    
    output_dir = _resolve_output_dir(Path(args.input), args.output)

    logger.info(
        "Starting CLI matting: input=%s output=%s mode=%s quality=%s mask=%s debug=%s",
        args.input,
        output_dir,
        config.background_mode.value,
        config.quality_mode.value,
        args.mask,
        args.debug,
    )
    
    # 执行
    try:
        result = pipeline.process(args.input, str(output_dir), on_progress)
        print()
        logger.info(
            "CLI matting completed: frames=%s time=%.1fs fps=%.2f output=%s",
            result["frames_processed"],
            result["elapsed_time"],
            result["fps"],
            result["output_dir"],
        )
    except Exception as e:
        print()
        logger.exception("CLI matting failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
