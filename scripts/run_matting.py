#!/usr/bin/env python3
"""
MatteFlow CLI - 命令行抠图工具

用法:
    python scripts/run_matting.py --input asset/video/test_green_1.mp4 --output output/green
    python scripts/run_matting.py --input asset/video/test_back_1.mp4 --output output/black --mode black
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.matteflow import MattingPipeline, MattingConfig, QualityMode, BackgroundMode


def main():
    parser = argparse.ArgumentParser(description="MatteFlow - High-quality video matting")
    parser.add_argument("--input", "-i", required=True, help="输入视频或序列帧目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--mode", choices=["auto", "green", "black"], default="auto",
                       help="背景模式 (default: auto)")
    parser.add_argument("--quality", choices=["fast", "standard", "high"], default="standard",
                       help="质量模式 (default: standard)")
    parser.add_argument("--mask", action="store_true", help="输出黑白遮罩")
    parser.add_argument("--debug", action="store_true", help="输出调试信息")
    
    args = parser.parse_args()
    
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
    
    print(f"[MatteFlow] Starting matting...")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Mode: {config.background_mode.value}")
    print(f"  Quality: {config.quality_mode.value}")
    
    # 执行
    try:
        result = pipeline.process(args.input, args.output, on_progress)
        print("\n" + "=" * 50)
        print(f"Done!")
        print(f"  Frames: {result['frames_processed']}")
        print(f"  Time: {result['elapsed_time']:.1f}s")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"  Output: {result['output_dir']}")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
