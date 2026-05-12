"""MatteFlow CLI 入口"""

import argparse
import sys

from . import MattingPipeline, MattingConfig, QualityMode, BackgroundMode


def main():
    parser = argparse.ArgumentParser(description="MatteFlow - Video Matting Tool")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--mode", choices=["green", "black", "auto"], default="auto", help="背景模式")
    parser.add_argument("--quality", choices=["fast", "standard", "high"], default="standard", help="质量模式")
    parser.add_argument("--ai", action="store_true", help="使用 AI 增强")
    parser.add_argument("--no-ai", action="store_true", help="禁用 AI")
    
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
    
    if args.ai:
        config.use_ai = True
        config.ai_enhance = True
    elif args.no_ai:
        config.use_ai = False
        config.ai_enhance = False
    
    # 处理
    pipeline = MattingPipeline(config)
    
    def on_progress(current, total, stage):
        print(f"\r[{current}/{total}] {stage}", end="", flush=True)
    
    try:
        result = pipeline.process(args.input, args.output, on_progress)
        print(f"\n✅ 完成！{result['frames_processed']}帧，{result['fps']:.1f} fps")
        return 0
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
