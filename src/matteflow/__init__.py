"""
MatteFlow - High-quality video, sequence, and image matting engine

支持：
- 绿幕主体抠图（人物、动物、毛发、发丝）
- 黑底特效抠图（粒子、烟雾、火焰、辉光）
- 视频、序列帧和单张图片输入
- 背景模式自动识别
- 边缘细化与时序稳定
"""

__version__ = "1.0.0"
__author__ = "ByteDance"

from .config import BackgroundMode, MattingConfig, QualityMode
from .pipeline import MattingPipeline

__all__ = [
    "MattingPipeline",
    "MattingConfig",
    "QualityMode",
    "BackgroundMode",
]
