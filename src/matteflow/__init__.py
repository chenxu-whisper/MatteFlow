"""
MatteFlow - High-quality video/sequence matting engine

支持：
- 绿幕主体抠图（人物、动物、毛发、发丝）
- 黑底特效抠图（粒子、烟雾、火焰、辉光）
- 背景模式自动识别
- 边缘细化与时序稳定
"""

__version__ = "0.1.0"
__author__ = "ByteDance"

from .pipeline import MattingPipeline
from .config import MattingConfig, QualityMode, BackgroundMode

__all__ = [
    "MattingPipeline",
    "MattingConfig",
    "QualityMode",
    "BackgroundMode",
]
