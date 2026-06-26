"""Rembg AI 抠图模块 - 稳定可用的开源抠图"""

from typing import List

import numpy as np
from PIL import Image

from ..config import MattingConfig


class RembgMatte:
    """基于 rembg 的 AI 抠图引擎"""

    def __init__(self, config: MattingConfig):
        self.config = config
        self._available = self._check_available()

    def _check_available(self):
        """检查 rembg 是否可用"""
        try:
            from rembg import remove as rembg_remove
        except ImportError:
            return False
        return callable(rembg_remove)

    def generate(self, frame: np.ndarray) -> np.ndarray:
        """单帧抠图"""
        if not self._available:
            from .green_screen_matte import GreenScreenMatte
            return GreenScreenMatte(self.config).generate(frame)

        from rembg import remove

        # 转换为 PIL Image
        if frame.dtype == np.uint8:
            frame_rgb = frame
        else:
            frame_rgb = (frame * 255).astype(np.uint8)

        pil_image = Image.fromarray(frame_rgb)

        # 使用 rembg 抠图
        result = remove(pil_image)

        # 提取 alpha 通道
        result_array = np.array(result)
        if result_array.shape[2] == 4:
            alpha = result_array[:, :, 3].astype(np.float32) / 255.0
        else:
            alpha = np.ones(result_array.shape[:2], dtype=np.float32)

        return alpha

    def generate_sequence(self, frames: List[np.ndarray], progress_callback=None) -> List[np.ndarray]:
        """序列抠图 — 注意：后处理由 hybrid_matte 统一处理"""
        alphas = []
        total = len(frames)

        for i, frame in enumerate(frames):
            alpha = self.generate(frame)
            alphas.append(alpha)

            if progress_callback and i % max(1, total // 20) == 0:
                progress_callback(i, total)

        return alphas
