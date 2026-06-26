"""背景模式识别模块"""

from typing import List

import numpy as np

from ..config import BackgroundMode


class BackgroundAnalyzer:
    """背景类型分析器"""

    def __init__(self, sample_frames: int = 5):
        self.sample_frames = sample_frames

    def analyze(self, frames: List[np.ndarray]) -> BackgroundMode:
        """
        分析背景类型

        Args:
            frames: RGB 帧列表

        Returns:
            BackgroundMode: 识别出的背景模式
        """
        if not frames:
            return BackgroundMode.UNKNOWN

        # 采样帧
        total = len(frames)
        indices = np.linspace(0, total - 1, min(self.sample_frames, total), dtype=int)
        samples = [frames[i] for i in indices]

        scores = {
            "green": 0.0,
            "black": 0.0,
        }

        for frame in samples:
            g_score = self._score_green_screen(frame)
            b_score = self._score_black_background(frame)
            scores["green"] += g_score
            scores["black"] += b_score

        # 平均
        scores["green"] /= len(samples)
        scores["black"] /= len(samples)

        # 判断
        if scores["green"] > 0.6 and scores["green"] > scores["black"]:
            return BackgroundMode.GREEN_SCREEN
        elif scores["black"] > 0.6 and scores["black"] > scores["green"]:
            return BackgroundMode.BLACK_BACKGROUND
        elif max(scores["green"], scores["black"]) > 0.4:
            # 有一定置信度，取较高者
            return BackgroundMode.GREEN_SCREEN if scores["green"] > scores["black"] else BackgroundMode.BLACK_BACKGROUND
        else:
            return BackgroundMode.UNKNOWN

    def _score_green_screen(self, frame: np.ndarray) -> float:
        """评分绿幕可能性 (0-1)"""
        # 转换到 HSV
        hsv = self._rgb_to_hsv(frame)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 绿色范围：H 在 60-180 度（OpenCV 中 0-179 对应 0-360 度）
        green_mask = (
            (h >= 35) & (h <= 85) &  # H 范围：绿到青绿
            (s >= 80) &             # 饱和度不低
            (v >= 30)               # 亮度不低
        )

        green_ratio = np.mean(green_mask)

        # 检查背景一致性（绿幕通常背景均匀）
        # 取边缘区域分析
        h, w = frame.shape[:2]
        edge = np.concatenate([
            frame[0:h//4, :].reshape(-1, 3),   # 顶部
            frame[-h//4:, :].reshape(-1, 3),   # 底部
            frame[:, 0:w//4].reshape(-1, 3),   # 左侧
            frame[:, -w//4:].reshape(-1, 3),   # 右侧
        ])

        edge_std = np.std(edge, axis=0).mean()
        uniformity = 1.0 - min(edge_std / 50.0, 1.0)  # 标准差越小越均匀

        # 综合得分
        score = green_ratio * 0.7 + uniformity * 0.3
        return min(score, 1.0)

    def _score_black_background(self, frame: np.ndarray) -> float:
        """评分黑底可能性 (0-1)"""
        # 转换到灰度
        gray = np.mean(frame, axis=2)

        # 低亮区域占比
        dark_mask = gray < 20
        dark_ratio = np.mean(dark_mask)

        # 检查是否有高亮主体（黑底通常有发光/亮主体）
        bright_mask = gray > 100
        bright_ratio = np.mean(bright_mask)

        # 边缘一致性（黑底边缘应该很黑）
        h, w = frame.shape[:2]
        edge = np.concatenate([
            frame[0:h//4, :].reshape(-1, 3),
            frame[-h//4:, :].reshape(-1, 3),
            frame[:, 0:w//4].reshape(-1, 3),
            frame[:, -w//4:].reshape(-1, 3),
        ])
        edge_brightness = np.mean(edge)

        # 综合得分：暗区域多 + 有亮主体 + 边缘暗
        score = dark_ratio * 0.5 + min(bright_ratio * 2, 0.3) + max(0, (30 - edge_brightness) / 30) * 0.2
        return min(score, 1.0)

    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """RGB -> HSV"""
        import cv2
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        return hsv
