"""时序稳定模块"""

from typing import List

import cv2
import numpy as np

from ..config import MattingConfig, QualityMode


class TemporalStabilizer:
    """时序稳定器"""

    def __init__(self, config: MattingConfig):
        self.config = config

    def stabilize(self, alphas: List[np.ndarray], frames: List[np.ndarray] | None = None) -> List[np.ndarray]:
        """
        时序稳定 Alpha

        Args:
            alphas: Alpha 帧列表

        Returns:
            稳定后的 Alpha 列表
        """
        if len(alphas) <= 1:
            return alphas

        if self.config.quality_mode == QualityMode.FAST:
            return self._ema_smooth(alphas)
        elif self.config.quality_mode == QualityMode.STANDARD:
            return self._adaptive_smooth(alphas)
        else:  # HIGH
            return self._optical_flow_smooth(alphas, frames=frames)

    def _transparency_mask(self, alpha: np.ndarray) -> np.ndarray:
        low = float(getattr(self.config, "transparency_temporal_low", 0.03))
        high = float(getattr(self.config, "transparency_temporal_high", 0.75))
        return (alpha > low) & (alpha < high)

    def _ema_smooth(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """快速 EMA 平滑"""
        strength = self.config.temporal_strength
        alpha = 1.0 - strength * 0.3  # EMA 系数

        smoothed = [alphas[0].copy()]
        for i in range(1, len(alphas)):
            ema = alpha * smoothed[-1] + (1 - alpha) * alphas[i]
            smoothed.append(ema)

        return smoothed

    def _adaptive_smooth(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """标准自适应平滑 - 增强版"""
        strength = self.config.temporal_strength
        transparency_blend = float(getattr(self.config, "transparency_temporal_blend", 0.20))

        # 第一次：前向平滑
        forward = []
        for i, alpha in enumerate(alphas):
            if i == 0:
                forward.append(alpha.copy())
                continue

            prev = forward[-1]

            # 计算差异
            diff = np.abs(alpha - prev)

            # 自适应平滑权重：
            # - 差异大 = 更可能是闪烁，需要更多平滑
            # - 差异小 = 保持稳定，减少平滑
            adaptive_weight = np.clip(diff * 3.0, 0, 1) * strength

            # 核心区（alpha 接近 0 或 1）减少平滑，保持边缘清晰
            core_mask = (alpha < 0.02) | (alpha > 0.98)
            adaptive_weight = np.where(core_mask, adaptive_weight * 0.1, adaptive_weight)

            # 应用平滑
            result = (1 - adaptive_weight) * alpha + adaptive_weight * prev
            transparency_mask = self._transparency_mask(alpha)
            transparency_result = alpha * (1.0 - transparency_blend) + prev * transparency_blend
            result = np.where(transparency_mask, transparency_result, result)
            forward.append(np.clip(result, 0, 1))

        # 第二次：后向平滑（双向）
        backward = [forward[-1].copy()]
        for i in range(len(forward) - 2, -1, -1):
            curr = forward[i]
            next_frame = backward[-1]

            diff = np.abs(curr - next_frame)
            adaptive_weight = np.clip(diff * 3.0, 0, 1) * strength * 0.5

            core_mask = (curr < 0.02) | (curr > 0.98)
            adaptive_weight = np.where(core_mask, adaptive_weight * 0.1, adaptive_weight)

            result = (1 - adaptive_weight) * curr + adaptive_weight * next_frame
            transparency_mask = self._transparency_mask(curr)
            transparency_result = curr * (1.0 - transparency_blend) + next_frame * transparency_blend
            result = np.where(transparency_mask, transparency_result, result)
            backward.append(np.clip(result, 0, 1))

        # 合并双向结果
        backward.reverse()

        final = []
        for f, b in zip(forward, backward):
            # 取平均，但权重偏向更稳定的值
            final.append((f + b) * 0.5)

        return final

    def _optical_flow_smooth(
        self,
        alphas: List[np.ndarray],
        frames: List[np.ndarray] | None = None,
    ) -> List[np.ndarray]:
        """高质量光流辅助平滑。"""
        if frames is None or len(frames) != len(alphas) or len(alphas) <= 1:
            return self._adaptive_smooth(alphas)

        strength = float(np.clip(getattr(self.config, "temporal_strength", 0.5), 0.0, 1.0))
        smoothed = [np.clip(alphas[0].astype(np.float32, copy=True), 0.0, 1.0)]
        for prev_frame, curr_frame, curr_alpha in zip(frames, frames[1:], alphas[1:]):
            prev_alpha = smoothed[-1]
            curr_alpha_f = np.clip(curr_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            warped_prev = self._warp_alpha_to_current_frame(prev_frame, curr_frame, prev_alpha)
            motion_supported = warped_prev > curr_alpha_f
            blended = curr_alpha_f * (1.0 - strength) + warped_prev * strength
            result = np.where(motion_supported, np.maximum(curr_alpha_f, blended), curr_alpha_f)
            smoothed.append(np.clip(result, 0.0, 1.0))
        return smoothed

    def _warp_alpha_to_current_frame(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_alpha: np.ndarray,
    ) -> np.ndarray:
        prev_gray = self._to_gray_float(prev_frame)
        curr_gray = self._to_gray_float(curr_frame)
        global_warped = self._warp_alpha_by_global_translation(prev_gray, curr_gray, prev_alpha)
        if float(global_warped.max()) > 0.05:
            return np.clip(global_warped, 0.0, 1.0)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        h, w = prev_alpha.shape
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x - flow[:, :, 0]
        map_y = grid_y - flow[:, :, 1]
        warped = cv2.remap(
            prev_alpha.astype(np.float32, copy=False),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if float(warped.max()) <= 0.05:
            warped = self._warp_alpha_by_global_translation(prev_gray, curr_gray, prev_alpha)
        return np.clip(warped, 0.0, 1.0)

    @staticmethod
    def _warp_alpha_by_global_translation(
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_alpha: np.ndarray,
    ) -> np.ndarray:
        shift, _ = cv2.phaseCorrelate(prev_gray, curr_gray)
        matrix = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
        h, w = prev_alpha.shape
        return cv2.warpAffine(
            prev_alpha.astype(np.float32, copy=False),
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @staticmethod
    def _to_gray_float(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float32, copy=False) / 255.0
