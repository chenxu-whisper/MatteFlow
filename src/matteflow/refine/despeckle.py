"""去噪点模块 - 去除 alpha 遮罩中的噪点"""

import numpy as np
import cv2

from ..analysis.region_ownership import RegionOwnershipAnalyzer
from ..config import MattingConfig


class Despeckle:
    """去除 alpha 遮罩中的噪点"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self._region_analyzer = RegionOwnershipAnalyzer()
    
    def process(self, alphas, frames=None, context=None):
        """
        去除噪点
        
        Args:
            alphas: Alpha 列表
        
        Returns:
            去噪后的 alpha 列表
        """
        if not self.config.despeckle_enable:
            return alphas
        
        cleaned = []
        context = context or {}
        for index, alpha in enumerate(alphas):
            frame = frames[index] if frames is not None and index < len(frames) else None
            ownership = self._ownership_from_context(context, index)
            cleaned_alpha = self._despeckle_single(
                alpha,
                frame=frame,
                context=context,
                ownership=ownership,
            )
            cleaned.append(cleaned_alpha)
        return cleaned
    
    def _despeckle_single(self, alpha: np.ndarray, frame=None, context=None, ownership=None) -> np.ndarray:
        """单帧去噪点"""
        alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
        alpha_u8 = np.clip(alpha_f * 255.0, 0.0, 255.0).astype(np.uint8)
        
        # 使用中值滤波去除噪点
        radius = self.config.despeckle_radius
        ksize = 2 * radius + 1
        cleaned = cv2.medianBlur(alpha_u8, ksize)
        cleaned = self._restore_supported_soft_alpha(
            alpha_u8,
            cleaned,
            ksize,
            frame=frame,
            context=context or {},
            ownership=ownership,
        )
        cleaned = self._restore_warm_luminous_props(alpha_u8, cleaned, frame=frame, ownership=ownership)
        
        # 阈值处理去除微小噪点
        threshold = int(self.config.despeckle_threshold * 255)
        if threshold > 0:
            cleaned = np.where(cleaned <= threshold, 0, cleaned).astype(np.uint8)
            if ownership is not None:
                protected = ownership.transparent_effect | ownership.luminous_prop
                cleaned = np.where(protected & (alpha_u8 > 0), np.maximum(cleaned, alpha_u8), cleaned).astype(np.uint8)
        
        return cleaned.astype(np.float32) / 255.0

    def _is_gvm_active(self, context: dict) -> bool:
        return context.get("active_ai_model") == "gvm"

    def _swirl_color_mask(self, frame: np.ndarray) -> np.ndarray:
        frame_f = frame.astype(np.float32, copy=False)
        r = frame_f[:, :, 0]
        g = frame_f[:, :, 1]
        b = frame_f[:, :, 2]
        brightness = (r + g + b) / 3.0
        chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        return (
            (b > g + 8.0)
            | ((r > g + 8.0) & (b > g + 4.0))
            | ((brightness > 150.0) & (chroma < 70.0) & (g < 205.0))
        )

    def _restore_supported_soft_alpha(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        ksize: int,
        frame=None,
        context=None,
        ownership=None,
    ) -> np.ndarray:
        """Keep supported soft alpha from being flattened into zero by large median kernels."""
        soft_floor = max(8, int(round(0.03 * 255.0)))
        crushed_soft = (original > soft_floor) & (cleaned <= soft_floor)
        if not np.any(crushed_soft):
            return cleaned

        support_kernel = np.ones((ksize, ksize), dtype=np.float32)
        support_count = cv2.filter2D(
            (original > soft_floor).astype(np.float32),
            cv2.CV_32F,
            support_kernel,
            borderType=cv2.BORDER_CONSTANT,
        )
        min_support_pixels = max(4, (ksize * ksize) // 12)
        support_mask = support_count >= float(min_support_pixels)

        if frame is not None and self._is_gvm_active(context or {}):
            swirl_mask = self._swirl_color_mask(frame)
            swirl_support_count = cv2.filter2D(
                swirl_mask.astype(np.float32),
                cv2.CV_32F,
                support_kernel,
                borderType=cv2.BORDER_CONSTANT,
            )
            swirl_min_support = max(4, min_support_pixels // 2)
            swirl_support_mask = support_count >= float(swirl_min_support)
            swirl_color_support = swirl_support_count >= float(swirl_min_support)
            protected_mask = crushed_soft & swirl_mask & swirl_support_mask & swirl_color_support
            restored = np.where(protected_mask, original, cleaned)
        else:
            restored = np.where(crushed_soft & support_mask, original, cleaned)

        if frame is not None:
            if ownership is None:
                ownership = self._region_analyzer.analyze(frame, original.astype(np.float32) / 255.0)
            transparent_effect = ownership.transparent_effect
            restored = np.where(crushed_soft & transparent_effect, original, restored)
        return restored.astype(np.uint8)

    def _restore_warm_luminous_props(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        frame=None,
        ownership=None,
    ) -> np.ndarray:
        if frame is None:
            return cleaned

        if ownership is None:
            ownership = self._region_analyzer.analyze(frame, original.astype(np.float32) / 255.0)
        luminous_prop = ownership.luminous_prop
        luminous_prop = luminous_prop & (original > cleaned)
        if not np.any(luminous_prop):
            return cleaned
        return np.where(luminous_prop, np.maximum(cleaned, original), cleaned).astype(np.uint8, copy=False)

    @staticmethod
    def _ownership_from_context(context: dict, index: int):
        ownerships = context.get("region_ownership") if context else None
        if ownerships is None or index >= len(ownerships):
            return None
        return ownerships[index]
