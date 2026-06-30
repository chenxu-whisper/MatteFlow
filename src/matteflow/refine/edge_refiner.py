"""边缘细化模块"""

import cv2
import numpy as np

from ..analysis.region_ownership import RegionOwnershipAnalyzer
from ..config import MattingConfig


class EdgeRefiner:
    """边缘细化器（毛发/羽毛/发丝）"""

    def __init__(self, config: MattingConfig):
        self.config = config
        self._region_analyzer = RegionOwnershipAnalyzer()
        self.last_edge_reconstruction = {}

    def refine(self, frames, alphas, context=None):
        """
        细化边缘 alpha

        Args:
            frames: RGB 帧列表
            alphas: Alpha 列表

        Returns:
            细化后的 alpha 列表
        """
        refined = []
        context = context or {}
        diagnostics = []
        for index, (frame, alpha) in enumerate(zip(frames, alphas)):
            ownership = self._ownership_from_context(context, index)
            refined_alpha = self._refine_single(frame, alpha, ownership=ownership)
            refined.append(refined_alpha)
            diagnostics.append(dict(self.last_edge_reconstruction))
        self.last_edge_reconstruction = {
            "frames": len(diagnostics),
            "changed_pixels": int(sum(item.get("changed_pixels", 0) for item in diagnostics)),
            "protected_pixels": int(sum(item.get("protected_pixels", 0) for item in diagnostics)),
            "mean_delta": float(np.mean([item.get("mean_delta", 0.0) for item in diagnostics]))
            if diagnostics
            else 0.0,
            "frames_detail": diagnostics,
        }
        return refined

    def _refine_single(self, frame: np.ndarray, alpha: np.ndarray, ownership=None) -> np.ndarray:
        """单帧边缘细化"""
        if ownership is None:
            ownership = self._region_analyzer.analyze(frame, alpha)

        # 1. 提取边缘区域（trimap: 确定前景/背景/未知）
        trimap = self._generate_trimap(alpha)

        # 2. 仅在未知区域进行细化
        unknown_mask = (trimap == 128)
        edge_mask = (ownership.hair_edge | ownership.uncertain_edge).astype(bool, copy=False)
        protected_mask = (ownership.luminous_prop | ownership.transparent_effect).astype(bool, copy=False)
        reconstruction_mask = unknown_mask & edge_mask & (~protected_mask)

        if not np.any(reconstruction_mask):
            self.last_edge_reconstruction = self._edge_reconstruction_diagnostics(
                original=alpha,
                refined=alpha,
                reconstruction_mask=reconstruction_mask,
                protected_mask=protected_mask,
            )
            return alpha

        # 3. 局部导向滤波细化
        refined = self._local_guided_filter(frame, alpha, reconstruction_mask)
        refined = self._restore_warm_luminous_props(frame, alpha, refined, ownership=ownership)
        refined = np.where(protected_mask, np.maximum(refined, alpha), refined)
        self.last_edge_reconstruction = self._edge_reconstruction_diagnostics(
            original=alpha,
            refined=refined,
            reconstruction_mask=reconstruction_mask,
            protected_mask=protected_mask,
        )

        return refined

    def _generate_trimap(self, alpha: np.ndarray) -> np.ndarray:
        """生成 trimap"""
        feather_strength = float(np.clip(getattr(self.config, "glow_feather_strength", 1.0), 0.0, 2.0))
        fg_mask = alpha >= 0.95
        bg_mask = alpha <= 0.05

        soft_mask = (alpha > 0.02) & (alpha < 0.98)
        expanded_soft = soft_mask.copy()
        if feather_strength > 0.0 and np.any(soft_mask):
            expansion_radius = int(round(feather_strength))
            if expansion_radius > 0:
                soft_kernel_size = expansion_radius * 2 + 1
                soft_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (soft_kernel_size, soft_kernel_size))
                expanded_soft = cv2.dilate(soft_mask.astype(np.uint8), soft_kernel, iterations=1).astype(bool)

        boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        boundary_band = cv2.dilate(fg_mask.astype(np.uint8), boundary_kernel, iterations=1).astype(bool) & cv2.dilate(
            bg_mask.astype(np.uint8), boundary_kernel, iterations=1
        ).astype(bool)
        unknown_mask = expanded_soft | boundary_band

        trimap = np.full(alpha.shape, 128, dtype=np.uint8)
        trimap[fg_mask] = 255
        trimap[bg_mask] = 0
        trimap[unknown_mask] = 128

        return trimap

    def _local_guided_filter(self, guide: np.ndarray, src: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """局部导向滤波"""
        # 简化版：对边缘区域进行小半径高斯平滑
        result = src.copy()

        # 仅在 mask 区域应用平滑
        src_u8 = (src * 255).astype(np.uint8)
        feather_strength = float(np.clip(getattr(self.config, "glow_feather_strength", 1.0), 0.0, 2.0))
        blur_radius = int(round(2 + feather_strength * 2))
        blur_kernel = blur_radius * 2 + 1
        smoothed = cv2.GaussianBlur(src_u8, (blur_kernel, blur_kernel), 0).astype(np.float32) / 255.0

        # 保留非边缘区域的原值
        result = np.where(mask, smoothed, result)

        return np.clip(result, 0, 1)

    def _restore_warm_luminous_props(
        self,
        frame: np.ndarray,
        original: np.ndarray,
        refined: np.ndarray,
        ownership=None,
    ) -> np.ndarray:
        if ownership is None:
            ownership = self._region_analyzer.analyze(frame, original)
        luminous_prop = ownership.luminous_prop & (original >= 0.95)
        if not np.any(luminous_prop):
            return refined
        return np.where(luminous_prop, np.maximum(refined, original), refined).astype(np.float32, copy=False)

    @staticmethod
    def _edge_reconstruction_diagnostics(
        *,
        original: np.ndarray,
        refined: np.ndarray,
        reconstruction_mask: np.ndarray,
        protected_mask: np.ndarray,
    ) -> dict:
        delta = np.abs(refined.astype(np.float32, copy=False) - original.astype(np.float32, copy=False))
        changed = reconstruction_mask & (delta > 1e-6)
        return {
            "changed_pixels": int(np.count_nonzero(changed)),
            "protected_pixels": int(np.count_nonzero(protected_mask)),
            "reconstruction_pixels": int(np.count_nonzero(reconstruction_mask)),
            "mean_delta": float(delta[changed].mean()) if np.any(changed) else 0.0,
        }

    @staticmethod
    def _ownership_from_context(context: dict, index: int):
        ownerships = context.get("region_ownership") if context else None
        if ownerships is None or index >= len(ownerships):
            return None
        return ownerships[index]
