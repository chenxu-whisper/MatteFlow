"""Foreground color reconstruction for green-screen spill."""

from __future__ import annotations

import numpy as np

from ..analysis.region_ownership import RegionOwnershipAnalyzer
from ..config import MattingConfig


class ForegroundColorRecovery:
    """Recover foreground RGB by locally unmixing screen color from owned regions."""

    def __init__(self, config: MattingConfig):
        self.config = config
        self._region_analyzer = RegionOwnershipAnalyzer()

    def recover(self, frame: np.ndarray, alpha: np.ndarray, ownership=None) -> np.ndarray:
        """Return RGB with green-screen contamination removed from owned soft regions."""
        frame_f = frame.astype(np.float32, copy=False)
        alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
        if ownership is None:
            ownership = self._region_analyzer.analyze(frame, alpha_f)

        red = frame_f[:, :, 0]
        green = frame_f[:, :, 1]
        blue = frame_f[:, :, 2]
        green_bias = green - np.maximum(red, blue)
        owned = ownership.transparent_effect
        uncertain_screen_mix = (
            ownership.uncertain_edge
            & (~ownership.luminous_prop)
            & (~ownership.hair_edge)
            & (green_bias > 22.0)
            & (alpha_f < 0.72)
        )
        repair_mask = (
            (owned | uncertain_screen_mix)
            & (~ownership.background_residue)
            & (alpha_f > 0.08)
            & (alpha_f < 0.86)
            & (green_bias > 4.0)
        )
        if not np.any(repair_mask):
            return np.asarray(frame, dtype=np.uint8).copy()

        screen_rgb = self._estimate_screen_color(frame_f, alpha_f)
        recovered = self._unmix_foreground(frame_f, alpha_f, screen_rgb)
        weight = self._recovery_weight(frame_f, alpha_f, repair_mask)
        if not np.any(weight > 0.0):
            return np.asarray(frame, dtype=np.uint8).copy()

        result = frame_f * (1.0 - weight[..., np.newaxis]) + recovered * weight[..., np.newaxis]
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)

    def _estimate_screen_color(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self.config.key_color is not None:
            return np.array(self.config.key_color, dtype=np.float32)

        h, w = alpha.shape
        border = max(min(min(h, w) // 6, 96), 1)
        border_mask = np.zeros_like(alpha, dtype=bool)
        border_mask[:border, :] = True
        border_mask[-border:, :] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True

        red = frame[:, :, 0]
        green = frame[:, :, 1]
        blue = frame[:, :, 2]
        screen_like = (green > red + 12.0) & (green > blue + 12.0)
        candidates = border_mask & (alpha <= 0.04) & screen_like
        if int(candidates.sum()) < 16:
            candidates = (alpha <= 0.06) & screen_like
        if int(candidates.sum()) < 16:
            candidates = border_mask & screen_like
        if int(candidates.sum()) < 16:
            return np.array([0.0, 255.0, 0.0], dtype=np.float32)
        return np.median(frame[candidates], axis=0).astype(np.float32)

    @staticmethod
    def _unmix_foreground(frame: np.ndarray, alpha: np.ndarray, screen_rgb: np.ndarray) -> np.ndarray:
        safe_alpha = np.maximum(alpha, 0.16)
        screen = screen_rgb.reshape(1, 1, 3).astype(np.float32, copy=False)
        recovered = (frame - (1.0 - alpha)[..., np.newaxis] * screen) / safe_alpha[..., np.newaxis]
        return np.clip(recovered, 0.0, 255.0)

    @staticmethod
    def _recovery_weight(frame: np.ndarray, alpha: np.ndarray, repair_mask: np.ndarray) -> np.ndarray:
        red = frame[:, :, 0]
        green = frame[:, :, 1]
        blue = frame[:, :, 2]
        green_bias = np.maximum(0.0, green - np.maximum(red, blue))
        soft_alpha_weight = np.clip((0.98 - alpha) / 0.50, 0.0, 1.0)
        contamination_weight = np.clip((green_bias - 4.0) / 32.0, 0.0, 1.0)
        return np.where(repair_mask, np.maximum(contamination_weight, 0.35) * soft_alpha_weight, 0.0).astype(
            np.float32,
            copy=False,
        )
