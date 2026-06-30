"""Unified enhancement for black-background mattes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import MattingConfig

_SMOKE_MIN_BRIGHTNESS = 14.0
_SMOKE_MAX_BRIGHTNESS = 64.0
_SMOKE_MAX_CHROMA = 14.0
_SMOKE_MAX_SATURATION = 0.28
_SMOKE_ALPHA_FLOOR = 0.08
_SMOKE_ALPHA_SCALE = 0.30

_COLORED_GLOW_MIN_BRIGHTNESS = 48.0
_COLORED_GLOW_MIN_CHROMA = 28.0
_COLORED_GLOW_MIN_SATURATION = 0.18
_WHITE_GLOW_MIN_BRIGHTNESS = 112.0
_WHITE_GLOW_MAX_CHROMA = 34.0
_GLOW_ALPHA_FLOOR = 0.30
_GLOW_ALPHA_SCALE = 0.78

_PARTICLE_MIN_BRIGHTNESS = 10.0
_PARTICLE_MIN_LOCAL_CONTRAST = 6.0
_PARTICLE_ALPHA_FLOOR = 0.18
_PARTICLE_ALPHA_SCALE = 0.52

_SUBJECT_SOLID_ALPHA = 0.75
_SUBJECT_EDGE_MIN_ALPHA = 0.02
_SUBJECT_EDGE_REPAIR_SCALE = 0.72


@dataclass(frozen=True)
class BlackEffectEnhancementResult:
    """Enhanced alpha, transparent-effect layer, and diagnostics."""

    alpha: np.ndarray
    effect_alpha: np.ndarray
    diagnostics: dict[str, Any]


class BlackEffectEnhancer:
    """Preserve black-background effects while repairing subject edges.

    The final thresholds are intentionally conservative after histogram checks on
    black-background samples: weak smoke/glow is retained, but flat black haze
    and low-intensity colored reflections are not lifted.
    """

    def __init__(self, config: MattingConfig):
        self.config = config

    def enhance(self, frame: np.ndarray, base_alpha: np.ndarray) -> BlackEffectEnhancementResult:
        frame_f = frame.astype(np.float32, copy=False)
        alpha = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        brightness = frame_f.mean(axis=2)
        max_channel = frame_f.max(axis=2)
        min_channel = frame_f.min(axis=2)
        chroma = max_channel - min_channel
        saturation = np.where(max_channel > 0.0, chroma / (max_channel + 1e-6), 0.0)

        pure_black = brightness <= max(2.0, self.config.black_threshold * 255.0 * 0.25)
        smoke_mask = self._smoke_mask(brightness, chroma, saturation, pure_black)
        glow_mask = self._glow_mask(brightness, chroma, saturation, pure_black)
        particle_mask = self._particle_mask(brightness, pure_black)
        subject_edge_mask = self._subject_edge_mask(alpha)

        smoke_alpha = np.clip((brightness - 4.0) / 48.0, 0.0, 1.0) * _SMOKE_ALPHA_SCALE
        smoke_alpha = np.maximum(smoke_alpha, _SMOKE_ALPHA_FLOOR) * smoke_mask
        glow_strength = np.maximum(
            self._smoothstep(brightness / 255.0, 0.08, 0.55),
            self._smoothstep(saturation, 0.08, 0.45),
        )
        glow_alpha = np.maximum(_GLOW_ALPHA_FLOOR, glow_strength * _GLOW_ALPHA_SCALE) * glow_mask
        particle_alpha = np.maximum(
            _PARTICLE_ALPHA_FLOOR,
            self._smoothstep(brightness / 255.0, 0.02, 0.20) * _PARTICLE_ALPHA_SCALE,
        )
        particle_alpha = particle_alpha * particle_mask

        subject_edge_alpha = self._subject_edge_repair(alpha, subject_edge_mask)
        effect_alpha = np.maximum.reduce([smoke_alpha, glow_alpha, particle_alpha])
        enhanced = np.maximum.reduce([alpha, effect_alpha, subject_edge_alpha])
        enhanced = np.where(pure_black & (alpha <= 0.01), 0.0, enhanced)
        enhanced = np.clip(enhanced, 0.0, 1.0).astype(np.float32, copy=False)
        effect_alpha = np.where(pure_black, 0.0, effect_alpha).astype(np.float32, copy=False)

        delta = np.abs(enhanced - alpha)
        diagnostics = {
            "smoke_pixels": int(np.count_nonzero(smoke_mask)),
            "glow_pixels": int(np.count_nonzero(glow_mask)),
            "particle_pixels": int(np.count_nonzero(particle_mask)),
            "subject_edge_pixels": int(np.count_nonzero(subject_edge_mask)),
            "black_residue_suppressed_pixels": int(np.count_nonzero(pure_black)),
            "mean_alpha_delta": float(delta[delta > 1e-6].mean()) if np.any(delta > 1e-6) else 0.0,
        }
        return BlackEffectEnhancementResult(
            alpha=enhanced,
            effect_alpha=effect_alpha,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _smoke_mask(
        brightness: np.ndarray,
        chroma: np.ndarray,
        saturation: np.ndarray,
        pure_black: np.ndarray,
    ) -> np.ndarray:
        return (
            (~pure_black)
            & (brightness >= _SMOKE_MIN_BRIGHTNESS)
            & (brightness <= _SMOKE_MAX_BRIGHTNESS)
            & (chroma <= _SMOKE_MAX_CHROMA)
            & (saturation <= _SMOKE_MAX_SATURATION)
        )

    @staticmethod
    def _glow_mask(
        brightness: np.ndarray,
        chroma: np.ndarray,
        saturation: np.ndarray,
        pure_black: np.ndarray,
    ) -> np.ndarray:
        colored_glow = (
            (brightness >= _COLORED_GLOW_MIN_BRIGHTNESS)
            & (chroma >= _COLORED_GLOW_MIN_CHROMA)
            & (saturation >= _COLORED_GLOW_MIN_SATURATION)
        )
        white_glow = (brightness >= _WHITE_GLOW_MIN_BRIGHTNESS) & (
            chroma <= _WHITE_GLOW_MAX_CHROMA
        )
        return (~pure_black) & (colored_glow | white_glow)

    @staticmethod
    def _particle_mask(brightness: np.ndarray, pure_black: np.ndarray) -> np.ndarray:
        brightness_u8 = np.clip(brightness, 0.0, 255.0).astype(np.uint8)
        local_max = cv2.dilate(brightness_u8, np.ones((3, 3), np.uint8), iterations=1)
        local_mean = cv2.blur(brightness.astype(np.float32, copy=False), (3, 3))
        return (
            (~pure_black)
            & (brightness >= _PARTICLE_MIN_BRIGHTNESS)
            & (brightness_u8 == local_max)
            & ((brightness - local_mean) >= _PARTICLE_MIN_LOCAL_CONTRAST)
        )

    @staticmethod
    def _subject_edge_mask(alpha: np.ndarray) -> np.ndarray:
        solid = alpha >= _SUBJECT_SOLID_ALPHA
        if not np.any(solid):
            return np.zeros_like(alpha, dtype=bool)
        kernel = np.ones((3, 3), np.uint8)
        reach = cv2.dilate(solid.astype(np.uint8), kernel, iterations=1).astype(bool)
        return reach & (alpha > _SUBJECT_EDGE_MIN_ALPHA) & (alpha < _SUBJECT_SOLID_ALPHA)

    @staticmethod
    def _subject_edge_repair(alpha: np.ndarray, subject_edge_mask: np.ndarray) -> np.ndarray:
        if not np.any(subject_edge_mask):
            return np.zeros_like(alpha, dtype=np.float32)
        local_support = cv2.GaussianBlur(alpha.astype(np.float32, copy=False), (3, 3), 0)
        repaired = np.maximum(alpha, local_support * _SUBJECT_EDGE_REPAIR_SCALE)
        return np.where(subject_edge_mask, repaired, 0.0).astype(np.float32, copy=False)

    @staticmethod
    def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
        if high <= low:
            return (value >= high).astype(np.float32)
        t = np.clip((value - low) / (high - low), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
