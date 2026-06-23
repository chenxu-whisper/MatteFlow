"""Effect prop integrity repair for green-screen composites."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import BackgroundMode, MattingConfig
from ..matte.green_screen_matte import GreenScreenMatte


class EffectPropRepair:
    """Rebuild small luminous prop alpha from structural color and key evidence."""

    def __init__(self, config: MattingConfig):
        self.config = config
        self._green_matte = GreenScreenMatte(config)

    def process(self, frames, alphas, bg_mode: BackgroundMode, active_model: str | None = None):
        if bg_mode != BackgroundMode.GREEN_SCREEN:
            return alphas
        if active_model not in {"gvm", "traditional_green_fallback", None}:
            return alphas

        repaired = []
        for frame, alpha in zip(frames, alphas):
            base_alpha = self._green_matte.generate(frame)
            repaired.append(self._repair_single(frame, alpha, base_alpha))
        return repaired

    def _repair_single(self, frame: np.ndarray, alpha: np.ndarray, base_alpha: np.ndarray) -> np.ndarray:
        alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
        base_f = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        prop_mask = self._luminous_prop_mask(frame, base_f)
        if not np.any(prop_mask):
            return alpha_f

        repaired = alpha_f.copy()
        restore_alpha = np.maximum(base_f, 0.985)
        repaired[prop_mask] = np.maximum(repaired[prop_mask], restore_alpha[prop_mask])
        return np.clip(repaired, 0.0, 1.0).astype(np.float32, copy=False)

    def _luminous_prop_mask(self, frame: np.ndarray, base_alpha: np.ndarray) -> np.ndarray:
        frame_f = frame.astype(np.float32, copy=False)
        red = frame_f[:, :, 0]
        green = frame_f[:, :, 1]
        blue = frame_f[:, :, 2]
        brightness = (red + green + blue) / 3.0
        chroma = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
        screen_green = (green > red + 30.0) & (green > blue + 20.0) & (green > 90.0)
        non_screen = ~screen_green

        warm_shell = (
            (red > 170.0)
            & (green > 115.0)
            & (blue < 175.0)
            & ((red - blue) > 45.0)
            & ((green - blue) > 8.0)
            & non_screen
        )
        if not np.any(warm_shell):
            return np.zeros(base_alpha.shape, dtype=bool)

        topology_shell = self._fill_internal_holes(warm_shell, base_alpha >= 0.55)
        prop_reach = cv2.dilate(
            topology_shell.astype(np.uint8, copy=False),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        ).astype(bool)

        bright_core = (
            (brightness > 165.0)
            & (red > 190.0)
            & (green > 145.0)
            & (blue > 70.0)
            & non_screen
        )
        white_yellow_core = (
            (brightness > 185.0)
            & (red > 175.0)
            & (green > 150.0)
            & (blue > 95.0)
            & (chroma < 120.0)
            & non_screen
        )
        warm_halo = (
            (red > 145.0)
            & (green > 105.0)
            & (blue < 190.0)
            & ((red - blue) > 20.0)
            & non_screen
        )

        shell_distance = cv2.distanceTransform((~topology_shell).astype(np.uint8), cv2.DIST_L2, 5)
        line_connected_core = (bright_core | white_yellow_core) & (shell_distance <= 45.0)
        structure_support = (base_alpha >= 0.18) | bright_core | white_yellow_core | warm_halo
        object_reach = prop_reach | line_connected_core
        candidates = object_reach & structure_support & (
            topology_shell | warm_shell | bright_core | white_yellow_core | warm_halo
        )
        candidates = cv2.morphologyEx(
            candidates.astype(np.uint8, copy=False),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
            iterations=1,
        ).astype(bool)
        candidates = self._fill_internal_holes(candidates, object_reach & structure_support)
        component_count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
        if component_count <= 1:
            return np.zeros(base_alpha.shape, dtype=bool)

        component_sizes = np.bincount(labels.ravel())
        warm_support = np.bincount(labels.ravel(), weights=warm_shell.ravel().astype(np.float32))
        keep_labels = np.where((component_sizes >= 8) & (warm_support >= 1.0))[0]
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size == 0:
            return np.zeros(base_alpha.shape, dtype=bool)
        return np.isin(labels, keep_labels)

    def _fill_internal_holes(self, mask: np.ndarray, allowed_region: np.ndarray) -> np.ndarray:
        mask_uint = mask.astype(np.uint8, copy=False)
        contours, _ = cv2.findContours(mask_uint, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        filled_uint = mask_uint.copy()
        for contour in contours:
            if cv2.contourArea(contour) < 12.0:
                continue
            contour_fill = np.zeros_like(mask_uint)
            cv2.drawContours(contour_fill, [contour], -1, 1, thickness=cv2.FILLED)
            filled_uint = np.maximum(filled_uint, contour_fill)
        return (filled_uint.astype(bool) & allowed_region) | mask
