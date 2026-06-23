"""Region ownership masks for matting repair decisions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RegionOwnership:
    """Per-pixel region ownership signals used by downstream repair stages."""

    subject: np.ndarray
    hair_edge: np.ndarray
    luminous_prop: np.ndarray
    transparent_effect: np.ndarray
    background_residue: np.ndarray
    uncertain_edge: np.ndarray


class RegionOwnershipAnalyzer:
    """Classify matte regions from color, alpha, and keyer structural evidence."""

    SUBJECT_COLOR = np.array([80, 220, 120], dtype=np.uint8)
    HAIR_EDGE_COLOR = np.array([255, 170, 0], dtype=np.uint8)
    LUMINOUS_PROP_COLOR = np.array([255, 230, 40], dtype=np.uint8)
    TRANSPARENT_EFFECT_COLOR = np.array([120, 220, 255], dtype=np.uint8)
    BACKGROUND_RESIDUE_COLOR = np.array([255, 60, 60], dtype=np.uint8)
    UNCERTAIN_EDGE_COLOR = np.array([180, 80, 255], dtype=np.uint8)

    def analyze(
        self,
        frame: np.ndarray,
        alpha: np.ndarray,
        base_alpha: np.ndarray | None = None,
    ) -> RegionOwnership:
        alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
        base_f = alpha_f if base_alpha is None else np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
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
        luminous_prop = self._luminous_prop_mask(
            warm_shell,
            bright_core,
            white_yellow_core,
            warm_halo,
            base_f,
        )

        transparent_effect = (
            (alpha_f > 0.08)
            & (alpha_f < 0.82)
            & (base_f > 0.08)
            & (brightness > 135.0)
            & (chroma < 120.0)
            & non_screen
            & (~luminous_prop)
        )
        uncertain_edge = self._uncertain_edge_mask(alpha_f)
        hair_edge = uncertain_edge & (brightness > 80.0) & (chroma < 180.0) & (~luminous_prop)

        document_like = (
            (base_f > 0.35)
            & (alpha_f > 0.35)
            & (brightness > 175.0)
            & (chroma < 45.0)
            & (~luminous_prop)
            & (~transparent_effect)
        )
        background_residue = self._large_component_mask(document_like, min_area=96)

        subject = (alpha_f >= 0.70) & (~background_residue)
        return RegionOwnership(
            subject=subject.astype(bool, copy=False),
            hair_edge=hair_edge.astype(bool, copy=False),
            luminous_prop=luminous_prop.astype(bool, copy=False),
            transparent_effect=transparent_effect.astype(bool, copy=False),
            background_residue=background_residue.astype(bool, copy=False),
            uncertain_edge=uncertain_edge.astype(bool, copy=False),
        )

    def build_debug_overlay(
        self,
        frame: np.ndarray,
        alpha: np.ndarray,
        base_alpha: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an RGB overlay that visualizes region ownership classes."""
        overlay = np.asarray(frame, dtype=np.uint8).copy()
        ownership = self.analyze(frame, alpha, base_alpha)
        overlay[ownership.subject] = self.SUBJECT_COLOR
        overlay[ownership.uncertain_edge] = self.UNCERTAIN_EDGE_COLOR
        overlay[ownership.hair_edge] = self.HAIR_EDGE_COLOR
        overlay[ownership.transparent_effect] = self.TRANSPARENT_EFFECT_COLOR
        overlay[ownership.luminous_prop] = self.LUMINOUS_PROP_COLOR
        overlay[ownership.background_residue] = self.BACKGROUND_RESIDUE_COLOR
        return overlay

    def _luminous_prop_mask(
        self,
        warm_shell: np.ndarray,
        bright_core: np.ndarray,
        white_yellow_core: np.ndarray,
        warm_halo: np.ndarray,
        base_alpha: np.ndarray,
    ) -> np.ndarray:
        if not np.any(warm_shell):
            return np.zeros(base_alpha.shape, dtype=bool)

        topology_shell = self._fill_internal_holes(warm_shell, base_alpha >= 0.55)
        prop_reach = cv2.dilate(
            topology_shell.astype(np.uint8, copy=False),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        ).astype(bool)
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

    @staticmethod
    def _uncertain_edge_mask(alpha: np.ndarray) -> np.ndarray:
        soft = (alpha > 0.05) & (alpha < 0.95)
        if not np.any(soft):
            return np.zeros_like(alpha, dtype=bool)
        return soft

    @staticmethod
    def _large_component_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        result = np.zeros_like(mask, dtype=bool)
        for label in range(1, labels_count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                result |= labels == label
        return result

    @staticmethod
    def _fill_internal_holes(mask: np.ndarray, allowed_region: np.ndarray) -> np.ndarray:
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
