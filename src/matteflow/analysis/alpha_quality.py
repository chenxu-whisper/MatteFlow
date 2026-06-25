"""Alpha matte quality diagnostics and debug overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class AlphaQualityReport:
    """Aggregate quality signals for a frame sequence."""

    frame_count: int
    mean_edge_uncertainty: float
    speckle_pixels: int
    hole_pixels: int
    background_residue: float
    temporal_flicker: float
    overall_score: float


class AlphaQualityAnalyzer:
    """Compute lightweight, model-free alpha matte quality signals."""

    EDGE_COLOR = np.array([255, 180, 0], dtype=np.uint8)
    SPECKLE_COLOR = np.array([255, 0, 255], dtype=np.uint8)
    HOLE_COLOR = np.array([0, 80, 255], dtype=np.uint8)

    def analyze_sequence(
        self,
        frames: Sequence[np.ndarray],
        alphas: Sequence[np.ndarray],
    ) -> AlphaQualityReport:
        del frames
        if not alphas:
            return AlphaQualityReport(0, 0.0, 0, 0, 0.0, 0.0, 1.0)

        edge_scores = []
        speckle_pixels = 0
        hole_pixels = 0
        residue_scores = []
        for alpha in alphas:
            alpha_f = self._as_alpha(alpha)
            edge_scores.append(float(self._uncertain_edge_mask(alpha_f).mean()))
            speckles = self._speckle_mask(alpha_f)
            holes = self._hole_mask(alpha_f)
            speckle_pixels += int(speckles.sum())
            hole_pixels += int(holes.sum())
            residue_scores.append(float(alpha_f[alpha_f < 0.05].sum() / max(alpha_f.size, 1)))

        flicker = self._temporal_flicker(alphas)
        mean_edge = float(np.mean(edge_scores)) if edge_scores else 0.0
        residue = float(np.mean(residue_scores)) if residue_scores else 0.0
        normalized_artifacts = min(1.0, (speckle_pixels + hole_pixels) / max(sum(a.size for a in alphas), 1))
        penalty = min(1.0, mean_edge * 0.35 + residue * 0.20 + flicker * 0.35 + normalized_artifacts * 0.50)
        return AlphaQualityReport(
            frame_count=len(alphas),
            mean_edge_uncertainty=mean_edge,
            speckle_pixels=speckle_pixels,
            hole_pixels=hole_pixels,
            background_residue=residue,
            temporal_flicker=flicker,
            overall_score=float(np.clip(1.0 - penalty, 0.0, 1.0)),
        )

    def build_debug_overlay(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Return RGB overlay that marks uncertain edges, speckles, and holes."""
        overlay = np.asarray(frame, dtype=np.uint8).copy()
        alpha_f = self._as_alpha(alpha)
        edge_mask = self._uncertain_edge_mask(alpha_f)
        speckle_mask = self._speckle_mask(alpha_f)
        hole_mask = self._hole_mask(alpha_f)

        overlay[edge_mask] = self.EDGE_COLOR
        overlay[speckle_mask] = self.SPECKLE_COLOR
        overlay[hole_mask] = self.HOLE_COLOR
        return overlay

    @staticmethod
    def _as_alpha(alpha: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)

    def _uncertain_edge_mask(self, alpha: np.ndarray) -> np.ndarray:
        soft = (alpha > 0.05) & (alpha < 0.95)
        if not np.any(soft):
            return np.zeros_like(alpha, dtype=bool)
        fg = alpha >= 0.95
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        near_fg = cv2.dilate(fg.astype(np.uint8), kernel, iterations=1).astype(bool)
        return soft & near_fg

    def _speckle_mask(self, alpha: np.ndarray) -> np.ndarray:
        candidate = (alpha > 0.03) & (alpha < 0.95)
        edge = self._uncertain_edge_mask(alpha)
        isolated = candidate & (~edge)
        return self._small_components(isolated, max_area=12)

    def _hole_mask(self, alpha: np.ndarray) -> np.ndarray:
        fg = alpha >= 0.70
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
        return self._large_components(closed & (~fg), min_area=5)

    @staticmethod
    def _small_components(mask: np.ndarray, max_area: int) -> np.ndarray:
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        result = np.zeros_like(mask, dtype=bool)
        for label in range(1, labels_count):
            if int(stats[label, cv2.CC_STAT_AREA]) <= max_area:
                result |= labels == label
        return result

    @staticmethod
    def _large_components(mask: np.ndarray, min_area: int) -> np.ndarray:
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        result = np.zeros_like(mask, dtype=bool)
        for label in range(1, labels_count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                result |= labels == label
        return result

    @staticmethod
    def _temporal_flicker(alphas: Sequence[np.ndarray]) -> float:
        if len(alphas) <= 1:
            return 0.0
        diffs = []
        for prev, curr in zip(alphas, alphas[1:]):
            prev_f = np.clip(np.asarray(prev, dtype=np.float32), 0.0, 1.0)
            curr_f = np.clip(np.asarray(curr, dtype=np.float32), 0.0, 1.0)
            diffs.append(float(np.abs(curr_f - prev_f).mean()))
        return float(np.mean(diffs)) if diffs else 0.0
