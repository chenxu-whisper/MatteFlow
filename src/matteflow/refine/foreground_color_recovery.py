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
        self.last_diagnostics = self._empty_diagnostics()
        self.last_sequence_diagnostics = self._empty_sequence_diagnostics()

    def recover(self, frame: np.ndarray, alpha: np.ndarray, ownership=None, screen_rgb=None) -> np.ndarray:
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
            self.last_diagnostics = self._build_diagnostics(
                repair_mask,
                np.zeros_like(alpha_f, dtype=np.float32),
                self._screen_color_or_estimate(frame_f, alpha_f, screen_rgb),
            )
            return np.asarray(frame, dtype=np.uint8).copy()

        screen_rgb = self._screen_color_or_estimate(frame_f, alpha_f, screen_rgb)
        recovered = self._unmix_foreground(frame_f, alpha_f, screen_rgb)
        weight = self._recovery_weight(frame_f, alpha_f, repair_mask)
        weight *= self._quality_gate_weight(frame_f, recovered, alpha_f, repair_mask)
        self.last_diagnostics = self._build_diagnostics(repair_mask, weight, screen_rgb)
        if not np.any(weight > 0.0):
            return np.asarray(frame, dtype=np.uint8).copy()

        result = frame_f * (1.0 - weight[..., np.newaxis]) + recovered * weight[..., np.newaxis]
        return np.clip(np.rint(result), 0, 255).astype(np.uint8)

    def recover_sequence(self, frames, alphas, ownerships=None) -> list[np.ndarray]:
        """Recover a frame sequence using one stable screen estimate and aggregate diagnostics."""
        frames_list = list(frames)
        alphas_list = list(alphas)
        ownerships_list = ownerships if ownerships is not None else [None] * len(frames_list)
        screen_rgb = self._estimate_sequence_screen_color(frames_list, alphas_list)

        recovered_frames = []
        attempted = 0
        accepted = 0
        rejected = 0
        weighted_sum = 0.0
        for frame, alpha, ownership in zip(frames_list, alphas_list, ownerships_list):
            recovered = self.recover(frame, alpha, ownership=ownership, screen_rgb=screen_rgb)
            recovered_frames.append(recovered)
            diagnostics = self.last_diagnostics
            attempted += int(diagnostics["attempted_pixels"])
            accepted += int(diagnostics["accepted_pixels"])
            rejected += int(diagnostics["rejected_pixels"])
            weighted_sum += float(diagnostics["mean_weight"]) * int(diagnostics["accepted_pixels"])

        mean_weight = weighted_sum / max(float(accepted), 1.0)
        self.last_sequence_diagnostics = {
            "frames": len(recovered_frames),
            "attempted_pixels": attempted,
            "accepted_pixels": accepted,
            "rejected_pixels": rejected,
            "mean_weight": float(mean_weight),
            "screen_rgb": self._screen_color_to_list(screen_rgb),
        }
        return recovered_frames

    def _screen_color_or_estimate(self, frame: np.ndarray, alpha: np.ndarray, screen_rgb) -> np.ndarray:
        if screen_rgb is not None:
            return np.asarray(screen_rgb, dtype=np.float32)
        return self._estimate_screen_color(frame, alpha)

    def _estimate_sequence_screen_color(self, frames: list[np.ndarray], alphas: list[np.ndarray]) -> np.ndarray:
        if self.config.key_color is not None:
            return np.array(self.config.key_color, dtype=np.float32)

        candidates = []
        for frame, alpha in zip(frames, alphas):
            frame_f = frame.astype(np.float32, copy=False)
            alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
            pixels = self._screen_candidate_pixels(frame_f, alpha_f, strict_background=True)
            if pixels.size:
                candidates.append(pixels)
        if not candidates:
            return np.array([0.0, 255.0, 0.0], dtype=np.float32)
        return np.median(np.concatenate(candidates, axis=0), axis=0).astype(np.float32)

    def _estimate_screen_color(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self.config.key_color is not None:
            return np.array(self.config.key_color, dtype=np.float32)

        pixels = self._screen_candidate_pixels(frame, alpha)
        if pixels.size:
            return np.median(pixels, axis=0).astype(np.float32)
        return np.array([0.0, 255.0, 0.0], dtype=np.float32)

    def _screen_candidate_pixels(
        self,
        frame: np.ndarray,
        alpha: np.ndarray,
        strict_background: bool = False,
    ) -> np.ndarray:
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
        if strict_background:
            return frame[candidates].astype(np.float32, copy=False)
        if int(candidates.sum()) < 16:
            candidates = (alpha <= 0.06) & screen_like
        if int(candidates.sum()) < 16:
            candidates = border_mask & screen_like
        if int(candidates.sum()) < 16:
            return np.empty((0, 3), dtype=np.float32)
        return frame[candidates].astype(np.float32, copy=False)

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

    @staticmethod
    def _quality_gate_weight(
        frame: np.ndarray,
        recovered: np.ndarray,
        alpha: np.ndarray,
        repair_mask: np.ndarray,
    ) -> np.ndarray:
        before_green_bias = frame[:, :, 1] - np.maximum(frame[:, :, 0], frame[:, :, 2])
        after_green_bias = recovered[:, :, 1] - np.maximum(recovered[:, :, 0], recovered[:, :, 2])
        green_improvement = before_green_bias - after_green_bias
        color_delta = np.mean(np.abs(recovered - frame), axis=2)
        destructive = (alpha < 0.16) | ((color_delta > 128.0) & (green_improvement < 24.0))
        improves_spill = green_improvement > 2.0
        accepted = repair_mask & improves_spill & (~destructive)
        return accepted.astype(np.float32, copy=False)

    @classmethod
    def _build_diagnostics(cls, repair_mask: np.ndarray, weight: np.ndarray, screen_rgb: np.ndarray) -> dict:
        attempted = int(repair_mask.sum())
        accepted = int((weight > 0.0).sum())
        return {
            "attempted_pixels": attempted,
            "accepted_pixels": accepted,
            "rejected_pixels": max(attempted - accepted, 0),
            "mean_weight": float(weight[weight > 0.0].mean()) if accepted else 0.0,
            "screen_rgb": cls._screen_color_to_list(screen_rgb),
        }

    @staticmethod
    def _screen_color_to_list(screen_rgb: np.ndarray) -> list[float]:
        return [float(value) for value in np.rint(np.asarray(screen_rgb, dtype=np.float32)).tolist()]

    @classmethod
    def _empty_diagnostics(cls) -> dict:
        return {
            "attempted_pixels": 0,
            "accepted_pixels": 0,
            "rejected_pixels": 0,
            "mean_weight": 0.0,
            "screen_rgb": [0.0, 255.0, 0.0],
        }

    @classmethod
    def _empty_sequence_diagnostics(cls) -> dict:
        diagnostics = cls._empty_diagnostics()
        diagnostics["frames"] = 0
        return diagnostics
