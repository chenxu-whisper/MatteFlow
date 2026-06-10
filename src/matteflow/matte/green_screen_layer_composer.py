"""Layer composer for green-screen subject and transparent effect mattes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class GreenScreenLayerComposer:
    """Compose independent green-screen subject and effect alpha layers."""

    def __init__(self, transparency_preserve: float = 0.7):
        self.transparency_preserve = float(np.clip(transparency_preserve, 0.0, 1.0))

    def compose(
        self,
        base_alpha: np.ndarray,
        ai_alpha: np.ndarray,
        *,
        effect_alpha: np.ndarray | None = None,
        subject_confidence: np.ndarray | None = None,
        semantic_subject_alpha: np.ndarray | None = None,
        non_screen_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compose one frame using a solid subject layer plus a recoverable effect layer."""
        base = self._as_alpha(base_alpha, "base_alpha")
        ai = self._as_alpha(ai_alpha, "ai_alpha")
        self._require_same_shape(base, ai, "ai_alpha")

        if self.transparency_preserve <= 0.0:
            return ai

        subject = self._solid_subject_layer(ai, semantic_subject_alpha, non_screen_mask)
        effect = self._effect_layer(base, effect_alpha, subject_confidence)
        return self._screen_blend(subject, effect)

    def compose_sequence(
        self,
        base_alphas: Sequence[np.ndarray],
        ai_alphas: Sequence[np.ndarray],
        *,
        effect_alphas: Sequence[np.ndarray | None] | None = None,
        subject_confidences: Sequence[np.ndarray | None] | None = None,
        semantic_subject_alphas: Sequence[np.ndarray | None] | None = None,
        non_screen_masks: Sequence[np.ndarray | None] | None = None,
    ) -> list[np.ndarray]:
        """Compose a sequence while preserving frame order."""
        frame_count = len(base_alphas)
        if len(ai_alphas) != frame_count:
            raise ValueError("base_alphas and ai_alphas must have the same length")

        effect_iter = self._optional_sequence(effect_alphas, frame_count, "effect_alphas")
        confidence_iter = self._optional_sequence(
            subject_confidences,
            frame_count,
            "subject_confidences",
        )
        semantic_iter = self._optional_sequence(
            semantic_subject_alphas,
            frame_count,
            "semantic_subject_alphas",
        )
        mask_iter = self._optional_sequence(non_screen_masks, frame_count, "non_screen_masks")

        return [
            self.compose(
                base_alpha,
                ai_alpha,
                effect_alpha=effect_alpha,
                subject_confidence=subject_confidence,
                semantic_subject_alpha=semantic_subject_alpha,
                non_screen_mask=non_screen_mask,
            )
            for (
                base_alpha,
                ai_alpha,
                effect_alpha,
                subject_confidence,
                semantic_subject_alpha,
                non_screen_mask,
            ) in zip(
                base_alphas,
                ai_alphas,
                effect_iter,
                confidence_iter,
                semantic_iter,
                mask_iter,
            )
        ]

    def _solid_subject_layer(
        self,
        ai_alpha: np.ndarray,
        semantic_subject_alpha: np.ndarray | None,
        non_screen_mask: np.ndarray | None,
    ) -> np.ndarray:
        subject = ai_alpha.copy()
        if semantic_subject_alpha is None:
            return subject

        semantic = self._as_alpha(semantic_subject_alpha, "semantic_subject_alpha")
        self._require_same_shape(subject, semantic, "semantic_subject_alpha")
        semantic_layer = np.clip(0.92 * self._smoothstep(semantic, 0.25, 0.75), 0.0, 0.92)
        if non_screen_mask is not None:
            mask = np.asarray(non_screen_mask, dtype=bool)
            self._require_same_shape(subject, mask, "non_screen_mask")
            semantic_layer = np.where(mask, semantic_layer, 0.0)

        return np.maximum(subject, semantic_layer).astype(np.float32, copy=False)

    def _effect_layer(
        self,
        base_alpha: np.ndarray,
        effect_alpha: np.ndarray | None,
        subject_confidence: np.ndarray | None,
    ) -> np.ndarray:
        if effect_alpha is None:
            return np.zeros_like(base_alpha, dtype=np.float32)

        effect = self._as_alpha(effect_alpha, "effect_alpha")
        self._require_same_shape(base_alpha, effect, "effect_alpha")
        effect = effect * self.transparency_preserve

        if subject_confidence is not None:
            confidence = self._as_alpha(subject_confidence, "subject_confidence")
            self._require_same_shape(base_alpha, confidence, "subject_confidence")
            effect = effect * (1.0 - confidence)

        return np.clip(effect, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _screen_blend(solid_alpha: np.ndarray, effect_alpha: np.ndarray) -> np.ndarray:
        return np.clip(solid_alpha + effect_alpha * (1.0 - solid_alpha), 0.0, 1.0).astype(
            np.float32,
            copy=False,
        )

    @staticmethod
    def _smoothstep(alpha: np.ndarray, low: float, high: float) -> np.ndarray:
        t = np.clip((alpha - low) / max(high - low, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def _as_alpha(alpha: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(alpha, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D alpha matte")
        return np.clip(array, 0.0, 1.0)

    @staticmethod
    def _require_same_shape(reference: np.ndarray, candidate: np.ndarray, name: str) -> None:
        if candidate.shape != reference.shape:
            raise ValueError(f"{name} shape must match base alpha shape")

    @staticmethod
    def _optional_sequence(
        values: Sequence[np.ndarray | None] | None,
        frame_count: int,
        name: str,
    ) -> Sequence[np.ndarray | None]:
        if values is None:
            return [None] * frame_count
        if len(values) != frame_count:
            raise ValueError(f"{name} must have the same length as base_alphas")
        return values


__all__ = ["GreenScreenLayerComposer"]
