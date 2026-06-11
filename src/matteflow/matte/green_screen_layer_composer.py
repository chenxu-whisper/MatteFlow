"""Competitive green-screen layer composition contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerCandidate:
    """Candidate layer alpha and its per-pixel confidence."""

    alpha: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class LayerOwnership:
    """Per-pixel ownership masks for the competitive layers."""

    subject: np.ndarray
    effect: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class CompetitiveLayerResult:
    """Shape-stable result returned by the competitive layer composer."""

    final_alpha: np.ndarray
    subject_alpha_out: np.ndarray
    effect_alpha_out: np.ndarray
    ownership: LayerOwnership
    debug_layers: dict[str, np.ndarray]


class GreenScreenCompetitiveLayerComposer:
    """Composes subject/effect candidates with explicit debug layer outputs."""

    def compose(
        self,
        *,
        subject: LayerCandidate,
        effect: LayerCandidate,
    ) -> CompetitiveLayerResult:
        shape = self._require_candidate_shape("subject", subject)
        self._require_candidate_shape("effect", effect, expected_shape=shape)

        subject_alpha = self._as_alpha(subject.alpha)
        effect_alpha = self._as_alpha(effect.alpha)
        subject_confidence = self._as_alpha(subject.confidence)
        effect_confidence = self._as_alpha(effect.confidence)

        subject_owns = subject_confidence >= effect_confidence
        effect_owns = (~subject_owns) & (effect_confidence > 0.0)
        background_owns = (subject_confidence <= 0.0) & (effect_confidence <= 0.0)
        subject_owns = subject_owns & (~background_owns)

        subject_alpha_out = np.where(subject_owns, subject_alpha, 0.0).astype(np.float32)
        effect_alpha_out = np.where(effect_owns, effect_alpha, 0.0).astype(np.float32)
        final_alpha = np.maximum(subject_alpha_out, effect_alpha_out).astype(np.float32)
        ownership = LayerOwnership(
            subject=subject_owns.astype(np.float32),
            effect=effect_owns.astype(np.float32),
            background=background_owns.astype(np.float32),
        )
        debug_layers = {
            "subject_alpha": subject_alpha,
            "subject_confidence": subject_confidence,
            "effect_alpha": effect_alpha,
            "effect_confidence": effect_confidence,
            "subject_alpha_out": subject_alpha_out,
            "effect_alpha_out": effect_alpha_out,
            "final_alpha": final_alpha,
            "ownership_subject": ownership.subject,
            "ownership_effect": ownership.effect,
            "ownership_background": ownership.background,
        }
        return CompetitiveLayerResult(
            final_alpha=final_alpha,
            subject_alpha_out=subject_alpha_out,
            effect_alpha_out=effect_alpha_out,
            ownership=ownership,
            debug_layers=debug_layers,
        )

    @staticmethod
    def _require_candidate_shape(
        name: str,
        candidate: LayerCandidate,
        expected_shape: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        shape = tuple(np.asarray(candidate.alpha).shape)
        confidence_shape = tuple(np.asarray(candidate.confidence).shape)
        if len(shape) != 2:
            raise ValueError(f"{name}.alpha must be 2D, got shape {shape}")
        if confidence_shape != shape:
            raise ValueError(
                f"{name}.confidence shape {confidence_shape} does not match alpha shape {shape}"
            )
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(f"{name}.alpha shape {shape} does not match {expected_shape}")
        return shape

    @staticmethod
    def _as_alpha(value: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)

__all__ = [
    "CompetitiveLayerResult",
    "GreenScreenCompetitiveLayerComposer",
    "LayerCandidate",
    "LayerOwnership",
]
