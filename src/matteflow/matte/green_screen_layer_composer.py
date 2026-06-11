"""Competitive green-screen layer composition contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerCandidate:
    """Candidate layer alpha, confidence, and optional ownership evidence."""

    alpha: np.ndarray
    confidence: np.ndarray
    evidence: np.ndarray | None = None


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

    _BACKGROUND_SUBJECT_SUPPORT_MAX = 0.20
    _BACKGROUND_EFFECT_EVIDENCE_MAX = 0.35

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
        subject_evidence = self._candidate_evidence(subject, subject_confidence)
        effect_evidence = self._candidate_evidence(effect, effect_confidence)
        subject_support = np.maximum.reduce(
            [subject_alpha, subject_confidence, subject_evidence]
        )
        effect_support = np.maximum(effect_confidence, effect_evidence)
        background_suppression = (
            (subject_support <= self._BACKGROUND_SUBJECT_SUPPORT_MAX)
            & (effect_support <= self._BACKGROUND_EFFECT_EVIDENCE_MAX)
        )

        effect_over_subject_evidence = (
            (effect_alpha > 0.0)
            & (effect_evidence > 0.0)
            & (effect_evidence > subject_evidence)
        )
        subject_owns = (subject_confidence >= effect_confidence) & (~effect_over_subject_evidence)
        effect_owns = effect_over_subject_evidence | ((~subject_owns) & (effect_confidence > 0.0))
        background_owns = (
            (
                (subject_confidence <= 0.0)
                & (effect_confidence <= 0.0)
                & (~effect_over_subject_evidence)
            )
            | background_suppression
        )
        subject_owns = subject_owns & (~background_owns)
        effect_owns = effect_owns & (~background_owns)

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
            "subject_evidence": subject_evidence,
            "effect_evidence": effect_evidence,
            "effect_over_subject_evidence": effect_over_subject_evidence.astype(np.float32),
            "background_suppression": background_suppression.astype(np.float32),
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
        evidence_shape = (
            None
            if candidate.evidence is None
            else tuple(np.asarray(candidate.evidence).shape)
        )
        if len(shape) != 2:
            raise ValueError(f"{name}.alpha must be 2D, got shape {shape}")
        if confidence_shape != shape:
            raise ValueError(
                f"{name}.confidence shape {confidence_shape} does not match alpha shape {shape}"
            )
        if evidence_shape is not None and evidence_shape != shape:
            raise ValueError(
                f"{name}.evidence shape {evidence_shape} does not match alpha shape {shape}"
            )
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(
                f"{name}.alpha shape {shape} does not match subject alpha shape {expected_shape}"
            )
        return shape

    @staticmethod
    def _as_alpha(value: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)

    @classmethod
    def _candidate_evidence(
        cls,
        candidate: LayerCandidate,
        fallback: np.ndarray,
    ) -> np.ndarray:
        if candidate.evidence is None:
            return fallback
        return cls._as_alpha(candidate.evidence)

__all__ = [
    "CompetitiveLayerResult",
    "GreenScreenCompetitiveLayerComposer",
    "LayerCandidate",
    "LayerOwnership",
]
