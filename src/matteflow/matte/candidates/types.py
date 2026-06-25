"""Candidate matte contracts for quality-driven selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np


class CandidateSkipReason(str, Enum):
    MODEL_UNAVAILABLE = "model_unavailable"
    GUIDANCE_MISSING = "guidance_missing"
    DISABLED_BY_CONFIG = "disabled_by_config"
    GENERATION_FAILED = "generation_failed"


@dataclass(frozen=True)
class MatteCandidate:
    name: str
    alpha: np.ndarray
    confidence: np.ndarray | None
    source: str
    runtime_ms: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MatteCandidateSequence:
    name: str
    alphas: list[np.ndarray]
    confidences: list[np.ndarray | None]
    source: str
    runtime_ms: float
    diagnostics: dict[str, Any]

    @classmethod
    def from_raw(
        cls,
        *,
        name: str,
        alphas: Sequence[np.ndarray],
        confidences: Sequence[np.ndarray | None] | None,
        source: str,
        runtime_ms: float,
        diagnostics: dict[str, Any] | None,
        frame_shapes: Sequence[tuple[int, int]],
    ) -> "MatteCandidateSequence":
        raw_alphas = list(alphas)
        if confidences is None:
            raw_confidences = [None] * len(raw_alphas)
        else:
            raw_confidences = list(confidences)
        if len(raw_confidences) != len(raw_alphas):
            raise ValueError(f"{name}.confidences length does not match alphas length")
        if len(frame_shapes) != len(raw_alphas):
            raise ValueError(f"{name}.frame_shapes length does not match alphas length")

        normalized_alphas = []
        normalized_confidences = []
        for index, (alpha, confidence, expected_shape) in enumerate(
            zip(raw_alphas, raw_confidences, frame_shapes)
        ):
            alpha_f = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
            if tuple(alpha_f.shape) != tuple(expected_shape):
                raise ValueError(
                    f"{name}.alpha shape {alpha_f.shape} does not match frame shape "
                    f"{expected_shape} at index {index}"
                )
            normalized_alphas.append(alpha_f)
            if confidence is None:
                normalized_confidences.append(None)
                continue
            confidence_f = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
            if confidence_f.shape != alpha_f.shape:
                raise ValueError(
                    f"{name}.confidence shape {confidence_f.shape} does not match alpha "
                    f"shape {alpha_f.shape} at index {index}"
                )
            normalized_confidences.append(confidence_f)

        return cls(
            name=str(name),
            alphas=normalized_alphas,
            confidences=normalized_confidences,
            source=str(source),
            runtime_ms=float(runtime_ms),
            diagnostics=dict(diagnostics or {}),
        )


@dataclass(frozen=True)
class CandidateGenerationResult:
    candidate: MatteCandidateSequence | None = None
    skipped: bool = False
    skip_reason: CandidateSkipReason | None = None
    message: str = ""

    def to_skip_dict(self, name: str) -> dict[str, Any]:
        reason = self.skip_reason.value if self.skip_reason is not None else None
        return {
            "name": str(name),
            "reason": reason,
            "message": str(self.message),
        }
