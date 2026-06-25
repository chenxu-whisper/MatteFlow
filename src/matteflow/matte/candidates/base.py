"""Base helpers for matte candidate generators."""

from __future__ import annotations

import time
from typing import Protocol, Sequence

import numpy as np

from .types import CandidateGenerationResult, MatteCandidateSequence


class CandidateGenerator(Protocol):
    name: str

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        ...


class TimedCandidateGenerator:
    name: str
    source: str

    def _build_candidate(
        self,
        *,
        start_time: float,
        alphas: Sequence[np.ndarray],
        confidences: Sequence[np.ndarray | None] | None,
        frame_shapes: Sequence[tuple[int, int]],
        diagnostics: dict,
    ) -> CandidateGenerationResult:
        candidate = MatteCandidateSequence.from_raw(
            name=self.name,
            alphas=alphas,
            confidences=confidences,
            source=self.source,
            runtime_ms=(time.perf_counter() - start_time) * 1000.0,
            diagnostics=diagnostics,
            frame_shapes=frame_shapes,
        )
        return CandidateGenerationResult(candidate=candidate)
