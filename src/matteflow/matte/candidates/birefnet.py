"""BiRefNet candidate generator."""

from __future__ import annotations

import inspect
import time
from typing import Sequence

import numpy as np

from ...config import MattingConfig
from .base import TimedCandidateGenerator
from .types import CandidateGenerationResult, CandidateSkipReason


class BiRefNetCandidateGenerator(TimedCandidateGenerator):
    name = "birefnet"
    source = "birefnet"

    def __init__(self, config: MattingConfig, engine=None):
        self.config = config
        self.engine = engine

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        if self.engine is None or getattr(self.engine, "model", None) is None:
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message="BiRefNet candidate engine is not available",
            )

        start_time = time.perf_counter()
        kwargs = {}
        params = inspect.signature(self.engine.generate_sequence).parameters
        if "progress_callback" in params:
            kwargs["progress_callback"] = progress_callback
        if "cancel_check" in params:
            kwargs["cancel_check"] = cancel_check
        alphas = self.engine.generate_sequence(list(frames), **kwargs)
        return self._build_candidate(
            start_time=start_time,
            alphas=alphas,
            confidences=[None] * len(alphas),
            frame_shapes=frame_shapes,
            diagnostics={"available": True, "model": self.name},
        )
