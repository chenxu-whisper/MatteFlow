"""Guided SAM2 candidate generator."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ...config import MattingConfig
from .base import TimedCandidateGenerator
from .types import CandidateGenerationResult, CandidateSkipReason


class SAM2GuidedCandidateGenerator(TimedCandidateGenerator):
    name = "sam2"
    source = "sam2_guided"

    def __init__(self, config: MattingConfig, engine=None, first_frame_mask: np.ndarray | None = None):
        self.config = config
        self.engine = engine
        self.first_frame_mask = first_frame_mask

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        del progress_callback
        if self.first_frame_mask is None:
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.GUIDANCE_MISSING,
                message="SAM2 candidate requires guidance and is not wired in this phase",
            )
        if self.engine is None or (
            getattr(self.engine, "predictor", None) is None and getattr(self.engine, "model", None) is None
        ):
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message="SAM2 candidate engine is not available",
            )

        start_time = time.perf_counter()
        masks = self.engine.track_video(list(frames), self.first_frame_mask, cancel_check=cancel_check)
        alphas = [np.asarray(mask, dtype=np.float32) for mask in masks]
        return self._build_candidate(
            start_time=start_time,
            alphas=alphas,
            confidences=[None] * len(alphas),
            frame_shapes=frame_shapes,
            diagnostics={"available": True, "model": self.name, "guided": True},
        )
