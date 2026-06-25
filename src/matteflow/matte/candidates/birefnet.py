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

    def __init__(self, config: MattingConfig, engine=None, engine_factory=None):
        self.config = config
        self.engine = engine
        self.engine_factory = engine_factory or self._default_engine_factory

    def _default_engine_factory(self, config: MattingConfig):
        from ..birefnet_matte import BiRefNetMatte

        return BiRefNetMatte(config)

    def _ensure_engine(self) -> CandidateGenerationResult | None:
        if self.engine is not None:
            if getattr(self.engine, "model", None) is not None:
                return None
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message="BiRefNet candidate engine is not available",
            )

        if not getattr(self.config, "quality_birefnet_auto_load", False):
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message="BiRefNet candidate engine is not available; auto-load is disabled",
            )

        try:
            self.engine = self.engine_factory(self.config)
        except Exception as exc:
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message=f"BiRefNet auto-load failed: {exc}",
            )

        if getattr(self.engine, "model", None) is None:
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message="BiRefNet auto-load completed but model is unavailable",
            )
        return None

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        skip_result = self._ensure_engine()
        if skip_result is not None:
            return skip_result

        start_time = time.perf_counter()
        kwargs = {}
        params = inspect.signature(self.engine.generate_sequence).parameters
        if "progress_callback" in params:
            kwargs["progress_callback"] = progress_callback
        if "cancel_check" in params:
            kwargs["cancel_check"] = cancel_check
        try:
            alphas = self.engine.generate_sequence(list(frames), **kwargs)
        except Exception as exc:
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.GENERATION_FAILED,
                message=str(exc),
            )
        return self._build_candidate(
            start_time=start_time,
            alphas=alphas,
            confidences=[None] * len(alphas),
            frame_shapes=frame_shapes,
            diagnostics={"available": True, "model": self.name},
        )
