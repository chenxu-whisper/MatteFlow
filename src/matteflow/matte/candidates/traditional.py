"""Traditional matte candidate wrapper."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from ...config import BackgroundMode, MattingConfig
from .base import TimedCandidateGenerator
from .types import CandidateGenerationResult


class TraditionalCandidateGenerator(TimedCandidateGenerator):
    name = "traditional"
    source = "traditional"

    def __init__(self, config: MattingConfig, background_mode: BackgroundMode | None = None):
        self.config = config
        self.background_mode = background_mode or getattr(config, "background_mode", BackgroundMode.AUTO)

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        start_time = time.perf_counter()
        alphas = []
        mode = self._effective_mode()
        engine = self._build_engine(mode)

        for index, frame in enumerate(frames):
            if cancel_check is not None and cancel_check():
                from ...errors import JobCancelledError

                raise JobCancelledError("Candidate generation cancelled by user")
            alphas.append(engine.generate(frame))
            if progress_callback and index % max(1, len(frames) // 20) == 0:
                progress_callback(index, len(frames))
        diagnostics = {"available": True, "background_mode": mode.value}
        effect_history = getattr(engine, "effect_enhancement_history", None)
        if effect_history:
            diagnostics["black_effect_enhancement_history"] = [
                dict(item) for item in effect_history
            ]

        return self._build_candidate(
            start_time=start_time,
            alphas=alphas,
            confidences=[None] * len(alphas),
            frame_shapes=frame_shapes,
            diagnostics=diagnostics,
        )

    def _effective_mode(self) -> BackgroundMode:
        if self.background_mode == BackgroundMode.BLACK_BACKGROUND:
            return BackgroundMode.BLACK_BACKGROUND
        return BackgroundMode.GREEN_SCREEN

    def _build_engine(self, mode: BackgroundMode):
        if mode == BackgroundMode.BLACK_BACKGROUND:
            from ..black_background_matte import BlackBackgroundMatte

            return BlackBackgroundMatte(self.config)
        from ..green_screen_matte import GreenScreenMatte

        return GreenScreenMatte(self.config)
