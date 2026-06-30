"""Quality-driven matte coordinator."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ..analysis.region_ownership import RegionOwnershipAnalyzer
from ..config import BackgroundMode, MattingConfig
from ..evaluation.matte_quality import MatteQualityEvaluator
from .candidates.base import CandidateGenerator
from .candidates.birefnet import BiRefNetCandidateGenerator
from .candidates.matanyone2 import MatAnyone2CandidateGenerator
from .candidates.sam2_guided import SAM2GuidedCandidateGenerator
from .candidates.traditional import TraditionalCandidateGenerator
from .candidates.types import (
    CandidateGenerationResult,
    CandidateSkipReason,
    MatteCandidateSequence,
)
from .quality_selector import QualitySelector

logger = logging.getLogger(__name__)


class QualityDrivenMatte:
    """Generate multiple matte candidates, evaluate them, and select by region."""

    def __init__(
        self,
        config: MattingConfig,
        *,
        background_mode: BackgroundMode | None = None,
        generators: Sequence[CandidateGenerator] | None = None,
        candidate_engines: dict[str, Any] | None = None,
    ):
        self.config = config
        self.background_mode = background_mode
        self.candidate_engines = candidate_engines or {}
        self.generators = list(generators) if generators is not None else self._build_generators()
        self.region_analyzer = RegionOwnershipAnalyzer()
        self.quality_evaluator = MatteQualityEvaluator()
        self.selector = QualitySelector()
        self.last_quality_selection: dict[str, Any] | None = None
        self.last_black_effect_enhancement_history: list[dict[str, Any]] = []

    def generate_sequence(
        self,
        frames: Sequence[np.ndarray],
        *,
        cancel_check=None,
        progress_callback=None,
    ) -> list[np.ndarray]:
        frame_shapes = [tuple(frame.shape[:2]) for frame in frames]
        candidates: list[MatteCandidateSequence] = []
        skipped: list[dict[str, Any]] = []
        self.last_black_effect_enhancement_history = []

        for generator in self.generators:
            name = getattr(generator, "name", generator.__class__.__name__)
            try:
                result = generator.generate(
                    frames,
                    frame_shapes=frame_shapes,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                logger.warning("Quality candidate %s failed: %s", name, exc)
                result = CandidateGenerationResult(
                    candidate=None,
                    skipped=True,
                    skip_reason=CandidateSkipReason.GENERATION_FAILED,
                    message=str(exc),
                )
            if result.candidate is not None:
                candidates.append(result.candidate)
                history = result.candidate.diagnostics.get("black_effect_enhancement_history")
                if isinstance(history, list):
                    self.last_black_effect_enhancement_history.extend(
                        dict(item) for item in history if isinstance(item, dict)
                    )
            else:
                skip_info = result.to_skip_dict(str(name))
                logger.info(
                    "Quality candidate skipped: name=%s reason=%s message=%s",
                    skip_info["name"],
                    skip_info["reason"],
                    skip_info["message"],
                )
                skipped.append(skip_info)

        if not candidates:
            self.last_quality_selection = {
                "available": False,
                "candidate_count": 0,
                "selected_model_counts": {},
                "candidate_quality": {},
                "skipped_candidates": skipped,
            }
            raise RuntimeError("No quality selection candidates were generated")

        ownerships = [
            self.region_analyzer.analyze(frame, candidates[0].alphas[index])
            for index, frame in enumerate(frames)
        ]
        quality_report = self.quality_evaluator.evaluate(
            frames=frames,
            candidates=candidates,
            ownerships=ownerships,
        )
        selection = self.selector.select(
            candidates=candidates,
            quality_report=quality_report,
            ownerships=ownerships,
            skipped_candidates=skipped,
        )
        self.last_quality_selection = selection.to_dict()
        return selection.alphas

    def _build_generators(self) -> list[CandidateGenerator]:
        generators: list[CandidateGenerator] = []
        for model_name in getattr(self.config, "quality_candidate_models", ("traditional",)):
            if model_name == "traditional":
                generators.append(
                    TraditionalCandidateGenerator(self.config, background_mode=self.background_mode)
                )
            elif model_name == "matanyone2":
                generators.append(
                    MatAnyone2CandidateGenerator(
                        self.config,
                        engine=self.candidate_engines.get("matanyone2"),
                    )
                )
            elif model_name == "birefnet":
                generators.append(
                    BiRefNetCandidateGenerator(
                        self.config,
                        engine=self.candidate_engines.get("birefnet"),
                    )
                )
            elif model_name == "sam2":
                generators.append(
                    SAM2GuidedCandidateGenerator(
                        self.config,
                        engine=self.candidate_engines.get("sam2"),
                    )
                )
            else:
                generators.append(
                    _SkippedCandidateGenerator(
                        name=model_name,
                        reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                        message=f"{model_name} candidate wrapper is not wired in this phase",
                    )
                )
        return generators


class _SkippedCandidateGenerator:
    def __init__(self, *, name: str, reason: CandidateSkipReason, message: str):
        self.name = name
        self.reason = reason
        self.message = message

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        return CandidateGenerationResult(
            candidate=None,
            skipped=True,
            skip_reason=self.reason,
            message=self.message,
        )
