"""Region-level matte candidate selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..analysis.region_ownership import RegionOwnership
from ..evaluation.matte_quality import CandidateQualityReport, REGION_FIELDS
from .candidates.types import MatteCandidateSequence


@dataclass(frozen=True)
class SelectionDecision:
    frame_index: int
    region: str
    candidate_name: str
    score: float
    pixel_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "region": self.region,
            "candidate_name": self.candidate_name,
            "score": float(self.score),
            "pixel_count": int(self.pixel_count),
        }


@dataclass(frozen=True)
class QualitySelectionResult:
    alphas: list[np.ndarray]
    decisions: tuple[SelectionDecision, ...]
    candidate_quality: dict[str, Any]
    skipped_candidates: list[dict[str, Any]]

    @property
    def selected_model_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.decisions:
            counts[decision.candidate_name] = counts.get(decision.candidate_name, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.alphas),
            "candidate_count": len(self.candidate_quality),
            "selected_model_counts": dict(self.selected_model_counts),
            "candidate_quality": self.candidate_quality,
            "skipped_candidates": list(self.skipped_candidates),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


class QualitySelector:
    def select(
        self,
        *,
        candidates: Sequence[MatteCandidateSequence],
        quality_report: CandidateQualityReport,
        ownerships: Sequence[RegionOwnership],
        skipped_candidates: Sequence[dict[str, Any]] | None = None,
    ) -> QualitySelectionResult:
        if not candidates:
            return QualitySelectionResult(
                alphas=[],
                decisions=(),
                candidate_quality={},
                skipped_candidates=list(skipped_candidates or []),
            )

        frame_count = len(candidates[0].alphas)
        for candidate in candidates:
            if len(candidate.alphas) != frame_count:
                raise ValueError("candidate frame counts do not match")
        if len(ownerships) != frame_count:
            raise ValueError("ownership count does not match candidate frame count")

        quality_by_frame = quality_report.by_frame_candidate()
        selected_alphas: list[np.ndarray] = []
        decisions: list[SelectionDecision] = []

        for frame_index in range(frame_count):
            result_alpha = candidates[0].alphas[frame_index].copy()
            ownership = ownerships[frame_index]
            assigned = np.zeros(result_alpha.shape, dtype=bool)

            for region in REGION_FIELDS:
                mask = getattr(ownership, region)
                if not np.any(mask):
                    continue
                best_candidate = self._best_candidate(
                    candidates,
                    quality_by_frame,
                    frame_index=frame_index,
                    region=region,
                )
                if best_candidate is None:
                    continue
                candidate, score = best_candidate
                region_mask = mask & (~assigned)
                if not np.any(region_mask):
                    continue
                result_alpha[region_mask] = candidate.alphas[frame_index][region_mask]
                assigned |= region_mask
                decisions.append(
                    SelectionDecision(
                        frame_index=frame_index,
                        region=region,
                        candidate_name=candidate.name,
                        score=score,
                        pixel_count=int(np.count_nonzero(region_mask)),
                    )
                )

            selected_alphas.append(np.clip(result_alpha, 0.0, 1.0).astype(np.float32, copy=False))

        return QualitySelectionResult(
            alphas=selected_alphas,
            decisions=tuple(decisions),
            candidate_quality=quality_report.to_summary(),
            skipped_candidates=list(skipped_candidates or []),
        )

    @staticmethod
    def _best_candidate(
        candidates: Sequence[MatteCandidateSequence],
        quality_by_frame: dict[tuple[int, str], Any],
        *,
        frame_index: int,
        region: str,
    ) -> tuple[MatteCandidateSequence, float] | None:
        best: tuple[MatteCandidateSequence, float] | None = None
        for candidate in candidates:
            quality = quality_by_frame.get((frame_index, candidate.name))
            if quality is None:
                continue
            score = float(quality.region_scores.get(region, quality.overall_score))
            if best is None or score > best[1]:
                best = (candidate, score)
        return best
