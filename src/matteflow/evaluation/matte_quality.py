"""Rule-based matte candidate quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..analysis.region_ownership import RegionOwnership
from ..matte.candidates.types import MatteCandidateSequence

REGION_FIELDS = (
    "subject",
    "hair_edge",
    "luminous_prop",
    "transparent_effect",
    "background_residue",
    "uncertain_edge",
)


@dataclass(frozen=True)
class CandidateQuality:
    candidate_name: str
    frame_index: int
    overall_score: float
    region_scores: dict[str, float]
    signals: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "frame_index": int(self.frame_index),
            "overall_score": float(self.overall_score),
            "region_scores": {key: float(value) for key, value in self.region_scores.items()},
            "signals": dict(self.signals),
        }


@dataclass(frozen=True)
class CandidateQualityReport:
    qualities: tuple[CandidateQuality, ...]

    @property
    def by_candidate(self) -> dict[str, list[CandidateQuality]]:
        result: dict[str, list[CandidateQuality]] = {}
        for quality in self.qualities:
            result.setdefault(quality.candidate_name, []).append(quality)
        return result

    def by_frame_candidate(self) -> dict[tuple[int, str], CandidateQuality]:
        return {(quality.frame_index, quality.candidate_name): quality for quality in self.qualities}

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for name, qualities in self.by_candidate.items():
            summary[name] = {
                "overall_score": round(float(np.mean([q.overall_score for q in qualities])), 6),
                "frame_count": len(qualities),
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualities": [quality.to_dict() for quality in self.qualities],
            "summary": self.to_summary(),
        }


class MatteQualityEvaluator:
    def evaluate(
        self,
        *,
        frames: Sequence[np.ndarray],
        candidates: Sequence[MatteCandidateSequence],
        ownerships: Sequence[RegionOwnership],
    ) -> CandidateQualityReport:
        if len(frames) != len(ownerships):
            raise ValueError(
                f"frames length {len(frames)} does not match ownerships length {len(ownerships)}"
            )

        qualities: list[CandidateQuality] = []
        for candidate in candidates:
            if len(candidate.alphas) != len(frames):
                raise ValueError(
                    f"candidate frame count for {candidate.name} is {len(candidate.alphas)}; "
                    f"expected {len(frames)}"
                )
            for frame_index, alpha in enumerate(candidate.alphas):
                ownership = ownerships[frame_index]
                qualities.append(self._evaluate_frame(candidate.name, frame_index, alpha, ownership))
        return CandidateQualityReport(qualities=tuple(qualities))

    def _evaluate_frame(
        self,
        candidate_name: str,
        frame_index: int,
        alpha: np.ndarray,
        ownership: RegionOwnership,
    ) -> CandidateQuality:
        _validate_ownership_shapes(alpha, ownership)
        region_scores = {
            "subject": _mean_alpha_score(alpha, ownership.subject, target="high"),
            "hair_edge": _soft_alpha_score(alpha, ownership.hair_edge),
            "luminous_prop": _soft_alpha_score(alpha, ownership.luminous_prop),
            "transparent_effect": _soft_alpha_score(alpha, ownership.transparent_effect),
            "background_residue": _mean_alpha_score(alpha, ownership.background_residue, target="low"),
            "uncertain_edge": _soft_alpha_score(alpha, ownership.uncertain_edge),
        }
        weights = {
            "subject": 1.4,
            "hair_edge": 1.1,
            "luminous_prop": 1.2,
            "transparent_effect": 1.2,
            "background_residue": 1.3,
            "uncertain_edge": 0.8,
        }
        weighted = sum(region_scores[key] * weights[key] for key in region_scores)
        overall = weighted / sum(weights.values())
        return CandidateQuality(
            candidate_name=candidate_name,
            frame_index=frame_index,
            overall_score=float(np.clip(overall, 0.0, 1.0)),
            region_scores=region_scores,
            signals={
                "mean_alpha": float(np.clip(alpha, 0.0, 1.0).mean()),
                "soft_pixel_ratio": float(((alpha > 0.05) & (alpha < 0.95)).mean()),
            },
        )


def _validate_ownership_shapes(alpha: np.ndarray, ownership: RegionOwnership) -> None:
    alpha_shape = tuple(alpha.shape)
    for field in REGION_FIELDS:
        mask = getattr(ownership, field)
        if tuple(mask.shape) != alpha_shape:
            raise ValueError(
                f"ownership mask shape for {field} is {mask.shape}; expected alpha shape {alpha_shape}"
            )


def _mean_alpha_score(alpha: np.ndarray, mask: np.ndarray, *, target: str) -> float:
    if not np.any(mask):
        return 0.5
    mean_alpha = float(np.clip(alpha[mask], 0.0, 1.0).mean())
    if target == "high":
        return mean_alpha
    return 1.0 - mean_alpha


def _soft_alpha_score(alpha: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.5
    values = np.clip(alpha[mask], 0.0, 1.0)
    soft = (1.0 - np.abs(values - 0.5) * 2.0).mean()
    collapsed = (values <= 0.05).mean()
    return float(np.clip(soft * 0.8 + (1.0 - collapsed) * 0.2, 0.0, 1.0))
