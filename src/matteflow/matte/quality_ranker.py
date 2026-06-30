"""Explainable ranking for region-level matte candidate selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

EDGE_GUARDED_REGIONS = frozenset({"hair_edge", "uncertain_edge"})
EDGE_REGION_SCORE_MARGIN = 0.03
EDGE_OVERALL_SCORE_DROP_TOLERANCE = 0.05
CRITICAL_REGION_RISK_WEIGHT = 0.01


@dataclass(frozen=True)
class RankingDecision:
    """Best candidate decision with explainable ranking diagnostics."""

    region: str
    candidate_name: str
    ranking_score: float
    factors: Mapping[str, float]
    reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "candidate_name": self.candidate_name,
            "ranking_score": float(self.ranking_score),
            "factors": dict(self.factors),
            "reasons": list(self.reasons),
            "diagnostics": dict(self.diagnostics),
        }


class QualityRanker:
    """Rank candidate quality for a single region using explainable factors."""

    def choose_best(
        self,
        *,
        region: str,
        candidate_quality: Mapping[str, Any],
        critical_regions: set[str] | None = None,
        candidate_risks: Mapping[str, Mapping[str, float]] | None = None,
    ) -> RankingDecision:
        if not candidate_quality:
            raise ValueError("candidate_quality must not be empty")

        critical_regions = critical_regions or set()
        candidate_risks = candidate_risks or {}
        baseline_name = next(iter(candidate_quality))
        baseline = self._score_candidate(
            baseline_name,
            candidate_quality[baseline_name],
            region=region,
            critical=region in critical_regions,
            risks=candidate_risks.get(baseline_name, {}),
        )
        diagnostics = {baseline_name: baseline}
        best_name = baseline_name
        best = baseline
        reasons: list[str] = []

        for candidate_name, quality in list(candidate_quality.items())[1:]:
            scored = self._score_candidate(
                candidate_name,
                quality,
                region=region,
                critical=region in critical_regions,
                risks=candidate_risks.get(candidate_name, {}),
            )
            diagnostics[candidate_name] = scored
            if self._is_better(region=region, candidate=scored, best=best):
                best_name = candidate_name
                best = scored
            elif region in EDGE_GUARDED_REGIONS:
                reasons.append("edge_margin_not_met")

        return RankingDecision(
            region=region,
            candidate_name=best_name,
            ranking_score=float(best["ranking_score"]),
            factors=best["factors"],
            reasons=tuple(dict.fromkeys(reasons)),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _score_candidate(
        candidate_name: str,
        quality: Any,
        *,
        region: str,
        critical: bool,
        risks: Mapping[str, float],
    ) -> dict[str, Any]:
        overall_score = float(getattr(quality, "overall_score", 0.0))
        region_scores = getattr(quality, "region_scores", {}) or {}
        region_score = float(region_scores.get(region, overall_score))
        risk_penalty = 0.0
        if critical:
            risk_penalty = sum(float(value) for value in risks.values()) * CRITICAL_REGION_RISK_WEIGHT
        ranking_score = region_score + overall_score * 0.10 - risk_penalty
        return {
            "candidate_name": candidate_name,
            "ranking_score": float(ranking_score),
            "overall_score": overall_score,
            "region_score": region_score,
            "factors": {
                "region_score": region_score,
                "overall_score": overall_score,
                "risk_penalty": risk_penalty,
            },
        }

    @staticmethod
    def _is_better(*, region: str, candidate: Mapping[str, Any], best: Mapping[str, Any]) -> bool:
        if region not in EDGE_GUARDED_REGIONS:
            return float(candidate["ranking_score"]) > float(best["ranking_score"])
        has_clear_region_gain = (
            float(candidate["region_score"]) >= float(best["region_score"]) + EDGE_REGION_SCORE_MARGIN
        )
        overall_is_not_regressed = (
            float(candidate["overall_score"])
            >= float(best["overall_score"]) - EDGE_OVERALL_SCORE_DROP_TOLERANCE
        )
        return (
            float(candidate["ranking_score"]) > float(best["ranking_score"])
            and has_clear_region_gain
            and overall_is_not_regressed
        )
