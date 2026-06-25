"""Region-level matte candidate selector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..analysis.alpha_quality import AlphaQualityAnalyzer, AlphaQualityReport
from ..analysis.region_ownership import RegionOwnership
from ..evaluation.matte_quality import CandidateQualityReport, REGION_FIELDS
from .candidates.types import MatteCandidateSequence

EDGE_GUARDED_REGIONS = frozenset({"hair_edge", "uncertain_edge"})
EDGE_REGION_SCORE_MARGIN = 0.03
EDGE_OVERALL_SCORE_DROP_TOLERANCE = 0.05
COMBINATION_HOLE_PIXEL_TOLERANCE = 5
COMBINATION_OVERALL_SCORE_DROP_TOLERANCE = 0.0003
COMBINATION_EDGE_UNCERTAINTY_TOLERANCE = 0.001
COMBINATION_GUARD_MIN_PIXELS = 25


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
    guarded_frames: list[dict[str, Any]]

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
            "guarded_frames": list(self.guarded_frames),
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
                guarded_frames=[],
            )

        frame_count = len(candidates[0].alphas)
        for candidate in candidates:
            if len(candidate.alphas) != frame_count:
                raise ValueError("candidate frame counts do not match")
        if len(ownerships) != frame_count:
            raise ValueError("ownership count does not match candidate frame count")

        quality_by_frame = quality_report.by_frame_candidate()
        alpha_analyzer = AlphaQualityAnalyzer()
        selected_alphas: list[np.ndarray] = []
        decisions: list[SelectionDecision] = []
        guarded_frames: list[dict[str, Any]] = []

        for frame_index in range(frame_count):
            result_alpha = candidates[0].alphas[frame_index].copy()
            ownership = ownerships[frame_index]
            assigned = np.zeros(result_alpha.shape, dtype=bool)
            frame_decisions: list[SelectionDecision] = []

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
                frame_decisions.append(
                    SelectionDecision(
                        frame_index=frame_index,
                        region=region,
                        candidate_name=candidate.name,
                        score=score,
                        pixel_count=int(np.count_nonzero(region_mask)),
                    )
                )

            selected_alpha = np.clip(result_alpha, 0.0, 1.0).astype(np.float32, copy=False)
            guard_reasons: list[str] = []
            fallback_candidate = candidates[0]
            fallback_report = self._analyze_alpha(alpha_analyzer, candidates[0].alphas[frame_index])
            selected_report = self._analyze_alpha(alpha_analyzer, selected_alpha)
            if selected_alpha.size >= COMBINATION_GUARD_MIN_PIXELS:
                fallback_candidate, fallback_report = self._best_standalone_candidate(
                    candidates,
                    frame_index=frame_index,
                    alpha_analyzer=alpha_analyzer,
                )
                guard_reasons = self._guard_reasons(selected_report, fallback_report)
                subject_fallback = self._subject_edge_takeover_fallback(
                    candidates=candidates,
                    decisions=frame_decisions,
                    frame_index=frame_index,
                    alpha_analyzer=alpha_analyzer,
                )
                if subject_fallback is not None:
                    fallback_candidate, fallback_report = subject_fallback
                    if "edge_takeover" not in guard_reasons:
                        guard_reasons.append("edge_takeover")
                    guard_reasons.extend(
                        reason
                        for reason in self._guard_reasons(selected_report, fallback_report)
                        if reason not in guard_reasons
                    )
            if guard_reasons:
                selected_alpha = fallback_candidate.alphas[frame_index].copy()
                frame_decisions = self._fallback_decisions(
                    ownership=ownership,
                    frame_index=frame_index,
                    candidate_name=fallback_candidate.name,
                    score=fallback_report.overall_score,
                )
                guarded_frames.append(
                    self._guard_diagnostics(
                        frame_index=frame_index,
                        fallback_candidate=fallback_candidate.name,
                        reasons=guard_reasons,
                        selected_report=selected_report,
                        fallback_report=fallback_report,
                    )
                )

            decisions.extend(frame_decisions)
            selected_alphas.append(np.clip(selected_alpha, 0.0, 1.0).astype(np.float32, copy=False))

        return QualitySelectionResult(
            alphas=selected_alphas,
            decisions=tuple(decisions),
            candidate_quality=quality_report.to_summary(),
            skipped_candidates=list(skipped_candidates or []),
            guarded_frames=guarded_frames,
        )

    @staticmethod
    def _best_candidate(
        candidates: Sequence[MatteCandidateSequence],
        quality_by_frame: dict[tuple[int, str], Any],
        *,
        frame_index: int,
        region: str,
    ) -> tuple[MatteCandidateSequence, float] | None:
        best: tuple[MatteCandidateSequence, float, float] | None = None
        for candidate in candidates:
            quality = quality_by_frame.get((frame_index, candidate.name))
            if quality is None:
                continue
            score = float(quality.region_scores.get(region, quality.overall_score))
            overall_score = float(quality.overall_score)
            if best is None:
                best = (candidate, score, overall_score)
            elif QualitySelector._is_better_candidate(
                region=region,
                score=score,
                overall_score=overall_score,
                best_score=best[1],
                best_overall_score=best[2],
            ):
                best = (candidate, score, overall_score)
        if best is None:
            return None
        return best[0], best[1]

    @staticmethod
    def _is_better_candidate(
        *,
        region: str,
        score: float,
        overall_score: float,
        best_score: float,
        best_overall_score: float,
    ) -> bool:
        if region not in EDGE_GUARDED_REGIONS:
            return score > best_score
        has_clear_region_gain = score >= best_score + EDGE_REGION_SCORE_MARGIN
        overall_is_not_regressed = (
            overall_score >= best_overall_score - EDGE_OVERALL_SCORE_DROP_TOLERANCE
        )
        return has_clear_region_gain and overall_is_not_regressed

    @staticmethod
    def _best_standalone_candidate(
        candidates: Sequence[MatteCandidateSequence],
        *,
        frame_index: int,
        alpha_analyzer: AlphaQualityAnalyzer,
    ) -> tuple[MatteCandidateSequence, AlphaQualityReport]:
        best_candidate = candidates[0]
        best_report = QualitySelector._analyze_alpha(alpha_analyzer, candidates[0].alphas[frame_index])
        for candidate in candidates[1:]:
            report = QualitySelector._analyze_alpha(alpha_analyzer, candidate.alphas[frame_index])
            if QualitySelector._is_better_standalone_report(report, best_report):
                best_candidate = candidate
                best_report = report
        return best_candidate, best_report

    @staticmethod
    def _is_better_standalone_report(
        report: AlphaQualityReport,
        best_report: AlphaQualityReport,
    ) -> bool:
        if report.hole_pixels != best_report.hole_pixels:
            return report.hole_pixels < best_report.hole_pixels
        if report.speckle_pixels != best_report.speckle_pixels:
            return report.speckle_pixels < best_report.speckle_pixels
        if report.overall_score != best_report.overall_score:
            return report.overall_score > best_report.overall_score
        return report.mean_edge_uncertainty < best_report.mean_edge_uncertainty

    @staticmethod
    def _analyze_alpha(
        alpha_analyzer: AlphaQualityAnalyzer,
        alpha: np.ndarray,
    ) -> AlphaQualityReport:
        frame = np.zeros((*alpha.shape, 3), dtype=np.uint8)
        return alpha_analyzer.analyze_sequence([frame], [alpha])

    @staticmethod
    def _guard_reasons(
        selected_report: AlphaQualityReport,
        fallback_report: AlphaQualityReport,
    ) -> list[str]:
        reasons: list[str] = []
        if selected_report.hole_pixels > fallback_report.hole_pixels + COMBINATION_HOLE_PIXEL_TOLERANCE:
            reasons.append("hole_pixels")
        if (
            fallback_report.overall_score - selected_report.overall_score
            > COMBINATION_OVERALL_SCORE_DROP_TOLERANCE
        ):
            reasons.append("overall_score")
        if (
            selected_report.mean_edge_uncertainty - fallback_report.mean_edge_uncertainty
            > COMBINATION_EDGE_UNCERTAINTY_TOLERANCE
        ):
            reasons.append("mean_edge_uncertainty")
        return reasons

    @staticmethod
    def _subject_edge_takeover_fallback(
        *,
        candidates: Sequence[MatteCandidateSequence],
        decisions: Sequence[SelectionDecision],
        frame_index: int,
        alpha_analyzer: AlphaQualityAnalyzer,
    ) -> tuple[MatteCandidateSequence, AlphaQualityReport] | None:
        subject_candidates = [decision.candidate_name for decision in decisions if decision.region == "subject"]
        if not subject_candidates:
            return None
        subject_candidate_name = subject_candidates[0]
        has_edge_takeover = any(
            decision.region in EDGE_GUARDED_REGIONS
            and decision.candidate_name != subject_candidate_name
            and decision.pixel_count > 0
            for decision in decisions
        )
        if not has_edge_takeover:
            return None
        by_name = {candidate.name: candidate for candidate in candidates}
        candidate = by_name.get(subject_candidate_name)
        if candidate is None:
            return None
        report = QualitySelector._analyze_alpha(alpha_analyzer, candidate.alphas[frame_index])
        return candidate, report

    @staticmethod
    def _fallback_decisions(
        *,
        ownership: RegionOwnership,
        frame_index: int,
        candidate_name: str,
        score: float,
    ) -> list[SelectionDecision]:
        decisions = []
        assigned = np.zeros(ownership.subject.shape, dtype=bool)
        for region in REGION_FIELDS:
            mask = getattr(ownership, region)
            region_mask = mask & (~assigned)
            if not np.any(region_mask):
                continue
            assigned |= region_mask
            decisions.append(
                SelectionDecision(
                    frame_index=frame_index,
                    region=region,
                    candidate_name=candidate_name,
                    score=score,
                    pixel_count=int(np.count_nonzero(region_mask)),
                )
            )
        return decisions

    @staticmethod
    def _guard_diagnostics(
        *,
        frame_index: int,
        fallback_candidate: str,
        reasons: Sequence[str],
        selected_report: AlphaQualityReport,
        fallback_report: AlphaQualityReport,
    ) -> dict[str, Any]:
        return {
            "frame_index": int(frame_index),
            "fallback_candidate": fallback_candidate,
            "reasons": list(reasons),
            "selected_hole_pixels": int(selected_report.hole_pixels),
            "fallback_hole_pixels": int(fallback_report.hole_pixels),
            "selected_overall_score": float(selected_report.overall_score),
            "fallback_overall_score": float(fallback_report.overall_score),
            "selected_mean_edge_uncertainty": float(selected_report.mean_edge_uncertainty),
            "fallback_mean_edge_uncertainty": float(fallback_report.mean_edge_uncertainty),
        }
