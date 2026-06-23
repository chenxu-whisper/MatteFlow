"""Quality-gated matte fusion decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..analysis.region_ownership import RegionOwnership


@dataclass(frozen=True)
class FusionCandidate:
    """A candidate alpha matte with per-pixel confidence evidence."""

    name: str
    alpha: np.ndarray
    confidence: np.ndarray | None = None


@dataclass(frozen=True)
class FusionResult:
    """Fused alpha matte and compact decision diagnostics."""

    alpha: np.ndarray
    diagnostics: dict


class FusionQualityGate:
    """Select candidate alpha values by region ownership and quality evidence."""

    REGION_ORDER = (
        "background_residue",
        "luminous_prop",
        "transparent_effect",
        "hair_edge",
        "subject",
        "uncertain_edge",
    )

    PROTECTED_LOW_ALPHA_REGIONS = {"background_residue"}
    PROTECTED_HIGH_ALPHA_REGIONS = {"luminous_prop"}

    def fuse(
        self,
        candidates: Iterable[FusionCandidate],
        ownership: RegionOwnership,
    ) -> FusionResult:
        candidates_list = list(candidates)
        if not candidates_list:
            raise ValueError("FusionQualityGate requires at least one candidate")

        shape = self._ownership_shape(ownership)
        normalized = [self._normalize_candidate(candidate, shape) for candidate in candidates_list]
        fallback = normalized[0]
        result = fallback["alpha"].copy()
        selected_by_region = {}
        rejected_takeovers = {region: 0 for region in self.REGION_ORDER}

        for region in self.REGION_ORDER:
            mask = np.asarray(getattr(ownership, region), dtype=bool)
            if not np.any(mask):
                selected_by_region[region] = None
                continue

            if region in self.PROTECTED_LOW_ALPHA_REGIONS:
                selected = self._select_low_alpha_candidate(normalized, mask)
                rejected_takeovers[region] = self._count_higher_alpha_takeovers(normalized, selected, mask)
            elif region in self.PROTECTED_HIGH_ALPHA_REGIONS:
                selected = self._select_high_alpha_candidate(normalized, mask)
                rejected_takeovers[region] = self._count_lower_alpha_takeovers(normalized, selected, mask)
            else:
                selected = self._select_confident_candidate(normalized, mask, region)

            result[mask] = selected["alpha"][mask]
            selected_by_region[region] = selected["name"]

        diagnostics = {
            "candidate_count": len(normalized),
            "selected_by_region": selected_by_region,
            "rejected_takeovers": rejected_takeovers,
        }
        return FusionResult(alpha=np.clip(result, 0.0, 1.0).astype(np.float32, copy=False), diagnostics=diagnostics)

    @staticmethod
    def _ownership_shape(ownership: RegionOwnership) -> tuple[int, int]:
        shape = tuple(np.asarray(ownership.subject).shape)
        for region in FusionQualityGate.REGION_ORDER:
            region_shape = tuple(np.asarray(getattr(ownership, region)).shape)
            if region_shape != shape:
                raise ValueError(f"{region} shape {region_shape} does not match subject shape {shape}")
        return shape

    @staticmethod
    def _normalize_candidate(candidate: FusionCandidate, shape: tuple[int, int]) -> dict:
        alpha = np.clip(np.asarray(candidate.alpha, dtype=np.float32), 0.0, 1.0)
        if alpha.shape != shape:
            raise ValueError(f"{candidate.name}.alpha shape {alpha.shape} does not match ownership shape {shape}")
        if candidate.confidence is None:
            confidence = np.ones(shape, dtype=np.float32)
        else:
            confidence = np.clip(np.asarray(candidate.confidence, dtype=np.float32), 0.0, 1.0)
            if confidence.shape != shape:
                raise ValueError(
                    f"{candidate.name}.confidence shape {confidence.shape} does not match ownership shape {shape}"
                )
        return {"name": candidate.name, "alpha": alpha, "confidence": confidence}

    def _select_confident_candidate(self, candidates: list[dict], mask: np.ndarray, region: str) -> dict:
        scores = [self._region_score(candidate, mask, region) for candidate in candidates]
        return candidates[int(np.argmax(scores))]

    @staticmethod
    def _select_high_alpha_candidate(candidates: list[dict], mask: np.ndarray) -> dict:
        scores = []
        for candidate in candidates:
            alpha_score = float(candidate["alpha"][mask].mean())
            confidence_score = float(candidate["confidence"][mask].mean())
            scores.append(alpha_score * 0.75 + confidence_score * 0.25)
        return candidates[int(np.argmax(scores))]

    @staticmethod
    def _select_low_alpha_candidate(candidates: list[dict], mask: np.ndarray) -> dict:
        scores = []
        for candidate in candidates:
            alpha_score = 1.0 - float(candidate["alpha"][mask].mean())
            confidence_score = float(candidate["confidence"][mask].mean())
            scores.append(alpha_score * 0.75 + confidence_score * 0.25)
        return candidates[int(np.argmax(scores))]

    @staticmethod
    def _region_score(candidate: dict, mask: np.ndarray, region: str) -> float:
        confidence_score = float(candidate["confidence"][mask].mean())
        alpha_score = float(candidate["alpha"][mask].mean())
        if region == "transparent_effect":
            return confidence_score * 0.80 + alpha_score * 0.20
        if region == "subject":
            return confidence_score * 0.70 + alpha_score * 0.30
        return confidence_score

    @staticmethod
    def _count_lower_alpha_takeovers(candidates: list[dict], selected: dict, mask: np.ndarray) -> int:
        selected_alpha = selected["alpha"][mask]
        count = 0
        for candidate in candidates:
            if candidate is selected:
                continue
            lower_alpha = candidate["alpha"][mask] < selected_alpha - 0.08
            higher_confidence = candidate["confidence"][mask] > selected["confidence"][mask] + 0.02
            count += int(np.count_nonzero(lower_alpha & higher_confidence))
        return count

    @staticmethod
    def _count_higher_alpha_takeovers(candidates: list[dict], selected: dict, mask: np.ndarray) -> int:
        selected_alpha = selected["alpha"][mask]
        count = 0
        for candidate in candidates:
            if candidate is selected:
                continue
            higher_alpha = candidate["alpha"][mask] > selected_alpha + 0.08
            higher_confidence = candidate["confidence"][mask] > selected["confidence"][mask] + 0.02
            count += int(np.count_nonzero(higher_alpha & higher_confidence))
        return count
