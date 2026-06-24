"""P0 classified quality risks for matting outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

P0_RISK_ORDER = (
    "hair_edge_loss",
    "background_residue",
    "light_subject_loss",
    "transparent_effect_loss",
    "temporal_instability",
    "subject_misidentification",
)


@dataclass(frozen=True)
class P0Risk:
    """One classified P0 quality risk."""

    score: float
    level: str
    signals: dict[str, float | int | str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": _round_float(self.score),
            "level": self.level,
            "signals": _json_safe_signals(self.signals),
        }


@dataclass(frozen=True)
class P0QualityReport:
    """A-F P0 risk report."""

    risks: dict[str, P0Risk]

    def to_dict(self) -> dict[str, Any]:
        return {name: self.risks[name].to_dict() for name in P0_RISK_ORDER}


class P0QualityAnalyzer:
    """Compute classified P0 quality risks from existing pipeline evidence."""

    def analyze_sequence(
        self,
        frames: Sequence[np.ndarray],
        alphas: Sequence[np.ndarray],
        *,
        quality_report: Any | None = None,
        region_context: Mapping[str, Any] | None = None,
    ) -> P0QualityReport:
        if not alphas:
            return self._empty_report()

        alpha_arrays = [_as_alpha(alpha) for alpha in alphas]
        ownerships = list((region_context or {}).get("region_ownership", []) or [])
        risks = {
            "hair_edge_loss": self._hair_edge_loss(alpha_arrays, quality_report, ownerships),
            "background_residue": self._background_residue(alpha_arrays, quality_report, ownerships),
            "light_subject_loss": self._light_subject_loss(frames, alpha_arrays, ownerships),
            "transparent_effect_loss": self._transparent_effect_loss(alpha_arrays, ownerships),
            "temporal_instability": self._temporal_instability(alpha_arrays, quality_report),
            "subject_misidentification": self._subject_misidentification(alpha_arrays, ownerships),
        }
        return P0QualityReport(risks=risks)

    @staticmethod
    def _empty_report() -> P0QualityReport:
        return P0QualityReport(
            risks={
                name: P0Risk(score=0.0, level="pass", signals={})
                for name in P0_RISK_ORDER
            }
        )

    def _hair_edge_loss(
        self,
        alphas: Sequence[np.ndarray],
        quality_report: Any | None,
        ownerships: Sequence[Any],
    ) -> P0Risk:
        edge_uncertainty = _float_attr(quality_report, "mean_edge_uncertainty")
        hole_pixels = _float_attr(quality_report, "hole_pixels")
        total_pixels = max(sum(alpha.size for alpha in alphas), 1)
        hole_ratio = hole_pixels / total_pixels
        hair_stats = self._masked_low_alpha_ratio_stats(alphas, ownerships, "hair_edge", threshold=0.08)
        score = max(
            edge_uncertainty / 0.12,
            hole_ratio / 0.01,
            hair_stats["max"] / 0.90,
            hair_stats["top3_mean"] / 0.50,
        )
        return _risk(
            score,
            {
                "quality_mean_edge_uncertainty": edge_uncertainty,
                "quality_hole_ratio": hole_ratio,
                "hair_low_alpha_ratio": hair_stats["mean"],
                "hair_low_alpha_ratio_max": hair_stats["max"],
                "hair_low_alpha_ratio_p95": hair_stats["p95"],
                "hair_low_alpha_ratio_top3_mean": hair_stats["top3_mean"],
            },
        )

    def _background_residue(
        self,
        alphas: Sequence[np.ndarray],
        quality_report: Any | None,
        ownerships: Sequence[Any],
    ) -> P0Risk:
        residue = _float_attr(quality_report, "background_residue")
        region_stats = self._masked_ratio_stats(alphas, ownerships, "background_residue")
        score = max(residue / 0.10, region_stats["max"] / 0.20, region_stats["top3_mean"] / 0.10)
        return _risk(
            score,
            {
                "quality_background_residue": residue,
                "region_background_residue_ratio": region_stats["mean"],
                "region_background_residue_ratio_max": region_stats["max"],
                "region_background_residue_ratio_p95": region_stats["p95"],
                "region_background_residue_ratio_top3_mean": region_stats["top3_mean"],
            },
        )

    def _light_subject_loss(
        self,
        frames: Sequence[np.ndarray],
        alphas: Sequence[np.ndarray],
        ownerships: Sequence[Any],
    ) -> P0Risk:
        ratios = []
        for index, alpha in enumerate(alphas):
            frame = np.asarray(frames[index]) if index < len(frames) else None
            if frame is None or frame.ndim != 3:
                continue
            light_mask = _light_low_saturation_mask(frame)
            subject_mask = _ownership_mask(ownerships, index, "subject", alpha.shape)
            candidate = light_mask & subject_mask
            ratios.append(_low_alpha_ratio(alpha, candidate, threshold=0.35))
        light_stats = _ratio_stats(ratios)
        return _risk(
            max(light_stats["max"] / 0.90, light_stats["top3_mean"] / 0.50),
            {
                "light_low_alpha_ratio": light_stats["mean"],
                "light_low_alpha_ratio_max": light_stats["max"],
                "light_low_alpha_ratio_p95": light_stats["p95"],
                "light_low_alpha_ratio_top3_mean": light_stats["top3_mean"],
            },
        )

    def _transparent_effect_loss(
        self,
        alphas: Sequence[np.ndarray],
        ownerships: Sequence[Any],
    ) -> P0Risk:
        ratios = []
        for index, alpha in enumerate(alphas):
            effect_mask = _ownership_mask(ownerships, index, "transparent_effect", alpha.shape)
            luminous_mask = _ownership_mask(ownerships, index, "luminous_prop", alpha.shape)
            ratios.append(_low_alpha_ratio(alpha, effect_mask | luminous_mask, threshold=0.08))
        effect_stats = _ratio_stats(ratios)
        return _risk(
            max(effect_stats["max"] / 0.90, effect_stats["top3_mean"] / 0.50),
            {
                "effect_low_alpha_ratio": effect_stats["mean"],
                "effect_low_alpha_ratio_max": effect_stats["max"],
                "effect_low_alpha_ratio_p95": effect_stats["p95"],
                "effect_low_alpha_ratio_top3_mean": effect_stats["top3_mean"],
            },
        )

    @staticmethod
    def _temporal_instability(
        alphas: Sequence[np.ndarray],
        quality_report: Any | None,
    ) -> P0Risk:
        quality_flicker = _float_attr(quality_report, "temporal_flicker")
        if len(alphas) <= 1:
            sequence_flicker = 0.0
        else:
            sequence_flicker = float(np.mean([np.abs(curr - prev).mean() for prev, curr in zip(alphas, alphas[1:])]))
        score = max(quality_flicker, sequence_flicker) / 0.20
        return _risk(
            score,
            {
                "quality_temporal_flicker": quality_flicker,
                "sequence_temporal_flicker": sequence_flicker,
            },
        )

    def _subject_misidentification(
        self,
        alphas: Sequence[np.ndarray],
        ownerships: Sequence[Any],
    ) -> P0Risk:
        subject_stats = self._masked_ratio_stats(alphas, ownerships, "subject")
        subject_ratio = subject_stats["mean"]
        solid_alpha_ratio = _mean([float((alpha >= 0.70).mean()) for alpha in alphas])
        if subject_ratio == 0.0 and solid_alpha_ratio == 0.0:
            score = 0.0
        elif subject_stats["min"] < 0.03:
            score = 0.60
        else:
            score = 0.0
        return _risk(
            score,
            {
                "subject_coverage_ratio": subject_ratio,
                "subject_coverage_ratio_min": subject_stats["min"],
                "subject_coverage_ratio_p95": subject_stats["p95"],
                "subject_coverage_ratio_top3_mean": subject_stats["top3_mean"],
                "solid_alpha_ratio": solid_alpha_ratio,
            },
        )

    @staticmethod
    def _masked_ratio_stats(
        alphas: Sequence[np.ndarray],
        ownerships: Sequence[Any],
        field: str,
    ) -> dict[str, float]:
        ratios = []
        for index, alpha in enumerate(alphas):
            mask = _ownership_mask(ownerships, index, field, alpha.shape)
            ratios.append(float(mask.mean()) if mask.size else 0.0)
        return _ratio_stats(ratios)

    @staticmethod
    def _masked_low_alpha_ratio_stats(
        alphas: Sequence[np.ndarray],
        ownerships: Sequence[Any],
        field: str,
        *,
        threshold: float,
    ) -> dict[str, float]:
        ratios = []
        for index, alpha in enumerate(alphas):
            mask = _ownership_mask(ownerships, index, field, alpha.shape)
            ratios.append(_low_alpha_ratio(alpha, mask, threshold=threshold))
        return _ratio_stats(ratios)


def _risk(score: float, signals: dict[str, float | int | str]) -> P0Risk:
    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.70:
        level = "fail"
    elif score >= 0.35:
        level = "warn"
    else:
        level = "pass"
    return P0Risk(score=score, level=level, signals=signals)


def _as_alpha(alpha: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)


def _float_attr(obj: Any | None, name: str) -> float:
    if obj is None:
        return 0.0
    value = getattr(obj, name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ownership_mask(
    ownerships: Sequence[Any],
    index: int,
    field: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if index >= len(ownerships):
        return np.zeros(shape, dtype=bool)
    mask = getattr(ownerships[index], field, None)
    if mask is None:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        return np.zeros(shape, dtype=bool)
    return mask


def _light_low_saturation_mask(frame: np.ndarray) -> np.ndarray:
    frame_f = frame.astype(np.float32, copy=False)
    max_channel = frame_f.max(axis=2)
    min_channel = frame_f.min(axis=2)
    brightness = frame_f.mean(axis=2)
    chroma = max_channel - min_channel
    return (brightness >= 180.0) & (chroma <= 35.0)


def _low_alpha_ratio(alpha: np.ndarray, mask: np.ndarray, *, threshold: float) -> float:
    if not np.any(mask):
        return 0.0
    return float((alpha[mask] <= threshold).mean())


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def _ratio_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "p95": 0.0, "min": 0.0, "top3_mean": 0.0}
    values_f = np.asarray(values, dtype=np.float32)
    top_count = min(3, values_f.size)
    top_values = np.sort(values_f)[-top_count:]
    return {
        "mean": float(values_f.mean()),
        "max": float(values_f.max()),
        "p95": float(np.percentile(values_f, 95)),
        "min": float(values_f.min()),
        "top3_mean": float(top_values.mean()) if top_values.size else 0.0,
    }


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _json_safe_signals(signals: Mapping[str, float | int | str]) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {}
    for key, value in signals.items():
        if isinstance(value, str):
            result[str(key)] = value
        elif isinstance(value, (int, np.integer)):
            result[str(key)] = int(value)
        else:
            result[str(key)] = _round_float(float(value))
    return result
