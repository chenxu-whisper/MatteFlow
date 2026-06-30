"""Structured processing quality reports for MatteFlow jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import BackgroundMode, MattingConfig

REPORT_SCHEMA_VERSION = 2
REPORT_FILENAME = "processing_report.json"


@dataclass(frozen=True)
class ProcessingReport:
    """JSON-serializable processing report payload."""

    schema_version: int
    job: dict[str, Any]
    timings: dict[str, float]
    quality: dict[str, Any]
    p0_risks: dict[str, Any]
    regions: dict[str, int]
    region_supervision: dict[str, Any]
    model_decisions: dict[str, Any]
    fusion: dict[str, Any]
    quality_selection: dict[str, Any]
    edge_reconstruction: dict[str, Any]
    black_effect_enhancement: dict[str, Any]
    foreground_recovery: dict[str, Any]
    artifacts: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable report schema as JSON-serializable Python values."""
        return {
            "schema_version": int(self.schema_version),
            "job": _json_safe(self.job),
            "timings": _json_safe(self.timings),
            "quality": _json_safe(self.quality),
            "p0_risks": _json_safe(self.p0_risks),
            "regions": _json_safe(self.regions),
            "region_supervision": _json_safe(self.region_supervision),
            "model_decisions": _json_safe(self.model_decisions),
            "fusion": _json_safe(self.fusion),
            "quality_selection": _json_safe(self.quality_selection),
            "edge_reconstruction": _json_safe(self.edge_reconstruction),
            "black_effect_enhancement": _json_safe(self.black_effect_enhancement),
            "foreground_recovery": _json_safe(self.foreground_recovery),
            "artifacts": _json_safe(self.artifacts),
            "warnings": _json_safe(self.warnings),
        }


class ProcessingReportBuilder:
    """Assemble a processing report from pipeline stage evidence."""

    REGION_FIELDS = (
        "subject",
        "hair_edge",
        "luminous_prop",
        "transparent_effect",
        "background_residue",
        "uncertain_edge",
    )

    def build(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        config: MattingConfig,
        frame_count: int,
        background_mode_effective: BackgroundMode,
        timings: Mapping[str, float] | None,
        quality_report: Any | None,
        p0_quality_report: Any | None = None,
        region_context: Mapping[str, Any] | None = None,
        hybrid_matte: Any | None = None,
        decontaminate_context: Mapping[str, Any] | None = None,
        edge_reconstruction: Mapping[str, Any] | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> ProcessingReport:
        output_dir = Path(output_dir)
        decontaminate_context = decontaminate_context or {}
        hybrid_active_model = getattr(hybrid_matte, "last_active_ai_model", None)

        return ProcessingReport(
            schema_version=REPORT_SCHEMA_VERSION,
            job={
                "input_path": _relative_path(input_path, output_dir),
                "output_dir": str(output_dir),
                "frame_count": int(frame_count),
                "background_mode_requested": _enum_value(config.background_mode),
                "background_mode_effective": _enum_value(background_mode_effective),
                "quality_mode": _enum_value(config.quality_mode),
                "ai_model_requested": getattr(config, "ai_model", "auto"),
                "ai_model_active": hybrid_active_model,
            },
            timings=self._build_timings(timings),
            quality=self._build_quality(quality_report, frame_count),
            p0_risks=self._build_p0_risks(p0_quality_report),
            regions=self._build_regions(region_context),
            region_supervision=self._build_region_supervision(region_context),
            model_decisions=self._build_model_decisions(hybrid_matte),
            fusion=self._build_fusion(decontaminate_context),
            quality_selection=self._build_quality_selection(hybrid_matte),
            edge_reconstruction=_json_safe(dict(edge_reconstruction or {})),
            black_effect_enhancement=self._build_black_effect_enhancement(hybrid_matte),
            foreground_recovery=self._build_foreground_recovery(decontaminate_context),
            artifacts=self._build_artifacts(artifacts or {}, output_dir),
            warnings=[],
        )

    @staticmethod
    def _build_timings(timings: Mapping[str, float] | None) -> dict[str, float]:
        if not timings:
            return {}
        return {str(key): float(value) for key, value in timings.items()}

    @staticmethod
    def _build_quality(quality_report: Any | None, frame_count: int) -> dict[str, Any]:
        if quality_report is None:
            return {
                "frame_count": int(frame_count),
                "overall_score": None,
                "mean_edge_uncertainty": None,
                "speckle_pixels": 0,
                "hole_pixels": 0,
                "background_residue": None,
                "temporal_flicker": None,
                "edge_temporal_flicker": None,
                "transparent_temporal_flicker": None,
                "max_frame_delta": None,
            }

        return {
            "frame_count": int(getattr(quality_report, "frame_count", frame_count)),
            "overall_score": _optional_float(getattr(quality_report, "overall_score", None)),
            "mean_edge_uncertainty": _optional_float(
                getattr(quality_report, "mean_edge_uncertainty", None)
            ),
            "speckle_pixels": int(getattr(quality_report, "speckle_pixels", 0)),
            "hole_pixels": int(getattr(quality_report, "hole_pixels", 0)),
            "background_residue": _optional_float(
                getattr(quality_report, "background_residue", None)
            ),
            "temporal_flicker": _optional_float(getattr(quality_report, "temporal_flicker", None)),
            "edge_temporal_flicker": _optional_float(
                getattr(quality_report, "edge_temporal_flicker", None)
            ),
            "transparent_temporal_flicker": _optional_float(
                getattr(quality_report, "transparent_temporal_flicker", None)
            ),
            "max_frame_delta": _optional_float(getattr(quality_report, "max_frame_delta", None)),
        }

    @staticmethod
    def _build_p0_risks(p0_quality_report: Any | None) -> dict[str, Any]:
        if p0_quality_report is not None and hasattr(p0_quality_report, "to_dict"):
            return p0_quality_report.to_dict()

        from ..analysis.p0_quality import P0QualityAnalyzer

        return P0QualityAnalyzer().analyze_sequence([], []).to_dict()

    def _build_regions(self, region_context: Mapping[str, Any] | None) -> dict[str, int]:
        counts = {f"{field}_pixels": 0 for field in self.REGION_FIELDS}
        ownerships = [] if not region_context else region_context.get("region_ownership", [])
        for ownership in ownerships or []:
            for field in self.REGION_FIELDS:
                counts[f"{field}_pixels"] += int(np.count_nonzero(getattr(ownership, field)))
        return counts

    @staticmethod
    def _build_region_supervision(region_context: Mapping[str, Any] | None) -> dict[str, Any]:
        ownerships = [] if not region_context else region_context.get("region_ownership", [])
        total_pixels = sum(int(ownership.subject.size) for ownership in ownerships or [])
        region_pixels = {
            field: sum(int(np.count_nonzero(getattr(ownership, field))) for ownership in ownerships or [])
            for field in ProcessingReportBuilder.REGION_FIELDS
        }
        denominator = max(total_pixels, 1)
        region_ratios = {
            field: round(count / denominator, 6) for field, count in region_pixels.items()
        }
        expectations = region_context.get("region_expectations") if region_context else None
        failures = _region_expectation_failures(expectations, region_pixels, region_ratios)
        return {
            "frame_count": len(ownerships or []),
            "total_pixels": total_pixels,
            "region_pixels": region_pixels,
            "region_ratios": region_ratios,
            "failures": failures,
        }

    @staticmethod
    def _build_model_decisions(hybrid_matte: Any | None) -> dict[str, Any]:
        if hybrid_matte is None:
            active_model = None
            fallback_metrics = {}
            fusion_quality_gate = {}
            has_green_debug = False
        else:
            active_model = getattr(hybrid_matte, "last_active_ai_model", None)
            fallback_metrics = getattr(hybrid_matte, "last_fallback_quality_metrics", None) or {}
            fusion_quality_gate = getattr(hybrid_matte, "last_fusion_quality_gate_diagnostics", None) or {}
            has_green_debug = bool(getattr(hybrid_matte, "green_screen_layer_debug", None))

        return {
            "active_ai_model": active_model,
            "fallback_quality_metrics": _json_safe(fallback_metrics),
            "fusion_quality_gate": _json_safe(fusion_quality_gate),
            "green_screen_layer_debug_available": has_green_debug,
        }

    @staticmethod
    def _build_fusion(decontaminate_context: Mapping[str, Any]) -> dict[str, Any]:
        fusion = decontaminate_context.get("fusion")
        if not fusion:
            return {
                "available": False,
                "selected_by_region": {},
                "rejected_takeovers": {},
            }
        return {
            "available": bool(fusion.get("available", True)),
            "selected_by_region": _json_safe(fusion.get("selected_by_region", {})),
            "rejected_takeovers": _json_safe(fusion.get("rejected_takeovers", {})),
        }

    @staticmethod
    def _build_quality_selection(hybrid_matte: Any | None) -> dict[str, Any]:
        selection = (
            getattr(hybrid_matte, "last_quality_selection", None)
            if hybrid_matte is not None
            else None
        )
        if not selection:
            return {
                "available": False,
                "candidate_count": 0,
                "selected_model_counts": {},
                "candidate_quality": {},
                "skipped_candidates": [],
            }
        return _json_safe(dict(selection))

    @staticmethod
    def _build_foreground_recovery(decontaminate_context: Mapping[str, Any]) -> dict[str, Any]:
        recovery = decontaminate_context.get("foreground_recovery")
        if not recovery:
            return {}
        return _json_safe(dict(recovery))

    @staticmethod
    def _build_black_effect_enhancement(hybrid_matte: Any | None) -> dict[str, Any]:
        black_matte = getattr(hybrid_matte, "black_matte", None) if hybrid_matte is not None else None
        history = getattr(black_matte, "effect_enhancement_history", None)
        if history:
            return _aggregate_black_effect_history(history)
        diagnostics = getattr(black_matte, "last_effect_enhancement", None)
        if not diagnostics:
            return {}
        return _json_safe(dict(diagnostics))

    @staticmethod
    def _build_artifacts(artifacts: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
        result = {}
        for key, value in artifacts.items():
            if value is None:
                continue
            if isinstance(value, Path):
                result[str(key)] = _relative_path(value, output_dir)
            else:
                result[str(key)] = _json_safe(value)
        return result


class ProcessingReportWriter:
    """Write processing reports to disk."""

    def write(self, report: ProcessingReport, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / REPORT_FILENAME
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report_path


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _relative_path(path: Path | str, output_dir: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return round(value, 6)
    return value


def _region_expectation_failures(
    expectations: Any,
    region_pixels: Mapping[str, int],
    region_ratios: Mapping[str, float],
) -> list[str]:
    if not isinstance(expectations, Mapping):
        return []
    failures: list[str] = []
    required = expectations.get("required_regions", ()) or ()
    for region in required:
        region_name = str(region)
        if region_name not in region_pixels:
            raise ValueError(f"Unsupported region: {region_name}")
        if region_pixels[region_name] <= 0:
            failures.append(f"required_region {region_name} missing")
    min_ratios = expectations.get("min_region_ratios", {}) or {}
    for region, minimum in dict(min_ratios).items():
        region_name = str(region)
        if region_name not in region_ratios:
            raise ValueError(f"Unsupported region: {region_name}")
        minimum_f = float(minimum)
        if region_ratios[region_name] < minimum_f:
            failures.append(
                f"region_ratio {region_name} {region_ratios[region_name]:.6f} "
                f"below minimum {minimum_f:.6f}"
            )
    return failures


def _aggregate_black_effect_history(history: Any) -> dict[str, Any]:
    if not isinstance(history, (list, tuple)):
        return {}
    items = [dict(item) for item in history if isinstance(item, Mapping)]
    if not items:
        return {}
    count_fields = (
        "smoke_pixels",
        "glow_pixels",
        "particle_pixels",
        "subject_edge_pixels",
        "black_residue_suppressed_pixels",
    )
    summary: dict[str, Any] = {"frames": len(items)}
    for field in count_fields:
        summary[field] = int(sum(int(item.get(field, 0)) for item in items))
    summary["mean_alpha_delta"] = float(
        np.mean([float(item.get("mean_alpha_delta", 0.0)) for item in items])
    )
    return _json_safe(summary)
