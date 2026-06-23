"""Structured processing quality reports for MatteFlow jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import BackgroundMode, MattingConfig

REPORT_SCHEMA_VERSION = 1
REPORT_FILENAME = "processing_report.json"


@dataclass(frozen=True)
class ProcessingReport:
    """JSON-serializable processing report payload."""

    schema_version: int
    job: dict[str, Any]
    timings: dict[str, float]
    quality: dict[str, Any]
    regions: dict[str, int]
    model_decisions: dict[str, Any]
    fusion: dict[str, Any]
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
            "regions": _json_safe(self.regions),
            "model_decisions": _json_safe(self.model_decisions),
            "fusion": _json_safe(self.fusion),
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
        region_context: Mapping[str, Any] | None,
        hybrid_matte: Any | None,
        decontaminate_context: Mapping[str, Any] | None,
        artifacts: Mapping[str, Any],
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
            regions=self._build_regions(region_context),
            model_decisions=self._build_model_decisions(hybrid_matte),
            fusion=self._build_fusion(decontaminate_context),
            foreground_recovery=self._build_foreground_recovery(decontaminate_context),
            artifacts=self._build_artifacts(artifacts, output_dir),
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
        }

    def _build_regions(self, region_context: Mapping[str, Any] | None) -> dict[str, int]:
        counts = {f"{field}_pixels": 0 for field in self.REGION_FIELDS}
        ownerships = [] if not region_context else region_context.get("region_ownership", [])
        for ownership in ownerships or []:
            for field in self.REGION_FIELDS:
                counts[f"{field}_pixels"] += int(np.count_nonzero(getattr(ownership, field)))
        return counts

    @staticmethod
    def _build_model_decisions(hybrid_matte: Any | None) -> dict[str, Any]:
        if hybrid_matte is None:
            active_model = None
            fallback_metrics = {}
            has_green_debug = False
        else:
            active_model = getattr(hybrid_matte, "last_active_ai_model", None)
            fallback_metrics = getattr(hybrid_matte, "last_fallback_quality_metrics", None) or {}
            has_green_debug = bool(getattr(hybrid_matte, "green_screen_layer_debug", None))

        return {
            "active_ai_model": active_model,
            "fallback_quality_metrics": _json_safe(fallback_metrics),
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
    def _build_foreground_recovery(decontaminate_context: Mapping[str, Any]) -> dict[str, Any]:
        recovery = decontaminate_context.get("foreground_recovery")
        if not recovery:
            return {}
        return _json_safe(dict(recovery))

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
