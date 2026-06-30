"""Manifest-based matting quality regression runner."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from ..config import BackgroundMode, MattingConfig, QualityMode
from ..pipeline import MattingPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MattingQualityRegressionSample:
    name: str
    input_path: Path
    background_mode: str
    quality_mode: str
    candidate_models: tuple[str, ...]
    risk_ceilings: dict[str, float]
    expected_alpha_path: Path | None = None
    alpha_mae_ceiling: float | None = None
    alpha_mse_ceiling: float | None = None
    required_temporal_models: tuple[str, ...] = ()
    region_expectations: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MattingQualityRegressionManifest:
    samples: tuple[MattingQualityRegressionSample, ...]

    @classmethod
    def from_path(cls, path: Path | str) -> "MattingQualityRegressionManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("manifest top-level JSON must be an object")
        raw_samples = payload.get("samples", [])
        if not isinstance(raw_samples, list):
            raise ValueError("manifest samples must be a list")

        samples = []
        for item in raw_samples:
            if not isinstance(item, Mapping):
                raise ValueError("manifest sample must be an object")
            samples.append(
                MattingQualityRegressionSample(
                    name=str(item["name"]),
                    input_path=Path(str(item["input_path"])),
                    background_mode=str(item["background_mode"]),
                    quality_mode=str(item.get("quality_mode", "standard")),
                    candidate_models=tuple(
                        str(model) for model in item.get("candidate_models", ())
                    ),
                    risk_ceilings={
                        str(key): float(value)
                        for key, value in dict(item.get("risk_ceilings", {})).items()
                    },
                    expected_alpha_path=(
                        Path(str(item["expected_alpha_path"]))
                        if item.get("expected_alpha_path") is not None
                        else None
                    ),
                    alpha_mae_ceiling=(
                        float(item["alpha_mae_ceiling"])
                        if item.get("alpha_mae_ceiling") is not None
                        else None
                    ),
                    alpha_mse_ceiling=(
                        float(item["alpha_mse_ceiling"])
                        if item.get("alpha_mse_ceiling") is not None
                        else None
                    ),
                    required_temporal_models=tuple(
                        str(model) for model in item.get("required_temporal_models", ())
                    ),
                    region_expectations=dict(item.get("region_expectations", {})),
                )
            )
        return cls(samples=tuple(samples))


class MattingQualityRegressionRunner:
    def __init__(
        self,
        *,
        manifest: MattingQualityRegressionManifest,
        pipeline_factory: Callable[[MattingConfig], Any] = MattingPipeline,
    ):
        self.manifest = manifest
        self.pipeline_factory = pipeline_factory

    def run(self, output_root: Path | str) -> Path:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        sample_results = []

        for sample in self.manifest.samples:
            sample_output_dir = output_root / sample.name
            config = self._build_config(sample)
            try:
                result = self.pipeline_factory(config).process(sample.input_path, sample_output_dir)
                alpha_supervision = self._evaluate_alpha_supervision(sample, sample_output_dir)
                failures = alpha_supervision.pop("failures", [])
                failures.extend(
                    self._evaluate_required_temporal_models(
                        sample,
                        result.get("processing_report_path"),
                    )
                )
                failures.extend(
                    self._evaluate_region_supervision(
                        sample,
                        result.get("processing_report_path"),
                    )
                )
                sample_results.append(
                    {
                        "name": sample.name,
                        "status": "failed" if failures else "completed",
                        "input_path": str(sample.input_path),
                        "output_dir": str(sample_output_dir),
                        "processing_report_path": result.get("processing_report_path"),
                        "candidate_models": list(sample.candidate_models),
                        "risk_ceilings": dict(sample.risk_ceilings),
                        "required_temporal_models": list(sample.required_temporal_models),
                        "region_expectations": dict(sample.region_expectations),
                        "alpha_supervision": alpha_supervision,
                        "failures": failures,
                    }
                )
            except Exception as exc:
                sample_results.append(
                    {
                        "name": sample.name,
                        "status": "failed",
                        "input_path": str(sample.input_path),
                        "output_dir": str(sample_output_dir),
                        "error": str(exc),
                        "candidate_models": list(sample.candidate_models),
                        "risk_ceilings": dict(sample.risk_ceilings),
                        "required_temporal_models": list(sample.required_temporal_models),
                        "region_expectations": dict(sample.region_expectations),
                        "alpha_supervision": {},
                        "failures": [str(exc)],
                    }
                )

        payload = {
            "summary": {
                "sample_count": len(sample_results),
                "completed_count": sum(1 for item in sample_results if item["status"] == "completed"),
                "failed_count": sum(1 for item in sample_results if item["status"] == "failed"),
            },
            "samples": sample_results,
        }
        summary_path = output_root / "quality_summary.json"
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary_path

    @staticmethod
    def _build_config(sample: MattingQualityRegressionSample) -> MattingConfig:
        config = MattingConfig()
        config.background_mode = _background_mode(sample.background_mode)
        config.quality_mode = _quality_mode(sample.quality_mode)
        config.quality_selection_enable = True
        config.quality_candidate_models = sample.candidate_models or ("traditional",)
        config.region_expectations = dict(sample.region_expectations)
        return config

    @staticmethod
    def _evaluate_region_supervision(
        sample: MattingQualityRegressionSample,
        processing_report_path: str | None,
    ) -> list[str]:
        if not sample.region_expectations:
            return []
        if processing_report_path is None:
            return ["region_supervision report missing"]

        payload = json.loads(Path(processing_report_path).read_text(encoding="utf-8"))
        supervision = payload.get("region_supervision")
        if not isinstance(supervision, Mapping):
            return ["region_supervision report missing"]
        failures = supervision.get("failures")
        if not isinstance(failures, list):
            return []
        return [str(failure) for failure in failures]

    @staticmethod
    def _evaluate_required_temporal_models(
        sample: MattingQualityRegressionSample,
        processing_report_path: str | None,
    ) -> list[str]:
        if not sample.required_temporal_models:
            return []
        if processing_report_path is None:
            return [
                f"required_temporal_model {model} did not contribute"
                for model in sample.required_temporal_models
            ]

        payload = json.loads(Path(processing_report_path).read_text(encoding="utf-8"))
        quality_selection = payload.get("quality_selection")
        if not isinstance(quality_selection, Mapping):
            quality_selection = {}
        candidate_quality = quality_selection.get("candidate_quality")
        if not isinstance(candidate_quality, Mapping):
            candidate_quality = {}
        selected_counts = quality_selection.get("selected_model_counts")
        if not isinstance(selected_counts, Mapping):
            selected_counts = {}
        contributed = {
            str(name)
            for name, value in selected_counts.items()
            if _is_positive_number(value)
        }

        failures = []
        for model in sample.required_temporal_models:
            if model not in contributed:
                failures.append(f"required_temporal_model {model} did not contribute")
        missing = [model for model in sample.required_temporal_models if model not in contributed]
        logger.info(
            "Temporal model contribution check: sample=%s required=%s contributed=%s missing=%s",
            sample.name,
            list(sample.required_temporal_models),
            sorted(contributed),
            missing,
        )
        return failures

    @staticmethod
    def _evaluate_alpha_supervision(
        sample: MattingQualityRegressionSample,
        sample_output_dir: Path,
    ) -> dict[str, Any]:
        if sample.expected_alpha_path is None:
            return {}

        expected = _read_alpha(sample.expected_alpha_path)
        predicted_path = _first_predicted_matte(sample_output_dir)
        predicted = _read_alpha(predicted_path)
        if predicted.shape != expected.shape:
            predicted = cv2.resize(
                predicted,
                (expected.shape[1], expected.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        diff = predicted - expected
        mae = float(np.abs(diff).mean())
        mse = float(np.square(diff).mean())
        failures = []
        if sample.alpha_mae_ceiling is not None and mae > sample.alpha_mae_ceiling:
            failures.append(
                f"alpha_mae {mae:.6f} above ceiling {sample.alpha_mae_ceiling:.6f}"
            )
        if sample.alpha_mse_ceiling is not None and mse > sample.alpha_mse_ceiling:
            failures.append(
                f"alpha_mse {mse:.6f} above ceiling {sample.alpha_mse_ceiling:.6f}"
            )
        logger.info(
            "Alpha supervision check: sample=%s mae=%.6f mae_ceiling=%.6f mse=%.6f "
            "mse_ceiling=%.6f status=%s",
            sample.name,
            mae,
            sample.alpha_mae_ceiling if sample.alpha_mae_ceiling is not None else float("nan"),
            mse,
            sample.alpha_mse_ceiling if sample.alpha_mse_ceiling is not None else float("nan"),
            "failed alpha_mae" if any("alpha_mae" in failure for failure in failures) else "passed",
        )
        return {
            "expected_alpha_path": str(sample.expected_alpha_path),
            "predicted_alpha_path": str(predicted_path),
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "alpha_mae_ceiling": sample.alpha_mae_ceiling,
            "alpha_mse_ceiling": sample.alpha_mse_ceiling,
            "failures": failures,
        }


def _background_mode(value: str) -> BackgroundMode:
    normalized = str(value)
    for mode in BackgroundMode:
        if normalized in {mode.value, mode.name.lower()}:
            return mode
    raise ValueError(f"Unsupported background mode: {value}")


def _quality_mode(value: str) -> QualityMode:
    normalized = str(value)
    for mode in QualityMode:
        if normalized in {mode.value, mode.name.lower()}:
            return mode
    raise ValueError(f"Unsupported quality mode: {value}")


def _read_alpha(path: Path) -> np.ndarray:
    alpha = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if alpha is None:
        raise ValueError(f"Cannot read alpha image: {path}")
    if alpha.ndim == 3:
        alpha = alpha[..., 0]
    alpha_f = alpha.astype(np.float32, copy=False)
    if alpha_f.max(initial=0.0) > 1.0:
        alpha_f = alpha_f / 255.0
    return np.clip(alpha_f, 0.0, 1.0)


def _first_predicted_matte(sample_output_dir: Path) -> Path:
    matte_dir = sample_output_dir / "Matte"
    candidates = sorted(matte_dir.glob("*.png"))
    if not candidates:
        raise ValueError(f"No predicted matte frames found in {matte_dir}")
    return candidates[0]


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
