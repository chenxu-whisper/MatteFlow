"""Manifest-based matting quality regression runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..config import BackgroundMode, MattingConfig, QualityMode
from ..pipeline import MattingPipeline


@dataclass(frozen=True)
class MattingQualityRegressionSample:
    name: str
    input_path: Path
    background_mode: str
    quality_mode: str
    candidate_models: tuple[str, ...]
    risk_ceilings: dict[str, float]


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
                sample_results.append(
                    {
                        "name": sample.name,
                        "status": "completed",
                        "input_path": str(sample.input_path),
                        "output_dir": str(sample_output_dir),
                        "processing_report_path": result.get("processing_report_path"),
                        "candidate_models": list(sample.candidate_models),
                        "risk_ceilings": dict(sample.risk_ceilings),
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
        return config


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
