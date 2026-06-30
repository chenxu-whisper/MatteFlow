"""Quality regression evaluation for processing reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

REPORT_FILENAME = "processing_report.json"


@dataclass(frozen=True)
class QualityRegressionThresholds:
    """Quality gates used when comparing generated processing reports."""

    min_overall_score: float = 0.80
    max_mean_edge_uncertainty: float = 0.08
    max_hole_pixels: int = 100
    max_background_residue: float = 0.02
    max_temporal_flicker: float = 0.08
    max_edge_temporal_flicker: float = 0.08
    max_transparent_temporal_flicker: float = 0.08
    max_max_frame_delta: float = 0.12
    max_hair_low_alpha_ratio: float = 0.30
    max_effect_low_alpha_ratio: float = 0.30
    max_score_drop: float = 0.05
    max_p0_risk_score_increase: float = 0.20


@dataclass(frozen=True)
class QualityRegressionBaseline:
    """Baseline metrics keyed by stable sample name."""

    samples: Mapping[str, Mapping[str, float]]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "QualityRegressionBaseline":
        if not payload:
            return cls(samples={})
        raw_samples = payload.get("samples", payload)
        if not isinstance(raw_samples, Mapping):
            return cls(samples={})
        samples: dict[str, dict[str, float]] = {}
        for sample_name, metrics in raw_samples.items():
            if not isinstance(metrics, Mapping):
                continue
            samples[str(sample_name)] = {
                str(metric_name): float(metric_value)
                for metric_name, metric_value in metrics.items()
                if _is_number(metric_value)
            }
        return cls(samples=samples)

    @classmethod
    def from_path(cls, baseline_path: Path | str | None) -> "QualityRegressionBaseline":
        if baseline_path is None:
            return cls(samples={})
        payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return cls(samples={})
        return cls.from_dict(payload)

    def metrics_for(self, sample_name: str) -> Mapping[str, float]:
        return self.samples.get(sample_name, {})


@dataclass(frozen=True)
class QualityRegressionSampleResult:
    """Evaluation result for one processing report."""

    sample_name: str
    report_path: Path
    metrics: Mapping[str, Any]
    baseline_metrics: Mapping[str, Any] = field(default_factory=dict)
    failures: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "report_path": str(self.report_path),
            "status": "pass" if self.passed else "fail",
            "metrics": dict(self.metrics),
            "baseline_metrics": dict(self.baseline_metrics),
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class QualityRegressionRun:
    """Aggregate quality regression result."""

    samples: tuple[QualityRegressionSampleResult, ...]
    thresholds: QualityRegressionThresholds

    @property
    def total_count(self) -> int:
        return len(self.samples)

    @property
    def failed_count(self) -> int:
        return sum(1 for sample in self.samples if not sample.passed)

    @property
    def passed_count(self) -> int:
        return self.total_count - self.failed_count

    @property
    def passed(self) -> bool:
        return self.failed_count == 0

    @property
    def samples_by_name(self) -> dict[str, QualityRegressionSampleResult]:
        return {sample.sample_name: sample for sample in self.samples}

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "status": "pass" if self.passed else "fail",
                "total_count": self.total_count,
                "passed_count": self.passed_count,
                "failed_count": self.failed_count,
            },
            "thresholds": {
                "min_overall_score": self.thresholds.min_overall_score,
                "max_mean_edge_uncertainty": self.thresholds.max_mean_edge_uncertainty,
                "max_hole_pixels": self.thresholds.max_hole_pixels,
                "max_background_residue": self.thresholds.max_background_residue,
                "max_temporal_flicker": self.thresholds.max_temporal_flicker,
                "max_edge_temporal_flicker": self.thresholds.max_edge_temporal_flicker,
                "max_transparent_temporal_flicker": self.thresholds.max_transparent_temporal_flicker,
                "max_max_frame_delta": self.thresholds.max_max_frame_delta,
                "max_hair_low_alpha_ratio": self.thresholds.max_hair_low_alpha_ratio,
                "max_effect_low_alpha_ratio": self.thresholds.max_effect_low_alpha_ratio,
                "max_score_drop": self.thresholds.max_score_drop,
                "max_p0_risk_score_increase": self.thresholds.max_p0_risk_score_increase,
            },
            "samples": [sample.to_dict() for sample in self.samples],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Quality Regression Report",
            "",
            f"Status: {'PASS' if self.passed else 'FAIL'}",
            f"Samples: {self.total_count}",
            f"Passed: {self.passed_count}",
            f"Failed: {self.failed_count}",
            "",
            "| Sample | Status | Overall | Holes | Failures |",
            "| --- | --- | ---: | ---: | --- |",
        ]
        for sample in self.samples:
            failures = "<br>".join(sample.failures) if sample.failures else ""
            lines.append(
                "| {sample} | {status} | {overall:.3f} | {holes} | {failures} |".format(
                    sample=sample.sample_name,
                    status="PASS" if sample.passed else "FAIL",
                    overall=float(sample.metrics.get("overall_score") or 0.0),
                    holes=int(sample.metrics.get("hole_pixels") or 0),
                    failures=failures,
                )
            )
        return "\n".join(lines) + "\n"


class QualityRegressionEvaluator:
    """Evaluate P3 processing reports against thresholds and optional baseline."""

    def __init__(
        self,
        thresholds: QualityRegressionThresholds | None = None,
        baseline: QualityRegressionBaseline | None = None,
    ) -> None:
        self.thresholds = thresholds or QualityRegressionThresholds()
        self.baseline = baseline or QualityRegressionBaseline(samples={})

    @staticmethod
    def discover_reports(root: Path | str) -> list[Path]:
        root = Path(root)
        if root.is_file():
            return [root]
        return sorted(root.rglob(REPORT_FILENAME))

    def evaluate_paths(self, report_paths: Iterable[Path | str]) -> QualityRegressionRun:
        samples = tuple(self._evaluate_path(Path(path)) for path in report_paths)
        return QualityRegressionRun(samples=samples, thresholds=self.thresholds)

    def evaluate_root(self, reports_root: Path | str) -> QualityRegressionRun:
        reports_root = Path(reports_root)
        report_paths = self.discover_reports(reports_root)
        if not report_paths:
            return QualityRegressionRun(
                samples=(
                    self._failure_sample(
                        reports_root / REPORT_FILENAME,
                        "no processing_report.json files found",
                    ),
                ),
                thresholds=self.thresholds,
            )
        return self.evaluate_paths(report_paths)

    def _evaluate_path(self, report_path: Path) -> QualityRegressionSampleResult:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return self._failure_sample(report_path, f"invalid processing report: {exc}")
        if not isinstance(payload, Mapping):
            return self._failure_sample(report_path, "invalid processing report: top-level JSON must be an object")
        sample_name = _sample_name(report_path, payload)
        try:
            metrics = _extract_metrics(payload)
            baseline_metrics = dict(self.baseline.metrics_for(sample_name))
            failures = self._build_failures(metrics, baseline_metrics)
        except (TypeError, ValueError) as exc:
            return self._failure_sample(report_path, f"invalid processing report metrics: {exc}")
        return QualityRegressionSampleResult(
            sample_name=sample_name,
            report_path=report_path,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            failures=tuple(failures),
        )

    def _failure_sample(self, report_path: Path, failure: str) -> QualityRegressionSampleResult:
        sample_name = _sample_name(report_path, {})
        metrics = _extract_metrics({})
        baseline_metrics = dict(self.baseline.metrics_for(sample_name))
        return QualityRegressionSampleResult(
            sample_name=sample_name,
            report_path=report_path,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            failures=(failure,),
        )

    def _build_failures(
        self,
        metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> list[str]:
        thresholds = self.thresholds
        failures: list[str] = []
        overall_score = _float_metric(metrics, "overall_score")
        if overall_score < thresholds.min_overall_score:
            failures.append(
                f"overall_score {overall_score:.3f} below minimum {thresholds.min_overall_score:.3f}"
            )
        mean_edge = _float_metric(metrics, "mean_edge_uncertainty")
        if mean_edge > thresholds.max_mean_edge_uncertainty:
            failures.append(
                "mean_edge_uncertainty "
                f"{mean_edge:.3f} above maximum {thresholds.max_mean_edge_uncertainty:.3f}"
            )
        hole_pixels = _int_metric(metrics, "hole_pixels")
        if hole_pixels > thresholds.max_hole_pixels:
            failures.append(f"hole_pixels {hole_pixels} above maximum {thresholds.max_hole_pixels}")
        background_residue = _float_metric(metrics, "background_residue")
        if background_residue > thresholds.max_background_residue:
            failures.append(
                "background_residue "
                f"{background_residue:.3f} above maximum {thresholds.max_background_residue:.3f}"
            )
        temporal_flicker = _float_metric(metrics, "temporal_flicker")
        if temporal_flicker > thresholds.max_temporal_flicker:
            failures.append(
                f"temporal_flicker {temporal_flicker:.3f} above maximum {thresholds.max_temporal_flicker:.3f}"
            )
        edge_temporal_flicker = _float_metric(metrics, "edge_temporal_flicker")
        if edge_temporal_flicker > thresholds.max_edge_temporal_flicker:
            failures.append(
                "edge_temporal_flicker "
                f"{edge_temporal_flicker:.3f} above maximum {thresholds.max_edge_temporal_flicker:.3f}"
            )
        transparent_temporal_flicker = _float_metric(metrics, "transparent_temporal_flicker")
        if transparent_temporal_flicker > thresholds.max_transparent_temporal_flicker:
            failures.append(
                "transparent_temporal_flicker "
                f"{transparent_temporal_flicker:.3f} above maximum "
                f"{thresholds.max_transparent_temporal_flicker:.3f}"
            )
        max_frame_delta = _float_metric(metrics, "max_frame_delta")
        if max_frame_delta > thresholds.max_max_frame_delta:
            failures.append(
                f"max_frame_delta {max_frame_delta:.3f} above maximum {thresholds.max_max_frame_delta:.3f}"
            )
        hair_low_alpha_ratio = _float_metric(
            metrics,
            "p0_risk.hair_edge_loss.signal.hair_low_alpha_ratio",
        )
        if hair_low_alpha_ratio > thresholds.max_hair_low_alpha_ratio:
            failures.append(
                "hair_low_alpha_ratio "
                f"{hair_low_alpha_ratio:.3f} above maximum {thresholds.max_hair_low_alpha_ratio:.3f}"
            )
        effect_low_alpha_ratio = _float_metric(
            metrics,
            "p0_risk.transparent_effect_loss.signal.effect_low_alpha_ratio",
        )
        if effect_low_alpha_ratio > thresholds.max_effect_low_alpha_ratio:
            failures.append(
                "effect_low_alpha_ratio "
                f"{effect_low_alpha_ratio:.3f} above maximum {thresholds.max_effect_low_alpha_ratio:.3f}"
            )
        if (
            metrics.get("quality_selection.available") is True
            and _int_metric(metrics, "quality_selection.candidate_count") <= 0
        ):
            failures.append("quality selection enabled but no candidates available")

        baseline_score = baseline_metrics.get("overall_score")
        if _is_number(baseline_score):
            score_drop = float(baseline_score) - overall_score
            if score_drop > thresholds.max_score_drop:
                failures.append(
                    "overall_score dropped "
                    f"{score_drop:.3f} from baseline {float(baseline_score):.3f}"
                )
        failures.extend(self._build_p0_failures(metrics, baseline_metrics))
        return failures

    def _build_p0_failures(
        self,
        metrics: Mapping[str, Any],
        baseline_metrics: Mapping[str, Any],
    ) -> list[str]:
        failures: list[str] = []
        for key, value in metrics.items():
            if not key.startswith("p0_risk.") or not key.endswith(".level"):
                continue
            risk_name = key[len("p0_risk.") : -len(".level")]
            if str(value) == "fail":
                failures.append(f"p0 {risk_name} level fail")

        for key, value in metrics.items():
            if not key.startswith("p0_risk.") or not key.endswith(".score"):
                continue
            baseline_value = baseline_metrics.get(key)
            if not _is_number(baseline_value):
                continue
            increase = float(value) - float(baseline_value)
            if increase > self.thresholds.max_p0_risk_score_increase:
                risk_name = key[len("p0_risk.") : -len(".score")]
                failures.append(
                    "p0 {risk_name} risk increased {increase:.3f} from baseline {baseline:.3f}".format(
                        risk_name=risk_name,
                        increase=increase,
                        baseline=float(baseline_value),
                    )
                )
        return failures


def _sample_name(report_path: Path, payload: Mapping[str, Any]) -> str:
    job = payload.get("job")
    if isinstance(job, Mapping):
        input_path = job.get("input_path")
        if input_path:
            return Path(str(input_path)).stem
    if report_path.name == REPORT_FILENAME and report_path.parent.name:
        return report_path.parent.name
    return report_path.stem


def _extract_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    quality = payload.get("quality")
    if not isinstance(quality, Mapping):
        quality = {}
    metrics: dict[str, Any] = {
        "overall_score": _float_metric(quality, "overall_score"),
        "mean_edge_uncertainty": _float_metric(quality, "mean_edge_uncertainty"),
        "speckle_pixels": _int_metric(quality, "speckle_pixels"),
        "hole_pixels": _int_metric(quality, "hole_pixels"),
        "background_residue": _float_metric(quality, "background_residue"),
        "temporal_flicker": _float_metric(quality, "temporal_flicker"),
        "edge_temporal_flicker": _float_metric(quality, "edge_temporal_flicker"),
        "transparent_temporal_flicker": _float_metric(quality, "transparent_temporal_flicker"),
        "max_frame_delta": _float_metric(quality, "max_frame_delta"),
    }
    p0_risks = payload.get("p0_risks")
    if isinstance(p0_risks, Mapping):
        for risk_name, risk_payload in p0_risks.items():
            if not isinstance(risk_payload, Mapping):
                continue
            metric_prefix = f"p0_risk.{risk_name}"
            metrics[f"{metric_prefix}.score"] = _float_metric(risk_payload, "score")
            metrics[f"{metric_prefix}.level"] = str(risk_payload.get("level", "pass"))
            signals = risk_payload.get("signals")
            if isinstance(signals, Mapping):
                for signal_name, signal_value in signals.items():
                    if _is_number(signal_value):
                        metrics[f"{metric_prefix}.signal.{signal_name}"] = float(signal_value)
    quality_selection = payload.get("quality_selection")
    if isinstance(quality_selection, Mapping):
        metrics["quality_selection.available"] = bool(quality_selection.get("available", False))
        metrics["quality_selection.candidate_count"] = _int_metric(
            quality_selection,
            "candidate_count",
        )
    return metrics


def _float_metric(metrics: Mapping[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_metric(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key)
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
