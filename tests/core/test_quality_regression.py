import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.evaluation import (  # noqa: E402
    QualityRegressionBaseline,
    QualityRegressionEvaluator,
    QualityRegressionThresholds,
)


def _processing_report(sample_name: str, overall_score: float, hole_pixels: int = 0) -> dict:
    return {
        "job": {"input_path": f"{sample_name}.png"},
        "quality": {
            "overall_score": overall_score,
            "mean_edge_uncertainty": 0.04,
            "hole_pixels": hole_pixels,
            "background_residue": 0.003,
            "temporal_flicker": 0.01,
        },
        "p0_risks": {},
    }


def test_quality_regression_evaluator_fails_on_threshold_violation(tmp_path):
    report_path = tmp_path / "processing_report.json"
    report_path.write_text(
        json.dumps(_processing_report("low_score", overall_score=0.7)),
        encoding="utf-8",
    )

    result = QualityRegressionEvaluator(
        thresholds=QualityRegressionThresholds(min_overall_score=0.8)
    ).evaluate_root(tmp_path)

    assert result.passed is False
    assert result.failed_count == 1
    assert "overall_score 0.700 below minimum 0.800" in result.samples[0].failures


def test_quality_regression_evaluator_fails_on_baseline_score_drop(tmp_path):
    report_path = tmp_path / "processing_report.json"
    report_path.write_text(
        json.dumps(_processing_report("sample", overall_score=0.9)),
        encoding="utf-8",
    )
    baseline = QualityRegressionBaseline.from_dict({"sample": {"overall_score": 0.98}})

    result = QualityRegressionEvaluator(
        thresholds=QualityRegressionThresholds(max_score_drop=0.05),
        baseline=baseline,
    ).evaluate_root(tmp_path)

    assert result.passed is False
    assert result.samples[0].baseline_metrics["overall_score"] == 0.98
    assert "overall_score dropped 0.080 from baseline 0.980" in result.samples[0].failures
