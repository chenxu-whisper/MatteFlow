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
            "edge_temporal_flicker": 0.02,
            "transparent_temporal_flicker": 0.03,
            "max_frame_delta": 0.04,
        },
        "p0_risks": {
            "hair_edge_loss": {
                "score": 0.1,
                "level": "pass",
                "signals": {
                    "hair_region_pixels": 12,
                    "hair_low_alpha_ratio": 0.02,
                    "hair_soft_alpha_ratio": 0.75,
                },
            },
            "transparent_effect_loss": {
                "score": 0.1,
                "level": "pass",
                "signals": {
                    "effect_region_pixels": 8,
                    "effect_low_alpha_ratio": 0.03,
                    "effect_soft_alpha_ratio": 0.65,
                },
            },
        },
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


def test_quality_regression_evaluator_extracts_temporal_detail_metrics(tmp_path):
    report_path = tmp_path / "processing_report.json"
    report_path.write_text(
        json.dumps(_processing_report("sample", overall_score=0.9)),
        encoding="utf-8",
    )

    result = QualityRegressionEvaluator().evaluate_root(tmp_path)

    metrics = result.samples[0].metrics
    assert metrics["edge_temporal_flicker"] == 0.02
    assert metrics["transparent_temporal_flicker"] == 0.03
    assert metrics["max_frame_delta"] == 0.04


def test_quality_regression_evaluator_extracts_p0_detail_signals(tmp_path):
    report_path = tmp_path / "processing_report.json"
    report_path.write_text(
        json.dumps(_processing_report("sample", overall_score=0.9)),
        encoding="utf-8",
    )

    result = QualityRegressionEvaluator().evaluate_root(tmp_path)

    metrics = result.samples[0].metrics
    assert metrics["p0_risk.hair_edge_loss.signal.hair_low_alpha_ratio"] == 0.02
    assert metrics["p0_risk.hair_edge_loss.signal.hair_soft_alpha_ratio"] == 0.75
    assert metrics["p0_risk.transparent_effect_loss.signal.effect_low_alpha_ratio"] == 0.03
    assert metrics["p0_risk.transparent_effect_loss.signal.effect_soft_alpha_ratio"] == 0.65


def test_quality_regression_evaluator_fails_on_detail_thresholds(tmp_path):
    payload = _processing_report("detail_fail", overall_score=0.9)
    payload["quality"]["edge_temporal_flicker"] = 0.20
    payload["quality"]["transparent_temporal_flicker"] = 0.18
    payload["p0_risks"]["hair_edge_loss"]["signals"]["hair_low_alpha_ratio"] = 0.40
    payload["p0_risks"]["transparent_effect_loss"]["signals"]["effect_low_alpha_ratio"] = 0.35
    report_path = tmp_path / "processing_report.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")

    result = QualityRegressionEvaluator(
        thresholds=QualityRegressionThresholds(
            max_edge_temporal_flicker=0.10,
            max_transparent_temporal_flicker=0.10,
            max_hair_low_alpha_ratio=0.20,
            max_effect_low_alpha_ratio=0.20,
        )
    ).evaluate_root(tmp_path)

    failures = result.samples[0].failures
    assert result.passed is False
    assert "edge_temporal_flicker 0.200 above maximum 0.100" in failures
    assert "transparent_temporal_flicker 0.180 above maximum 0.100" in failures
    assert "hair_low_alpha_ratio 0.400 above maximum 0.200" in failures
    assert "effect_low_alpha_ratio 0.350 above maximum 0.200" in failures
