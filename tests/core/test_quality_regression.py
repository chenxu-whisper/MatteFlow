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


def _p0_risks(**overrides) -> dict:
    risks = {
        "hair_edge_loss": {"score": 0.0, "level": "pass", "signals": {}},
        "background_residue": {"score": 0.0, "level": "pass", "signals": {}},
        "light_subject_loss": {"score": 0.0, "level": "pass", "signals": {}},
        "transparent_effect_loss": {"score": 0.0, "level": "pass", "signals": {}},
        "temporal_instability": {"score": 0.0, "level": "pass", "signals": {}},
        "subject_misidentification": {"score": 0.0, "level": "pass", "signals": {}},
    }
    risks.update(overrides)
    return risks


def _processing_report(
    sample_name: str,
    overall_score: float,
    hole_pixels: int = 0,
    p0_risks: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "job": {
            "input_path": f"{sample_name}.png",
            "frame_count": 1,
            "background_mode_effective": "green_screen",
            "quality_mode": "standard",
            "ai_model_active": "gvm",
        },
        "quality": {
            "overall_score": overall_score,
            "mean_edge_uncertainty": 0.04,
            "speckle_pixels": 2,
            "hole_pixels": hole_pixels,
            "background_residue": 0.003,
            "temporal_flicker": 0.01,
        },
        "p0_risks": p0_risks or _p0_risks(),
        "regions": {
            "subject_pixels": 100,
            "hair_edge_pixels": 10,
            "luminous_prop_pixels": 3,
            "transparent_effect_pixels": 2,
            "background_residue_pixels": 0,
            "uncertain_edge_pixels": 4,
        },
        "model_decisions": {"active_ai_model": "gvm"},
        "fusion": {"available": False},
        "foreground_recovery": {},
        "artifacts": {},
        "warnings": [],
    }


def _write_report(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quality_regression_evaluator_flags_threshold_and_baseline_regressions(tmp_path):
    good_report = _write_report(
        tmp_path / "good" / "processing_report.json",
        _processing_report("good", overall_score=0.94, hole_pixels=1),
    )
    regressed_report = _write_report(
        tmp_path / "regressed" / "processing_report.json",
        _processing_report("regressed", overall_score=0.81, hole_pixels=25),
    )
    baseline = QualityRegressionBaseline.from_dict(
        {
            "samples": {
                "good": {"overall_score": 0.93},
                "regressed": {"overall_score": 0.90},
            }
        }
    )
    thresholds = QualityRegressionThresholds(
        min_overall_score=0.80,
        max_hole_pixels=10,
        max_score_drop=0.05,
    )

    result = QualityRegressionEvaluator(thresholds=thresholds, baseline=baseline).evaluate_paths(
        [good_report, regressed_report]
    )

    assert result.total_count == 2
    assert result.passed_count == 1
    assert result.failed_count == 1
    assert result.passed is False
    regressed = result.samples_by_name["regressed"]
    assert regressed.passed is False
    assert any("hole_pixels" in failure for failure in regressed.failures)
    assert any("overall_score dropped" in failure for failure in regressed.failures)
    payload = result.to_dict()
    assert payload["summary"]["failed_count"] == 1
    assert payload["samples"][1]["sample_name"] == "regressed"
    markdown = result.to_markdown()
    assert "Quality Regression Report" in markdown
    assert "regressed" in markdown
    assert "FAIL" in markdown


def test_quality_regression_evaluator_fails_p0_fail_levels_and_baseline_risk_increases(tmp_path):
    p0_failed_report = _write_report(
        tmp_path / "p0-failed" / "processing_report.json",
        _processing_report(
            "p0-failed",
            overall_score=0.94,
            p0_risks=_p0_risks(
                transparent_effect_loss={
                    "score": 0.82,
                    "level": "fail",
                    "signals": {"effect_low_alpha_ratio": 0.9},
                }
            ),
        ),
    )
    p0_regressed_report = _write_report(
        tmp_path / "p0-regressed" / "processing_report.json",
        _processing_report(
            "p0-regressed",
            overall_score=0.94,
            p0_risks=_p0_risks(
                hair_edge_loss={
                    "score": 0.55,
                    "level": "warn",
                    "signals": {"hair_low_alpha_ratio": 0.55},
                }
            ),
        ),
    )
    baseline = QualityRegressionBaseline.from_dict(
        {
            "samples": {
                "p0-regressed": {
                    "overall_score": 0.94,
                    "p0_risk.hair_edge_loss.score": 0.10,
                }
            }
        }
    )
    thresholds = QualityRegressionThresholds(max_p0_risk_score_increase=0.20)

    result = QualityRegressionEvaluator(thresholds=thresholds, baseline=baseline).evaluate_paths(
        [p0_failed_report, p0_regressed_report]
    )

    assert result.failed_count == 2
    p0_failed = result.samples_by_name["p0-failed"]
    assert any("p0 transparent_effect_loss level fail" in failure for failure in p0_failed.failures)
    p0_regressed = result.samples_by_name["p0-regressed"]
    assert any("p0 hair_edge_loss risk increased" in failure for failure in p0_regressed.failures)
    assert p0_regressed.metrics["p0_risk.hair_edge_loss.score"] == 0.55


def test_quality_regression_evaluator_discovers_processing_reports(tmp_path):
    report_path = _write_report(
        tmp_path / "sample-a" / "processing_report.json",
        _processing_report("sample-a", overall_score=0.91),
    )

    discovered = QualityRegressionEvaluator.discover_reports(tmp_path)

    assert discovered == [report_path]


def test_quality_regression_evaluator_isolates_invalid_report_failures(tmp_path):
    valid_report = _write_report(
        tmp_path / "valid" / "processing_report.json",
        _processing_report("valid", overall_score=0.91),
    )
    invalid_report = tmp_path / "broken" / "processing_report.json"
    invalid_report.parent.mkdir(parents=True, exist_ok=True)
    invalid_report.write_text("{not-json", encoding="utf-8")

    result = QualityRegressionEvaluator().evaluate_paths([valid_report, invalid_report])

    assert result.total_count == 2
    assert result.passed_count == 1
    assert result.failed_count == 1
    broken = result.samples_by_name["broken"]
    assert broken.passed is False
    assert any("invalid processing report" in failure for failure in broken.failures)


def test_quality_regression_evaluator_fails_empty_report_roots(tmp_path):
    result = QualityRegressionEvaluator().evaluate_root(tmp_path)

    assert result.total_count == 1
    assert result.failed_count == 1
    assert result.samples[0].sample_name == tmp_path.name
    assert any("no processing_report.json files found" in failure for failure in result.samples[0].failures)
