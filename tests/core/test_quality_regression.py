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


def test_quality_regression_evaluator_discovers_processing_reports(tmp_path):
    report_path = _write_report(
        tmp_path / "sample-a" / "processing_report.json",
        _processing_report("sample-a", overall_score=0.91),
    )

    discovered = QualityRegressionEvaluator.discover_reports(tmp_path)

    assert discovered == [report_path]
