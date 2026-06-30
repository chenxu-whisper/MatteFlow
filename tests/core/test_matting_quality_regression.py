import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.evaluation.matting_quality_regression import (  # noqa: E402
    MattingQualityRegressionManifest,
    MattingQualityRegressionRunner,
)


def test_manifest_loads_quality_regression_sample():
    manifest = MattingQualityRegressionManifest.from_path(
        Path("tests/fixtures/matting_quality/manifest.json")
    )

    samples_by_path = {sample.input_path.as_posix(): sample for sample in manifest.samples}
    assert len(manifest.samples) == 12
    assert "assets/frame/test_frame_1.png" in samples_by_path
    assert "assets/video/test_black_1.mp4" in samples_by_path
    assert "assets/video/test_green_1.mp4" in samples_by_path
    assert samples_by_path["assets/frame/test_frame_1.png"].name == "green_frame_smoke"
    assert samples_by_path["assets/frame/test_frame_1.png"].candidate_models == ("traditional",)
    assert samples_by_path["assets/frame/test_frame_1.png"].risk_ceilings["background_residue"] == 0.2
    assert samples_by_path["assets/video/test_black_1.mp4"].background_mode == "black_background"
    assert samples_by_path["assets/video/test_green_1.mp4"].background_mode == "green_screen"


def test_runner_writes_quality_summary_json(tmp_path):
    calls = []

    class FakePipeline:
        def __init__(self, config):
            calls.append(config)

        def process(self, input_path, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report_path = output_dir / "processing_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "job": {"input_path": str(input_path)},
                        "quality": {"overall_score": 0.95},
                        "quality_selection": {"available": True, "candidate_count": 1},
                    }
                ),
                encoding="utf-8",
            )
            return {"processing_report_path": str(report_path)}

    manifest = MattingQualityRegressionManifest.from_path(
        Path("tests/fixtures/matting_quality/manifest.json")
    )
    summary_path = MattingQualityRegressionRunner(
        manifest=manifest,
        pipeline_factory=FakePipeline,
    ).run(tmp_path)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["summary"]["sample_count"] == len(manifest.samples)
    assert payload["summary"]["completed_count"] == len(manifest.samples)
    assert {sample["status"] for sample in payload["samples"]} == {"completed"}
    assert calls[0].quality_selection_enable is True
    assert calls[0].quality_candidate_models == ("traditional",)


def test_manifest_loads_expected_alpha_supervision_fields(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "name": "gt_case",
                        "input_path": "input.png",
                        "expected_alpha_path": "expected_alpha.png",
                        "background_mode": "green_screen",
                        "quality_mode": "high",
                        "candidate_models": ["traditional"],
                        "alpha_mae_ceiling": 0.05,
                        "alpha_mse_ceiling": 0.01,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = MattingQualityRegressionManifest.from_path(manifest_path)

    sample = manifest.samples[0]
    assert sample.expected_alpha_path == Path("expected_alpha.png")
    assert sample.alpha_mae_ceiling == 0.05
    assert sample.alpha_mse_ceiling == 0.01


def test_manifest_loads_region_expectations(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "name": "region_case",
                        "input_path": "input.png",
                        "background_mode": "green_screen",
                        "candidate_models": ["traditional"],
                        "region_expectations": {
                            "required_regions": ["hair_edge"],
                            "min_region_ratios": {"hair_edge": 0.001},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    sample = MattingQualityRegressionManifest.from_path(manifest_path).samples[0]

    assert sample.region_expectations["required_regions"] == ["hair_edge"]
    assert sample.region_expectations["min_region_ratios"]["hair_edge"] == 0.001


def test_runner_fails_sample_when_region_supervision_report_has_failures(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "name": "region_fail",
                        "input_path": "input.png",
                        "background_mode": "green_screen",
                        "quality_mode": "high",
                        "candidate_models": ["traditional"],
                        "region_expectations": {"required_regions": ["hair_edge"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def process(self, input_path, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True)
            report_path = output_dir / "processing_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "quality": {"overall_score": 0.95},
                        "region_supervision": {
                            "failures": ["required_region hair_edge missing"]
                        },
                    }
                ),
                encoding="utf-8",
            )
            return {"processing_report_path": str(report_path)}

    manifest = MattingQualityRegressionManifest.from_path(manifest_path)
    summary_path = MattingQualityRegressionRunner(
        manifest=manifest,
        pipeline_factory=FakePipeline,
    ).run(tmp_path / "out")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    assert sample["status"] == "failed"
    assert sample["region_expectations"] == {"required_regions": ["hair_edge"]}
    assert "required_region hair_edge missing" in sample["failures"]


def test_runner_fails_sample_when_output_matte_exceeds_expected_alpha_ceiling(tmp_path, caplog):
    expected_alpha = tmp_path / "expected_alpha.png"
    cv2.imwrite(str(expected_alpha), np.full((4, 4), 255, dtype=np.uint8))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "name": "gt_fail",
                        "input_path": "input.png",
                        "expected_alpha_path": str(expected_alpha),
                        "background_mode": "green_screen",
                        "quality_mode": "high",
                        "candidate_models": ["traditional"],
                        "alpha_mae_ceiling": 0.05,
                        "alpha_mse_ceiling": 0.01,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def process(self, input_path, output_dir):
            output_dir = Path(output_dir)
            matte_dir = output_dir / "Matte"
            matte_dir.mkdir(parents=True)
            cv2.imwrite(str(matte_dir / "matte_000000.png"), np.zeros((4, 4), dtype=np.uint8))
            report_path = output_dir / "processing_report.json"
            report_path.write_text(json.dumps({"quality": {"overall_score": 0.95}}), encoding="utf-8")
            return {"processing_report_path": str(report_path)}

    manifest = MattingQualityRegressionManifest.from_path(manifest_path)
    with caplog.at_level(logging.INFO, logger="matteflow.evaluation.matting_quality_regression"):
        summary_path = MattingQualityRegressionRunner(
            manifest=manifest,
            pipeline_factory=FakePipeline,
        ).run(tmp_path / "out")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    assert sample["status"] == "failed"
    assert sample["alpha_supervision"]["mae"] == 1.0
    assert sample["alpha_supervision"]["mse"] == 1.0
    assert "alpha_mae 1.000000 above ceiling 0.050000" in sample["failures"]
    assert "Alpha supervision check: sample=gt_fail" in caplog.text
    assert "mae=1.000000 mae_ceiling=0.050000" in caplog.text
    assert "failed alpha_mae" in caplog.text


def test_runner_fails_when_required_temporal_model_did_not_contribute(tmp_path, caplog):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "name": "temporal_required",
                        "input_path": "sequence",
                        "background_mode": "unknown",
                        "quality_mode": "high",
                        "candidate_models": ["matanyone2", "traditional"],
                        "required_temporal_models": ["matanyone2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def process(self, input_path, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True)
            report_path = output_dir / "processing_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "quality": {"overall_score": 0.95},
                        "quality_selection": {
                            "available": True,
                            "candidate_count": 1,
                            "candidate_quality": {"traditional": {"overall_score": 0.9}},
                            "selected_model_counts": {"traditional": 1},
                            "skipped_candidates": [
                                {"name": "matanyone2", "reason": "model_unavailable"}
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            return {"processing_report_path": str(report_path)}

    manifest = MattingQualityRegressionManifest.from_path(manifest_path)
    with caplog.at_level(logging.INFO, logger="matteflow.evaluation.matting_quality_regression"):
        summary_path = MattingQualityRegressionRunner(
            manifest=manifest,
            pipeline_factory=FakePipeline,
        ).run(tmp_path / "out")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    assert sample["status"] == "failed"
    assert sample["required_temporal_models"] == ["matanyone2"]
    assert "required_temporal_model matanyone2 did not contribute" in sample["failures"]
    assert "Temporal model contribution check: sample=temporal_required" in caplog.text
    assert "required=['matanyone2']" in caplog.text
    assert "contributed=['traditional']" in caplog.text
    assert "missing=['matanyone2']" in caplog.text


def test_runner_fails_when_required_temporal_model_was_evaluated_but_not_selected(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "name": "temporal_evaluated_only",
                        "input_path": "sequence",
                        "background_mode": "unknown",
                        "quality_mode": "high",
                        "candidate_models": ["matanyone2", "traditional"],
                        "required_temporal_models": ["matanyone2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def process(self, input_path, output_dir):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True)
            report_path = output_dir / "processing_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "quality": {"overall_score": 0.95},
                        "quality_selection": {
                            "available": True,
                            "candidate_count": 2,
                            "candidate_quality": {
                                "matanyone2": {"overall_score": 0.8},
                                "traditional": {"overall_score": 0.9},
                            },
                            "selected_model_counts": {"traditional": 1},
                        },
                    }
                ),
                encoding="utf-8",
            )
            return {"processing_report_path": str(report_path)}

    manifest = MattingQualityRegressionManifest.from_path(manifest_path)
    summary_path = MattingQualityRegressionRunner(
        manifest=manifest,
        pipeline_factory=FakePipeline,
    ).run(tmp_path / "out")

    sample = json.loads(summary_path.read_text(encoding="utf-8"))["samples"][0]
    assert sample["status"] == "failed"
    assert "required_temporal_model matanyone2 did not contribute" in sample["failures"]
