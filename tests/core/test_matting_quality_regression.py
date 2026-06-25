import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.evaluation.matting_quality_regression import (  # noqa: E402
    MattingQualityRegressionManifest,
    MattingQualityRegressionRunner,
)


def test_manifest_loads_samples():
    manifest = MattingQualityRegressionManifest.from_path(
        Path("tests/fixtures/matting_quality/manifest.json")
    )

    assert len(manifest.samples) == 1
    assert manifest.samples[0].name == "green_frame_smoke"
    assert manifest.samples[0].candidate_models == ("traditional",)
    assert manifest.samples[0].risk_ceilings["background_residue"] == 0.2


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
    assert payload["summary"]["sample_count"] == 1
    assert payload["samples"][0]["name"] == "green_frame_smoke"
    assert payload["samples"][0]["status"] == "completed"
    assert calls[0].quality_selection_enable is True
    assert calls[0].quality_candidate_models == ("traditional",)
