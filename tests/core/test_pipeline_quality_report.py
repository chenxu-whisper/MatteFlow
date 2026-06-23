import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, MattingConfig, QualityMode  # noqa: E402
from matteflow.pipeline import MattingPipeline  # noqa: E402


def test_pipeline_writes_processing_report_and_returns_path(monkeypatch, tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.FAST,
        output_fg=False,
        output_matte=False,
        output_comp=False,
        output_processed=False,
        output_debug=False,
    )
    pipeline = MattingPipeline(config)
    frame = np.full((4, 4, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[1:3, 1:3] = 1.0

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: ([frame], {"width": 4, "height": 4, "fps": 1.0}),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda frames_arg, bg_mode, progress_callback, cancel_check=None: [alpha],
    )
    monkeypatch.setattr(
        pipeline.decontaminate,
        "process",
        lambda frames_arg, alphas_arg, bg_mode, context=None: frames_arg,
    )

    result = pipeline.process(tmp_path / "input.png", tmp_path / "out")

    report_path = tmp_path / "out" / "processing_report.json"
    assert result["processing_report_path"] == str(report_path)
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["job"]["background_mode_effective"] == "green_screen"
    assert payload["job"]["frame_count"] == 1
    assert payload["quality"]["frame_count"] == 1
    assert payload["regions"]["subject_pixels"] >= 4
    assert payload["foreground_recovery"] == {}
