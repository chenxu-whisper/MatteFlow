import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig, QualityMode
from matteflow.pipeline import MattingPipeline


def test_pipeline_writes_quality_debug_overlays_and_report(tmp_path):
    pipeline = MattingPipeline(MattingConfig(output_debug=True))
    frames = [np.zeros((6, 6, 3), dtype=np.uint8)]
    alphas = [np.zeros((6, 6), dtype=np.float32)]
    alphas[0][:, :3] = 1.0
    alphas[0][2, 3] = 0.5

    report = pipeline._write_quality_debug_outputs(frames, alphas, tmp_path)

    overlay_path = tmp_path / "debug" / "quality_overlay_000000.png"
    report_path = tmp_path / "debug" / "quality_report.txt"
    assert overlay_path.exists()
    assert report_path.exists()
    assert "overall_score=" in report_path.read_text(encoding="utf-8")
    assert report.frame_count == 1
    written = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    assert written is not None


def test_pipeline_skips_quality_debug_outputs_when_disabled(tmp_path):
    pipeline = MattingPipeline(MattingConfig(output_debug=False))
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    alphas = [np.zeros((4, 4), dtype=np.float32)]

    report = pipeline._write_quality_debug_outputs(frames, alphas, tmp_path)

    assert report is None
    assert not (tmp_path / "debug").exists()


def test_pipeline_passes_frames_to_high_quality_temporal_stabilizer():
    pipeline = MattingPipeline(MattingConfig(quality_mode=QualityMode.HIGH))
    captured = {}

    class RecorderStabilizer:
        def stabilize(self, alphas, frames=None):
            captured["frames"] = frames
            return alphas

    pipeline.stabilizer = RecorderStabilizer()
    frames = [np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2, 3), dtype=np.uint8)]
    alphas = [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)]

    result = pipeline._stabilize_alphas(frames, alphas)

    assert result is alphas
    assert captured["frames"] is frames
