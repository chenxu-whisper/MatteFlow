import sys
import logging
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, MattingConfig  # noqa: E402
from matteflow.matte.candidates.types import CandidateGenerationResult, MatteCandidateSequence  # noqa: E402
from matteflow.matte.quality_driven_matte import QualityDrivenMatte  # noqa: E402


class FakeGenerator:
    name = "fake"

    def __init__(self, alpha: np.ndarray):
        self.alpha = alpha

    def generate(self, frames, *, frame_shapes, cancel_check=None, progress_callback=None):
        return CandidateGenerationResult(
            candidate=MatteCandidateSequence.from_raw(
                name=self.name,
                alphas=[self.alpha for _ in frames],
                confidences=[None for _ in frames],
                source="fake",
                runtime_ms=1.0,
                diagnostics={"available": True},
                frame_shapes=frame_shapes,
            )
        )


def test_quality_driven_matte_runs_generators_and_records_summary():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = [255, 255, 255]
    frame[1, 0] = [255, 180, 80]
    generator = FakeGenerator(np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32))

    matte = QualityDrivenMatte(
        MattingConfig(background_mode=BackgroundMode.GREEN_SCREEN),
        generators=[generator],
    )
    alphas = matte.generate_sequence([frame])

    assert len(alphas) == 1
    assert alphas[0].shape == (2, 2)
    assert matte.last_quality_selection["available"] is True
    assert matte.last_quality_selection["candidate_count"] == 1
    assert matte.last_quality_selection["candidate_quality"]["fake"]["frame_count"] == 1


def test_quality_driven_matte_reports_unwired_default_model_candidates_as_skipped():
    matte = QualityDrivenMatte(MattingConfig(background_mode=BackgroundMode.GREEN_SCREEN))
    names = [generator.name for generator in matte.generators]

    assert names == ["matanyone2", "sam2", "birefnet", "traditional"]

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    matte.generate_sequence([frame])

    skipped = matte.last_quality_selection["skipped_candidates"]
    assert [item["name"] for item in skipped] == ["matanyone2", "sam2", "birefnet"]
    assert skipped[1]["reason"] == "guidance_missing"


def test_quality_driven_matte_logs_skipped_candidates(caplog):
    matte = QualityDrivenMatte(MattingConfig(background_mode=BackgroundMode.GREEN_SCREEN))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with caplog.at_level(logging.INFO, logger="matteflow.matte.quality_driven_matte"):
        matte.generate_sequence([frame])

    assert (
        "Quality candidate skipped: name=sam2 reason=guidance_missing "
        "message=SAM2 candidate requires guidance and is not wired in this phase"
    ) in caplog.text
