import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, MattingConfig  # noqa: E402
from matteflow.matte.candidates.matanyone2 import MatAnyone2CandidateGenerator  # noqa: E402
from matteflow.matte.hybrid_matte import HybridMatte  # noqa: E402
from matteflow.reporting.processing_report import ProcessingReportBuilder  # noqa: E402


def test_hybrid_matte_delegates_to_quality_driven_matte_when_enabled():
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_selection_enable=True,
        quality_candidate_models=("traditional",),
    )
    config.use_ai = False
    hybrid = HybridMatte(config)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    frame[0, 0] = [255, 0, 0]

    alphas = hybrid.generate_sequence([frame], BackgroundMode.GREEN_SCREEN)

    assert len(alphas) == 1
    assert hybrid.last_active_ai_model == "quality_selection"
    assert hybrid.last_quality_selection["available"] is True
    assert hybrid.last_quality_selection["candidate_count"] == 1


class _FakeMatAnyone2Engine:
    model = object()

    def generate_sequence(self, frames, cancel_check=None):
        return [np.full(frame.shape[:2], 0.25, dtype=np.float32) for frame in frames]


def test_hybrid_quality_driven_matte_reuses_loaded_candidate_engines():
    config = MattingConfig(
        quality_selection_enable=True,
        quality_candidate_models=("matanyone2",),
    )
    config.use_ai = False
    hybrid = HybridMatte(config)
    hybrid.matanyone2 = _FakeMatAnyone2Engine()

    quality_matte = hybrid._build_quality_driven_matte(BackgroundMode.GREEN_SCREEN)

    assert isinstance(quality_matte.generators[0], MatAnyone2CandidateGenerator)
    assert quality_matte.generators[0].engine is hybrid.matanyone2


def test_unknown_background_traditional_fallback_does_not_select_black_for_green_screen():
    config = MattingConfig(background_mode=BackgroundMode.UNKNOWN)
    config.use_ai = False
    hybrid = HybridMatte(config)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, :, 1] = 177
    frame[:, :, 2] = 64
    frame[8:24, 8:24] = [255, 0, 0]

    alphas = hybrid.generate_sequence([frame], BackgroundMode.UNKNOWN)

    assert hybrid.last_active_ai_model == "traditional_green_fallback"
    assert float(alphas[0].mean()) < 0.5


def test_black_quality_selection_reports_black_effect_enhancement(tmp_path):
    config = MattingConfig(background_mode=BackgroundMode.BLACK_BACKGROUND)
    config.use_ai = False
    config.quality_selection_enable = True
    config.quality_candidate_models = ("traditional",)
    hybrid = HybridMatte(config)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[4:12, 4:12] = [30, 30, 30]

    hybrid.generate_sequence([frame], BackgroundMode.BLACK_BACKGROUND)
    report = ProcessingReportBuilder().build(
        input_path=tmp_path / "input.png",
        output_dir=tmp_path,
        config=config,
        frame_count=1,
        background_mode_effective=BackgroundMode.BLACK_BACKGROUND,
        timings={},
        quality_report=None,
        hybrid_matte=hybrid,
    )

    enhancement = report.to_dict()["black_effect_enhancement"]
    assert enhancement["frames"] == 1
    assert enhancement["smoke_pixels"] > 0
