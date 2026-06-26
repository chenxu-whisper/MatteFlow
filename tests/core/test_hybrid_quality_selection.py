import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, MattingConfig  # noqa: E402
from matteflow.matte.candidates.matanyone2 import MatAnyone2CandidateGenerator  # noqa: E402
from matteflow.matte.hybrid_matte import HybridMatte  # noqa: E402


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
