import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig
from matteflow.refine.despeckle import Despeckle


def test_despeckle_threshold_does_not_override_clip_black():
    config = MattingConfig()

    config.clip_black = 0.0
    config.despeckle_threshold = 0.1

    assert config.clip_black == 0.0
    assert config.despeckle_threshold == 0.1


def test_despeckle_preserves_soft_alpha_values():
    config = MattingConfig()
    config.despeckle_enable = True
    config.despeckle_radius = 1
    config.despeckle_threshold = 0.0

    alpha = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.7, 1.0],
        ],
        dtype=np.float32,
    )

    cleaned = Despeckle(config).process([alpha])[0]

    assert np.any((cleaned > 0.0) & (cleaned < 1.0))
    assert len(np.unique((cleaned * 255).astype(np.uint8))) > 2


def test_despeckle_keeps_supported_soft_alpha_island():
    config = MattingConfig()
    config.despeckle_enable = True
    config.despeckle_radius = 5
    config.despeckle_threshold = 0.0

    alpha = np.zeros((15, 15), dtype=np.float32)
    alpha[4:11, 4:11] = 0.6
    alpha[6:9, 6:9] = 0.2

    cleaned = Despeckle(config).process([alpha])[0]

    assert cleaned[7, 7] >= 0.18
    assert np.any(cleaned[4:11, 4:11] > 0.03)


def test_despeckle_gvm_context_keeps_supported_swirl_soft_alpha():
    config = MattingConfig()
    config.despeckle_enable = True
    config.despeckle_radius = 5
    config.despeckle_threshold = 0.0

    alpha = np.zeros((15, 15), dtype=np.float32)
    alpha[4:11, 4:11] = 0.6
    alpha[6:9, 6:9] = 0.2
    frame = np.full((15, 15, 3), [20, 200, 20], dtype=np.uint8)
    frame[4:11, 4:11] = [170, 135, 240]

    cleaned = Despeckle(config).process(
        [alpha],
        frames=[frame],
        context={"active_ai_model": "gvm"},
    )[0]

    assert cleaned[7, 7] >= 0.18


def test_despeckle_gvm_context_does_not_keep_non_swirl_soft_alpha():
    config = MattingConfig()
    config.despeckle_enable = True
    config.despeckle_radius = 5
    config.despeckle_threshold = 0.0

    alpha = np.zeros((15, 15), dtype=np.float32)
    alpha[4:11, 4:11] = 0.6
    alpha[6:9, 6:9] = 0.2
    frame = np.full((15, 15, 3), [20, 200, 20], dtype=np.uint8)
    frame[4:11, 4:11] = [155, 210, 155]

    cleaned = Despeckle(config).process(
        [alpha],
        frames=[frame],
        context={"active_ai_model": "gvm"},
    )[0]

    assert cleaned[7, 7] <= 0.03


def test_despeckle_still_removes_isolated_soft_speckle():
    config = MattingConfig()
    config.despeckle_enable = True
    config.despeckle_radius = 5
    config.despeckle_threshold = 0.0

    alpha = np.zeros((15, 15), dtype=np.float32)
    alpha[7, 7] = 0.2

    cleaned = Despeckle(config).process([alpha])[0]

    assert cleaned[7, 7] == 0.0
