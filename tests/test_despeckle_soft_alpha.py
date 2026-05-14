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
