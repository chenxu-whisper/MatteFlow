import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig
from matteflow.matte.green_screen_matte import GreenScreenMatte


def test_green_similarity_controls_how_aggressively_green_is_keyed():
    frame = np.full((8, 8, 3), [70, 145, 75], dtype=np.uint8)

    conservative = MattingConfig(screen_color="green", green_similarity=0.1)
    aggressive = MattingConfig(screen_color="green", green_similarity=1.0)

    conservative_alpha = GreenScreenMatte(conservative).generate(frame)
    aggressive_alpha = GreenScreenMatte(aggressive).generate(frame)

    assert aggressive_alpha.mean() < conservative_alpha.mean() - 0.2
