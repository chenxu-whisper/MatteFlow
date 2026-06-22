import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig
from matteflow.refine.edge_refiner import EdgeRefiner


def test_refiner_softens_hard_binary_edge_near_transition():
    refiner = EdgeRefiner(MattingConfig())
    frame = np.full((7, 7, 3), 255, dtype=np.uint8)
    alpha = np.zeros((7, 7), dtype=np.float32)
    alpha[:, :4] = 1.0

    refined = refiner.refine([frame], [alpha])[0]

    assert 0.0 < float(refined[3, 4]) < 1.0
    assert float(refined[3, 0]) == 1.0
    assert float(refined[3, 6]) == 0.0


def test_refiner_glow_feather_strength_expands_soft_transition_band():
    frame = np.full((9, 9, 3), 255, dtype=np.uint8)
    alpha = np.zeros((9, 9), dtype=np.float32)
    alpha[:, :4] = 1.0
    alpha[:, 4] = 0.65
    alpha[:, 5] = 0.25

    weak = EdgeRefiner(MattingConfig(glow_feather_strength=0.0)).refine([frame], [alpha])[0]
    moderate = EdgeRefiner(MattingConfig(glow_feather_strength=1.0)).refine([frame], [alpha])[0]
    strong = EdgeRefiner(MattingConfig(glow_feather_strength=2.0)).refine([frame], [alpha])[0]

    assert float(weak[4, 6]) <= float(moderate[4, 6]) <= float(strong[4, 6])
    assert float(weak[4, 1]) == 1.0
    assert float(strong[4, 1]) == 1.0
