import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig, QualityMode
from matteflow.temporal.temporal_stabilizer import TemporalStabilizer


def test_temporal_stabilizer_only_blends_mid_alpha_range():
    stabilizer = TemporalStabilizer(
        MattingConfig(
            quality_mode=QualityMode.STANDARD,
            temporal_strength=0.8,
            transparency_temporal_low=0.03,
            transparency_temporal_high=0.75,
            transparency_temporal_blend=0.2,
        )
    )

    prev_alpha = np.array([[0.0, 0.40, 1.0]], dtype=np.float32)
    curr_alpha = np.array([[0.0, 0.20, 1.0]], dtype=np.float32)

    stabilized = stabilizer.stabilize([prev_alpha, curr_alpha])

    assert np.isclose(stabilized[1][0, 0], curr_alpha[0, 0])
    assert stabilized[1][0, 1] > curr_alpha[0, 1]
    assert stabilized[1][0, 1] < prev_alpha[0, 1]
    assert np.isclose(stabilized[1][0, 2], curr_alpha[0, 2])
