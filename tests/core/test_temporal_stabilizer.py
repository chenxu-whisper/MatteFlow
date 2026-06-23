import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig, QualityMode
from matteflow.temporal.temporal_stabilizer import TemporalStabilizer


def test_high_quality_stabilizer_uses_frame_motion_to_preserve_moving_alpha_shape():
    stabilizer = TemporalStabilizer(MattingConfig(quality_mode=QualityMode.HIGH, temporal_strength=0.8))
    frame_a = np.zeros((32, 32, 3), dtype=np.uint8)
    frame_b = np.zeros((32, 32, 3), dtype=np.uint8)
    frame_a[10:18, 8:16] = 255
    frame_b[10:18, 12:20] = 255
    alpha_a = np.zeros((32, 32), dtype=np.float32)
    alpha_b = np.zeros((32, 32), dtype=np.float32)
    alpha_a[10:18, 8:16] = 1.0
    alpha_b[10:18, 12:20] = 0.55

    stabilized = stabilizer.stabilize([alpha_a, alpha_b], frames=[frame_a, frame_b])

    assert float(stabilized[1][10:18, 12:20].mean()) > float(alpha_b[10:18, 12:20].mean())
    assert float(stabilized[1][10:18, 8:12].mean()) < 0.20


def test_high_quality_stabilizer_keeps_alpha_only_fallback_when_frames_are_missing():
    stabilizer = TemporalStabilizer(MattingConfig(quality_mode=QualityMode.HIGH, temporal_strength=0.8))
    alpha_a = np.zeros((6, 6), dtype=np.float32)
    alpha_b = np.ones((6, 6), dtype=np.float32)

    stabilized = stabilizer.stabilize([alpha_a, alpha_b])

    assert len(stabilized) == 2
    assert stabilized[1].shape == alpha_b.shape
