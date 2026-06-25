import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.alpha_quality import AlphaQualityAnalyzer  # noqa: E402


def test_mean_edge_uncertainty_only_counts_soft_alpha_near_foreground_boundary():
    alpha = np.zeros((10, 10), dtype=np.float32)
    alpha[1:4, 1:4] = 0.5
    alpha[6:9, 6:9] = 1.0
    alpha[5, 6:9] = 0.5

    report = AlphaQualityAnalyzer().analyze_sequence(
        [np.zeros((10, 10, 3), dtype=np.uint8)],
        [alpha],
    )

    assert report.mean_edge_uncertainty == 0.03


def test_hole_mask_ignores_noise_components_with_area_at_most_four():
    alpha = np.ones((12, 12), dtype=np.float32)
    alpha[5:7, 5:7] = 0.69

    holes = AlphaQualityAnalyzer()._hole_mask(alpha)

    assert not np.any(holes)


def test_hole_mask_keeps_larger_closed_components():
    alpha = np.ones((12, 12), dtype=np.float32)
    alpha[5:7, 5:8] = 0.69

    holes = AlphaQualityAnalyzer()._hole_mask(alpha)

    assert int(holes.sum()) == 6
