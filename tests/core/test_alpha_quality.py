import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.alpha_quality import AlphaQualityAnalyzer


def test_alpha_quality_reports_speckles_holes_uncertain_edges_and_temporal_flicker():
    analyzer = AlphaQualityAnalyzer()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    alpha_a = np.zeros((8, 8), dtype=np.float32)
    alpha_a[2:6, 2:6] = 1.0
    alpha_a[3, 3] = 0.0
    alpha_a[0, 0] = 0.6
    alpha_a[1, 6] = 0.45
    alpha_b = alpha_a.copy()
    alpha_b[2:6, 2:6] = 0.35

    report = analyzer.analyze_sequence([frame, frame], [alpha_a, alpha_b])

    assert report.frame_count == 2
    assert report.mean_edge_uncertainty > 0.0
    assert report.speckle_pixels >= 2
    assert report.hole_pixels >= 1
    assert report.temporal_flicker > 0.10
    assert 0.0 <= report.overall_score <= 1.0


def test_alpha_quality_debug_overlay_marks_problem_regions_with_distinct_colors():
    analyzer = AlphaQualityAnalyzer()
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    alpha = np.zeros((6, 6), dtype=np.float32)
    alpha[:, :3] = 1.0
    alpha[2, 3] = 0.5
    alpha[0, 5] = 0.4
    alpha[4, 1] = 0.0

    overlay = analyzer.build_debug_overlay(frame, alpha)

    assert overlay.shape == frame.shape
    assert overlay.dtype == np.uint8
    assert np.array_equal(overlay[2, 3], np.array([255, 180, 0], dtype=np.uint8))
    assert np.array_equal(overlay[0, 5], np.array([255, 0, 255], dtype=np.uint8))
    assert np.array_equal(overlay[4, 1], np.array([0, 80, 255], dtype=np.uint8))
