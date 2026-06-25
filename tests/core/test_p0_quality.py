import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.alpha_quality import AlphaQualityReport  # noqa: E402
from matteflow.analysis.p0_quality import P0QualityAnalyzer  # noqa: E402
from matteflow.analysis.region_ownership import RegionOwnership, RegionOwnershipAnalyzer  # noqa: E402


def _quality_report(**overrides):
    values = {
        "frame_count": 1,
        "mean_edge_uncertainty": 0.02,
        "speckle_pixels": 0,
        "hole_pixels": 0,
        "background_residue": 0.0,
        "temporal_flicker": 0.0,
        "overall_score": 0.95,
    }
    values.update(overrides)
    return AlphaQualityReport(**values)


def _ownership(shape=(4, 4), **masks):
    fields = {
        "subject": np.zeros(shape, dtype=bool),
        "hair_edge": np.zeros(shape, dtype=bool),
        "luminous_prop": np.zeros(shape, dtype=bool),
        "transparent_effect": np.zeros(shape, dtype=bool),
        "background_residue": np.zeros(shape, dtype=bool),
        "uncertain_edge": np.zeros(shape, dtype=bool),
    }
    fields.update(masks)
    return RegionOwnership(**fields)


def test_p0_quality_report_contains_all_six_risks_for_empty_input():
    report = P0QualityAnalyzer().analyze_sequence([], [])

    payload = report.to_dict()
    assert set(payload) == {
        "hair_edge_loss",
        "background_residue",
        "light_subject_loss",
        "transparent_effect_loss",
        "temporal_instability",
        "subject_misidentification",
    }
    assert all(risk["level"] == "pass" for risk in payload.values())
    assert all(risk["score"] == 0.0 for risk in payload.values())


def test_p0_quality_flags_background_residue_from_quality_and_region_evidence():
    residue_mask = np.zeros((4, 4), dtype=bool)
    residue_mask[:2, :2] = True

    report = P0QualityAnalyzer().analyze_sequence(
        [np.zeros((4, 4, 3), dtype=np.uint8)],
        [np.full((4, 4), 0.2, dtype=np.float32)],
        quality_report=_quality_report(background_residue=0.08),
        region_context={"region_ownership": [_ownership(background_residue=residue_mask)]},
    )

    risk = report.to_dict()["background_residue"]
    assert risk["level"] == "fail"
    assert risk["score"] >= 0.8
    assert risk["signals"]["quality_background_residue"] == 0.08
    assert risk["signals"]["region_background_residue_ratio"] == 0.25


def test_region_ownership_does_not_mark_high_alpha_white_subject_as_background_residue():
    frame = np.full((32, 32, 3), [232, 235, 242], dtype=np.uint8)
    alpha = np.ones((32, 32), dtype=np.float32)

    ownership = RegionOwnershipAnalyzer().analyze(frame, alpha, alpha)

    assert not np.any(ownership.background_residue)
    assert np.all(ownership.subject)


def test_p0_quality_flags_light_subject_loss_from_bright_low_saturation_pixels():
    frame = np.full((4, 4, 3), 230, dtype=np.uint8)
    alpha = np.full((4, 4), 0.15, dtype=np.float32)
    subject_mask = np.ones((4, 4), dtype=bool)

    report = P0QualityAnalyzer().analyze_sequence(
        [frame],
        [alpha],
        quality_report=_quality_report(),
        region_context={"region_ownership": [_ownership(subject=subject_mask)]},
    )

    risk = report.to_dict()["light_subject_loss"]
    assert risk["level"] == "fail"
    assert risk["signals"]["light_low_alpha_ratio"] == 1.0


def test_p0_quality_flags_transparent_effect_loss_when_effect_regions_are_cleared():
    frame = np.full((4, 4, 3), 180, dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.float32)
    effect_mask = np.zeros((4, 4), dtype=bool)
    effect_mask[:, :2] = True

    report = P0QualityAnalyzer().analyze_sequence(
        [frame],
        [alpha],
        quality_report=_quality_report(),
        region_context={"region_ownership": [_ownership(transparent_effect=effect_mask)]},
    )

    risk = report.to_dict()["transparent_effect_loss"]
    assert risk["level"] == "fail"
    assert risk["signals"]["effect_low_alpha_ratio"] == 1.0


def test_p0_quality_uses_max_risk_so_one_bad_frame_is_not_hidden_by_sequence_mean():
    frames = [np.full((4, 4, 3), 180, dtype=np.uint8) for _ in range(10)]
    alphas = [np.ones((4, 4), dtype=np.float32) for _ in range(10)]
    effect_masks = [np.zeros((4, 4), dtype=bool) for _ in range(10)]
    effect_masks[7][:, :] = True
    alphas[7] = np.zeros((4, 4), dtype=np.float32)

    report = P0QualityAnalyzer().analyze_sequence(
        frames,
        alphas,
        quality_report=_quality_report(frame_count=10),
        region_context={
            "region_ownership": [
                _ownership(transparent_effect=effect_mask)
                for effect_mask in effect_masks
            ]
        },
    )

    risk = report.to_dict()["transparent_effect_loss"]
    assert risk["level"] == "fail"
    assert risk["signals"]["effect_low_alpha_ratio"] == 0.1
    assert risk["signals"]["effect_low_alpha_ratio_max"] == 1.0
    assert risk["signals"]["effect_low_alpha_ratio_top3_mean"] == 0.333333


def test_p0_quality_uses_top3_mean_for_repeated_medium_risk_frames():
    frames = [np.full((4, 4, 3), 180, dtype=np.uint8) for _ in range(8)]
    alphas = [np.ones((4, 4), dtype=np.float32) for _ in range(8)]
    effect_masks = [np.zeros((4, 4), dtype=bool) for _ in range(8)]
    medium_loss_mask = np.zeros((4, 4), dtype=bool)
    medium_loss_mask[:2, :] = True
    medium_loss_mask[2, :2] = True

    for index in (1, 4, 6):
        effect_masks[index][:, :] = True
        alphas[index][medium_loss_mask] = 0.0

    report = P0QualityAnalyzer().analyze_sequence(
        frames,
        alphas,
        quality_report=_quality_report(frame_count=8),
        region_context={
            "region_ownership": [
                _ownership(transparent_effect=effect_mask)
                for effect_mask in effect_masks
            ]
        },
    )

    risk = report.to_dict()["transparent_effect_loss"]
    assert risk["level"] == "fail"
    assert risk["score"] == 1.0
    assert risk["signals"]["effect_low_alpha_ratio"] == 0.234375
    assert risk["signals"]["effect_low_alpha_ratio_max"] == 0.625
    assert risk["signals"]["effect_low_alpha_ratio_top3_mean"] == 0.625


def test_p0_quality_flags_temporal_instability_from_quality_report():
    report = P0QualityAnalyzer().analyze_sequence(
        [np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)],
        [np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32)],
        quality_report=_quality_report(frame_count=2, temporal_flicker=0.22),
    )

    risk = report.to_dict()["temporal_instability"]
    assert risk["level"] == "fail"
    assert risk["signals"]["quality_temporal_flicker"] == 0.22


def test_p0_quality_flags_subject_misidentification_when_subject_coverage_is_tiny():
    alpha = np.zeros((10, 10), dtype=np.float32)
    alpha[0, 0] = 1.0
    subject_mask = np.zeros((10, 10), dtype=bool)
    subject_mask[0, 0] = True

    report = P0QualityAnalyzer().analyze_sequence(
        [np.zeros((10, 10, 3), dtype=np.uint8)],
        [alpha],
        quality_report=_quality_report(),
        region_context={"region_ownership": [_ownership((10, 10), subject=subject_mask)]},
    )

    risk = report.to_dict()["subject_misidentification"]
    assert risk["level"] == "warn"
    assert risk["signals"]["subject_coverage_ratio"] == 0.01
