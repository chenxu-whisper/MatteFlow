import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.region_ownership import RegionOwnership  # noqa: E402
from matteflow.evaluation.matte_quality import (  # noqa: E402
    CandidateQuality,
    CandidateQualityReport,
    MatteQualityEvaluator,
)
from matteflow.matte.candidates.types import MatteCandidateSequence  # noqa: E402
from matteflow.matte.quality_selector import QualitySelector  # noqa: E402


def _ownership() -> RegionOwnership:
    return RegionOwnership(
        subject=np.array([[True, False], [False, False]]),
        hair_edge=np.array([[False, True], [False, False]]),
        luminous_prop=np.array([[False, False], [True, False]]),
        transparent_effect=np.array([[False, False], [True, False]]),
        background_residue=np.array([[False, False], [False, True]]),
        uncertain_edge=np.array([[False, True], [False, False]]),
    )


def _candidate(name: str, alpha: np.ndarray) -> MatteCandidateSequence:
    return MatteCandidateSequence.from_raw(
        name=name,
        alphas=[alpha],
        confidences=[None],
        source="test",
        runtime_ms=1.0,
        diagnostics={},
        frame_shapes=[(2, 2)],
    )


def _candidate_with_shape(name: str, alpha: np.ndarray) -> MatteCandidateSequence:
    return MatteCandidateSequence.from_raw(
        name=name,
        alphas=[alpha],
        confidences=[None],
        source="test",
        runtime_ms=1.0,
        diagnostics={},
        frame_shapes=[tuple(alpha.shape)],
    )


def test_quality_selector_chooses_best_candidate_per_region():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    candidate_a = _candidate(
        "subject_good",
        np.array([[1.0, 0.0], [0.1, 0.9]], dtype=np.float32),
    )
    candidate_b = _candidate(
        "effect_good",
        np.array([[0.4, 0.6], [0.45, 0.0]], dtype=np.float32),
    )
    ownership = _ownership()
    quality_report = MatteQualityEvaluator().evaluate(
        frames=[frame],
        candidates=[candidate_a, candidate_b],
        ownerships=[ownership],
    )

    result = QualitySelector().select(
        candidates=[candidate_a, candidate_b],
        quality_report=quality_report,
        ownerships=[ownership],
    )

    assert result.alphas[0][0, 0] == 1.0
    assert result.alphas[0][1, 0] == 0.45
    assert result.alphas[0][1, 1] == 0.0
    assert result.selected_model_counts["subject_good"] >= 1
    assert result.selected_model_counts["effect_good"] >= 1
    assert result.to_dict()["available"] is True


def test_quality_selector_skips_unavailable_candidate_and_selects_best_model_by_region():
    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    ownership = RegionOwnership(
        subject=np.array(
            [
                [True, True, False],
                [False, False, False],
                [False, False, False],
            ]
        ),
        hair_edge=np.array(
            [
                [False, False, False],
                [True, False, False],
                [False, False, False],
            ]
        ),
        luminous_prop=np.array(
            [
                [False, False, False],
                [False, True, False],
                [False, False, False],
            ]
        ),
        transparent_effect=np.array(
            [
                [False, False, False],
                [False, False, True],
                [False, False, False],
            ]
        ),
        background_residue=np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, True],
            ]
        ),
        uncertain_edge=np.array(
            [
                [False, False, False],
                [True, False, False],
                [False, False, False],
            ]
        ),
    )
    matanyone2 = _candidate_with_shape(
        "matanyone2",
        np.array(
            [
                [1.0, 0.98, 0.0],
                [0.95, 0.02, 0.03],
                [0.0, 0.0, 0.40],
            ],
            dtype=np.float32,
        ),
    )
    traditional = _candidate_with_shape(
        "traditional",
        np.array(
            [
                [0.60, 0.62, 0.0],
                [0.45, 0.50, 0.55],
                [0.0, 0.0, 0.20],
            ],
            dtype=np.float32,
        ),
    )
    birefnet = _candidate_with_shape(
        "birefnet",
        np.array(
            [
                [0.70, 0.72, 0.0],
                [0.10, 0.10, 0.10],
                [0.0, 0.0, 0.00],
            ],
            dtype=np.float32,
        ),
    )
    skipped = [
        {
            "name": "sam2",
            "reason": "guidance_missing",
            "message": "SAM2 candidate requires guidance",
        }
    ]
    candidates = [matanyone2, traditional, birefnet]
    quality_report = MatteQualityEvaluator().evaluate(
        frames=[frame],
        candidates=candidates,
        ownerships=[ownership],
    )

    result = QualitySelector().select(
        candidates=candidates,
        quality_report=quality_report,
        ownerships=[ownership],
        skipped_candidates=skipped,
    )

    selected = result.alphas[0]
    assert selected[0, 0] == 1.0
    assert selected[0, 1] == 0.98
    assert selected[1, 0] == 0.45
    assert selected[1, 1] == 0.50
    assert selected[1, 2] == 0.55
    assert selected[2, 2] == 0.00

    payload = result.to_dict()
    assert payload["candidate_count"] == 3
    assert payload["skipped_candidates"] == skipped
    assert "sam2" not in payload["selected_model_counts"]
    assert payload["selected_model_counts"] == {
        "matanyone2": 1,
        "traditional": 3,
        "birefnet": 1,
    }


def test_quality_selector_requires_clear_edge_score_gain_and_overall_guard():
    ownership = RegionOwnership(
        subject=np.zeros((2, 2), dtype=bool),
        hair_edge=np.array([[True, True], [False, False]]),
        luminous_prop=np.zeros((2, 2), dtype=bool),
        transparent_effect=np.zeros((2, 2), dtype=bool),
        background_residue=np.zeros((2, 2), dtype=bool),
        uncertain_edge=np.zeros((2, 2), dtype=bool),
    )
    traditional = _candidate(
        "traditional",
        np.array([[0.45, 0.55], [0.0, 0.0]], dtype=np.float32),
    )
    birefnet = _candidate(
        "birefnet",
        np.array([[0.25, 0.75], [0.0, 0.0]], dtype=np.float32),
    )
    quality_report = CandidateQualityReport(
        qualities=(
            CandidateQuality(
                candidate_name="traditional",
                frame_index=0,
                overall_score=0.70,
                region_scores={"hair_edge": 0.62},
                signals={},
            ),
            CandidateQuality(
                candidate_name="birefnet",
                frame_index=0,
                overall_score=0.60,
                region_scores={"hair_edge": 0.64},
                signals={},
            ),
        )
    )

    result = QualitySelector().select(
        candidates=[traditional, birefnet],
        quality_report=quality_report,
        ownerships=[ownership],
    )

    selected = result.alphas[0]
    assert np.allclose(selected[ownership.hair_edge], traditional.alphas[0][ownership.hair_edge])
    assert result.selected_model_counts == {"traditional": 1}


def test_quality_selector_allows_clear_edge_improvement_without_overall_regression():
    ownership = RegionOwnership(
        subject=np.zeros((2, 2), dtype=bool),
        hair_edge=np.array([[True, True], [False, False]]),
        luminous_prop=np.zeros((2, 2), dtype=bool),
        transparent_effect=np.zeros((2, 2), dtype=bool),
        background_residue=np.zeros((2, 2), dtype=bool),
        uncertain_edge=np.zeros((2, 2), dtype=bool),
    )
    traditional = _candidate(
        "traditional",
        np.array([[0.45, 0.55], [0.0, 0.0]], dtype=np.float32),
    )
    birefnet = _candidate(
        "birefnet",
        np.array([[0.35, 0.65], [0.0, 0.0]], dtype=np.float32),
    )
    quality_report = CandidateQualityReport(
        qualities=(
            CandidateQuality(
                candidate_name="traditional",
                frame_index=0,
                overall_score=0.70,
                region_scores={"hair_edge": 0.62},
                signals={},
            ),
            CandidateQuality(
                candidate_name="birefnet",
                frame_index=0,
                overall_score=0.67,
                region_scores={"hair_edge": 0.67},
                signals={},
            ),
        )
    )

    result = QualitySelector().select(
        candidates=[traditional, birefnet],
        quality_report=quality_report,
        ownerships=[ownership],
    )

    selected = result.alphas[0]
    assert np.allclose(selected[ownership.hair_edge], birefnet.alphas[0][ownership.hair_edge])
    assert result.selected_model_counts == {"birefnet": 1}
