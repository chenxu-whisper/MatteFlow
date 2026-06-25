import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.region_ownership import RegionOwnership  # noqa: E402
from matteflow.evaluation.matte_quality import MatteQualityEvaluator  # noqa: E402
from matteflow.matte.candidates.types import MatteCandidateSequence  # noqa: E402


def _ownership(shape=(2, 2)) -> RegionOwnership:
    empty = np.zeros(shape, dtype=bool)
    subject = empty.copy()
    subject[0, 0] = True
    hair_edge = empty.copy()
    hair_edge[0, min(1, shape[1] - 1)] = True
    background_residue = empty.copy()
    background_residue[min(1, shape[0] - 1), min(1, shape[1] - 1)] = True
    return RegionOwnership(
        subject=subject,
        hair_edge=hair_edge,
        luminous_prop=empty.copy(),
        transparent_effect=empty.copy(),
        background_residue=background_residue,
        uncertain_edge=hair_edge.copy(),
    )


def _candidate(alpha: np.ndarray) -> MatteCandidateSequence:
    return MatteCandidateSequence.from_raw(
        name="candidate",
        alphas=[alpha],
        confidences=[None],
        source="test",
        runtime_ms=1.0,
        diagnostics={},
        frame_shapes=[tuple(alpha.shape)],
    )


def test_matte_quality_evaluator_scores_regions_and_serializes_to_json():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    candidate = _candidate(np.array([[1.0, 0.5], [0.0, 0.0]], dtype=np.float32))

    report = MatteQualityEvaluator().evaluate(
        frames=[frame],
        candidates=[candidate],
        ownerships=[_ownership()],
    )

    quality = report.qualities[0]
    assert quality.region_scores["subject"] == 1.0
    assert quality.region_scores["background_residue"] == 1.0
    assert quality.region_scores["hair_edge"] > 0.9
    json.dumps(report.to_dict())
    assert report.to_summary()["candidate"]["frame_count"] == 1


def test_matte_quality_evaluator_rejects_frame_ownership_length_mismatch():
    with pytest.raises(ValueError, match="frames length"):
        MatteQualityEvaluator().evaluate(
            frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
            candidates=[],
            ownerships=[],
        )


def test_matte_quality_evaluator_rejects_candidate_frame_count_mismatch():
    candidate = MatteCandidateSequence.from_raw(
        name="bad",
        alphas=[
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        ],
        confidences=[None, None],
        source="test",
        runtime_ms=1.0,
        diagnostics={},
        frame_shapes=[(2, 2), (2, 2)],
    )

    with pytest.raises(ValueError, match="candidate frame count"):
        MatteQualityEvaluator().evaluate(
            frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
            candidates=[candidate],
            ownerships=[_ownership()],
        )


def test_matte_quality_evaluator_rejects_ownership_shape_mismatch():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    candidate = _candidate(np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="ownership mask shape"):
        MatteQualityEvaluator().evaluate(
            frames=[frame],
            candidates=[candidate],
            ownerships=[_ownership(shape=(1, 2))],
        )
