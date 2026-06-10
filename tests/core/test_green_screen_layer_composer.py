import sys
from pathlib import Path
from dataclasses import is_dataclass

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.matte.green_screen_layer_composer import (
    CompetitiveLayerResult,
    GreenScreenCompetitiveLayerComposer,
    LayerCandidate,
    LayerOwnership,
)


def test_competitive_layer_contract_types_are_dataclasses():
    assert is_dataclass(LayerCandidate)
    assert is_dataclass(LayerOwnership)
    assert is_dataclass(CompetitiveLayerResult)


def test_debug_shape_contract_records_candidate_ownership_and_result_shapes():
    candidate = LayerCandidate(
        name="subject",
        alpha=np.zeros((2, 3), dtype=np.float32),
        debug={"source": "ai"},
    )
    ownership = LayerOwnership(
        owner=np.full((2, 3), "subject", dtype=object),
        confidence=np.ones((2, 3), dtype=np.float32),
    )
    result = CompetitiveLayerResult(
        alpha=np.ones((2, 3), dtype=np.float32),
        ownership=ownership,
        candidates=(candidate,),
    )

    contract = result.debug_shape_contract()

    assert contract == {
        "alpha": (2, 3),
        "ownership.owner": (2, 3),
        "ownership.confidence": (2, 3),
        "candidates.subject.alpha": (2, 3),
    }


def test_competitive_layer_composer_validates_debug_shape_contract():
    composer = GreenScreenCompetitiveLayerComposer()
    candidate = LayerCandidate("subject", np.zeros((2, 3), dtype=np.float32))
    ownership = LayerOwnership(
        owner=np.full((2, 3), "subject", dtype=object),
        confidence=np.ones((2, 3), dtype=np.float32),
    )
    result = CompetitiveLayerResult(
        alpha=np.ones((2, 3), dtype=np.float32),
        ownership=ownership,
        candidates=(candidate,),
    )

    assert composer.validate_debug_shape_contract(result) == result


def test_debug_shape_contract_rejects_mismatched_candidate_shape():
    composer = GreenScreenCompetitiveLayerComposer()
    candidate = LayerCandidate("subject", np.zeros((1, 3), dtype=np.float32))
    ownership = LayerOwnership(
        owner=np.full((2, 3), "subject", dtype=object),
        confidence=np.ones((2, 3), dtype=np.float32),
    )
    result = CompetitiveLayerResult(
        alpha=np.ones((2, 3), dtype=np.float32),
        ownership=ownership,
        candidates=(candidate,),
    )

    try:
        composer.validate_debug_shape_contract(result)
    except ValueError as exc:
        assert "candidates.subject.alpha" in str(exc)
        assert "(1, 3)" in str(exc)
        assert "(2, 3)" in str(exc)
    else:
        raise AssertionError("Expected mismatched candidate shape to raise ValueError")
