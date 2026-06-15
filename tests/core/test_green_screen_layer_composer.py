import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.matte.green_screen_layer_composer import (
    CompetitiveLayerResult,
    GreenScreenCompetitiveLayerComposer,
    LayerCandidate,
    LayerOwnership,
)


def test_green_screen_competitive_layer_composer_shape_contract():
    subject = LayerCandidate(
        alpha=np.array([[1.0, 0.8, 0.0], [0.6, 0.2, 0.0]], dtype=np.float32),
        confidence=np.array([[1.0, 0.9, 0.0], [0.8, 0.4, 0.0]], dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.array([[0.0, 0.3, 0.9], [0.2, 0.4, 0.7]], dtype=np.float32),
        confidence=np.array([[0.0, 0.5, 1.0], [0.3, 0.6, 0.9]], dtype=np.float32),
    )

    result = GreenScreenCompetitiveLayerComposer().compose(subject=subject, effect=effect)

    assert isinstance(result, CompetitiveLayerResult)
    assert result.final_alpha.shape == (2, 3)
    assert result.subject_alpha_out.shape == (2, 3)
    assert result.effect_alpha_out.shape == (2, 3)
    assert isinstance(result.ownership, LayerOwnership)
    assert result.ownership.subject.shape == (2, 3)
    assert result.ownership.effect.shape == (2, 3)
    assert result.ownership.background.shape == (2, 3)
    assert set(result.debug_layers) == {
        "subject_alpha",
        "subject_confidence",
        "effect_alpha",
        "effect_confidence",
        "subject_evidence",
        "effect_evidence",
        "background_evidence",
        "background_evidence_owns",
        "effect_over_subject_evidence",
        "background_suppression",
        "subject_alpha_out",
        "effect_alpha_out",
        "final_alpha",
        "ownership_subject",
        "ownership_effect",
        "ownership_background",
    }
    assert all(layer.shape == (2, 3) for layer in result.debug_layers.values())


def test_minimal_compose_contract_uses_confidence_winner_values_and_float32_outputs():
    subject = LayerCandidate(
        alpha=np.array([[0.2, 0.8], [0.4, 0.9]], dtype=np.float64),
        confidence=np.array([[0.9, 0.1], [0.0, 0.0]], dtype=np.float64),
    )
    effect = LayerCandidate(
        alpha=np.array([[0.7, 0.6], [0.5, 0.3]], dtype=np.float64),
        confidence=np.array([[0.2, 0.8], [0.0, 1.0]], dtype=np.float64),
    )

    result = GreenScreenCompetitiveLayerComposer().compose(subject=subject, effect=effect)

    assert np.array_equal(
        result.ownership.subject,
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    assert np.array_equal(
        result.ownership.effect,
        np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert np.array_equal(
        result.ownership.background,
        np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    assert np.allclose(
        result.subject_alpha_out,
        np.array([[0.2, 0.0], [0.0, 0.0]], dtype=np.float32),
    )
    assert np.allclose(
        result.effect_alpha_out,
        np.array([[0.0, 0.6], [0.0, 0.3]], dtype=np.float32),
    )
    assert np.allclose(
        result.final_alpha,
        np.array([[0.2, 0.6], [0.0, 0.3]], dtype=np.float32),
    )
    for layer in (
        result.final_alpha,
        result.subject_alpha_out,
        result.effect_alpha_out,
        result.ownership.subject,
        result.ownership.effect,
        result.ownership.background,
        *result.debug_layers.values(),
    ):
        assert layer.dtype == np.float32


def test_effect_over_subject_evidence_routes_luminous_effect_out_of_semantic_subject_bleed():
    subject = LayerCandidate(
        alpha=np.array([[0.92, 0.85]], dtype=np.float32),
        confidence=np.array([[0.95, 0.90]], dtype=np.float32),
        evidence=np.array([[0.20, 0.88]], dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.array([[0.84, 0.72]], dtype=np.float32),
        confidence=np.array([[0.60, 0.50]], dtype=np.float32),
        evidence=np.array([[0.91, 0.40]], dtype=np.float32),
    )

    result = GreenScreenCompetitiveLayerComposer().compose(subject=subject, effect=effect)

    assert np.array_equal(result.ownership.effect, np.array([[1.0, 0.0]], dtype=np.float32))
    assert np.array_equal(result.ownership.subject, np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.allclose(result.effect_alpha_out, np.array([[0.84, 0.0]], dtype=np.float32))
    assert np.allclose(result.subject_alpha_out, np.array([[0.0, 0.85]], dtype=np.float32))


def test_background_suppression_prioritizes_low_support_blue_background_over_effect_leak():
    subject = LayerCandidate(
        alpha=np.array([[0.04, 0.62, 0.05]], dtype=np.float32),
        confidence=np.array([[0.03, 0.70, 0.04]], dtype=np.float32),
        evidence=np.array([[0.02, 0.64, 0.03]], dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.array([[0.58, 0.20, 0.72]], dtype=np.float32),
        confidence=np.array([[0.18, 0.10, 0.26]], dtype=np.float32),
        evidence=np.array([[0.16, 0.08, 0.24]], dtype=np.float32),
    )

    result = GreenScreenCompetitiveLayerComposer().compose(subject=subject, effect=effect)

    assert np.array_equal(
        result.ownership.background,
        np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
    )
    assert np.array_equal(
        result.debug_layers["background_suppression"],
        np.array([[1.0, 0.0, 1.0]], dtype=np.float32),
    )
    assert np.allclose(result.final_alpha, np.array([[0.0, 0.62, 0.0]], dtype=np.float32))


def test_frame_aware_background_evidence_overrides_competing_subject_and_effect_support():
    subject = LayerCandidate(
        alpha=np.array([[0.26, 0.82]], dtype=np.float32),
        confidence=np.array([[0.24, 0.84]], dtype=np.float32),
        evidence=np.array([[0.23, 0.86]], dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.array([[0.41, 0.62]], dtype=np.float32),
        confidence=np.array([[0.32, 0.64]], dtype=np.float32),
        evidence=np.array([[0.34, 0.66]], dtype=np.float32),
    )
    background_evidence = np.array([[0.95, 0.05]], dtype=np.float32)

    result = GreenScreenCompetitiveLayerComposer().compose(
        subject=subject,
        effect=effect,
        background_evidence=background_evidence,
    )

    assert np.array_equal(result.ownership.background, np.array([[1.0, 0.0]], dtype=np.float32))
    assert np.array_equal(result.ownership.subject, np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(result.ownership.effect, np.array([[0.0, 0.0]], dtype=np.float32))
    assert np.array_equal(
        result.debug_layers["background_evidence"],
        background_evidence,
    )
    assert np.allclose(result.final_alpha, np.array([[0.0, 0.82]], dtype=np.float32))


def test_background_evidence_does_not_override_high_confidence_subject_or_effect():
    subject = LayerCandidate(
        alpha=np.array([[0.93, 0.20]], dtype=np.float32),
        confidence=np.array([[0.94, 0.10]], dtype=np.float32),
        evidence=np.array([[0.95, 0.10]], dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.array([[0.20, 0.88]], dtype=np.float32),
        confidence=np.array([[0.10, 0.90]], dtype=np.float32),
        evidence=np.array([[0.10, 0.92]], dtype=np.float32),
    )
    background_evidence = np.array([[1.0, 1.0]], dtype=np.float32)

    result = GreenScreenCompetitiveLayerComposer().compose(
        subject=subject,
        effect=effect,
        background_evidence=background_evidence,
    )

    assert np.array_equal(result.ownership.background, np.array([[0.0, 0.0]], dtype=np.float32))
    assert np.array_equal(result.ownership.subject, np.array([[1.0, 0.0]], dtype=np.float32))
    assert np.array_equal(result.ownership.effect, np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.allclose(result.final_alpha, np.array([[0.93, 0.88]], dtype=np.float32))


def test_background_evidence_does_not_clear_high_support_subject_with_low_evidence():
    subject = LayerCandidate(
        alpha=np.array([[0.82]], dtype=np.float32),
        confidence=np.array([[0.88]], dtype=np.float32),
        evidence=np.array([[0.20]], dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.array([[0.0]], dtype=np.float32),
        confidence=np.array([[0.0]], dtype=np.float32),
        evidence=np.array([[0.0]], dtype=np.float32),
    )

    result = GreenScreenCompetitiveLayerComposer().compose(
        subject=subject,
        effect=effect,
        background_evidence=np.array([[1.0]], dtype=np.float32),
    )

    assert np.array_equal(result.ownership.background, np.array([[0.0]], dtype=np.float32))
    assert np.array_equal(result.ownership.subject, np.array([[1.0]], dtype=np.float32))
    assert np.allclose(result.final_alpha, np.array([[0.82]], dtype=np.float32))


def test_background_evidence_does_not_clear_high_confidence_effect_at_evidence_boundary():
    subject = LayerCandidate(
        alpha=np.array([[0.0]], dtype=np.float32),
        confidence=np.array([[0.0]], dtype=np.float32),
        evidence=np.array([[0.0]], dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.array([[0.86]], dtype=np.float32),
        confidence=np.array([[0.92]], dtype=np.float32),
        evidence=np.array([[0.85]], dtype=np.float32),
    )

    result = GreenScreenCompetitiveLayerComposer().compose(
        subject=subject,
        effect=effect,
        background_evidence=np.array([[1.0]], dtype=np.float32),
    )

    assert np.array_equal(result.ownership.background, np.array([[0.0]], dtype=np.float32))
    assert np.array_equal(result.ownership.effect, np.array([[1.0]], dtype=np.float32))
    assert np.allclose(result.final_alpha, np.array([[0.86]], dtype=np.float32))


def test_minimal_compose_contract_clips_alpha_and_confidence_inputs():
    subject = LayerCandidate(
        alpha=np.array([[1.5, -0.5]], dtype=np.float64),
        confidence=np.array([[2.0, -1.0]], dtype=np.float64),
    )
    effect = LayerCandidate(
        alpha=np.array([[-0.25, 1.25]], dtype=np.float64),
        confidence=np.array([[-0.5, 2.0]], dtype=np.float64),
    )

    result = GreenScreenCompetitiveLayerComposer().compose(subject=subject, effect=effect)

    assert np.array_equal(result.debug_layers["subject_alpha"], np.array([[1.0, 0.0]], dtype=np.float32))
    assert np.array_equal(result.debug_layers["subject_confidence"], np.array([[1.0, 0.0]], dtype=np.float32))
    assert np.array_equal(result.debug_layers["effect_alpha"], np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(result.debug_layers["effect_confidence"], np.array([[0.0, 1.0]], dtype=np.float32))
    assert np.array_equal(result.final_alpha, np.array([[1.0, 1.0]], dtype=np.float32))


def test_minimal_compose_contract_rejects_shape_mismatch():
    subject = LayerCandidate(
        alpha=np.zeros((2, 2), dtype=np.float32),
        confidence=np.zeros((2, 2), dtype=np.float32),
    )
    bad_effect = LayerCandidate(
        alpha=np.zeros((1, 2), dtype=np.float32),
        confidence=np.zeros((1, 2), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match=r"effect\.alpha shape \(1, 2\) does not match subject alpha shape \(2, 2\)",
    ):
        GreenScreenCompetitiveLayerComposer().compose(subject=subject, effect=bad_effect)


def test_minimal_compose_contract_rejects_candidate_confidence_shape_mismatch():
    subject = LayerCandidate(
        alpha=np.zeros((2, 2), dtype=np.float32),
        confidence=np.zeros((1, 2), dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.zeros((2, 2), dtype=np.float32),
        confidence=np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match=r"subject\.confidence shape \(1, 2\) does not match alpha shape \(2, 2\)",
    ):
        GreenScreenCompetitiveLayerComposer().compose(subject=subject, effect=effect)


def test_minimal_compose_contract_rejects_candidate_evidence_shape_mismatch():
    subject = LayerCandidate(
        alpha=np.zeros((2, 2), dtype=np.float32),
        confidence=np.zeros((2, 2), dtype=np.float32),
        evidence=np.zeros((1, 2), dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.zeros((2, 2), dtype=np.float32),
        confidence=np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match=r"subject\.evidence shape \(1, 2\) does not match alpha shape \(2, 2\)",
    ):
        GreenScreenCompetitiveLayerComposer().compose(subject=subject, effect=effect)


def test_minimal_compose_contract_rejects_background_evidence_shape_mismatch():
    subject = LayerCandidate(
        alpha=np.zeros((2, 2), dtype=np.float32),
        confidence=np.zeros((2, 2), dtype=np.float32),
    )
    effect = LayerCandidate(
        alpha=np.zeros((2, 2), dtype=np.float32),
        confidence=np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(
        ValueError,
        match=r"background_evidence shape \(1, 2\) does not match subject alpha shape \(2, 2\)",
    ):
        GreenScreenCompetitiveLayerComposer().compose(
            subject=subject,
            effect=effect,
            background_evidence=np.zeros((1, 2), dtype=np.float32),
        )
