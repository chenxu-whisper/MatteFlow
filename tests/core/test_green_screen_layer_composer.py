import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.matte.green_screen_layer_composer import GreenScreenLayerComposer


def test_transparency_preserve_zero_returns_ai_alpha_without_effect_recovery():
    composer = GreenScreenLayerComposer(transparency_preserve=0.0)
    base_alpha = np.array([[0.9]], dtype=np.float32)
    ai_alpha = np.array([[0.2]], dtype=np.float32)
    effect_alpha = np.array([[1.0]], dtype=np.float32)

    result = composer.compose(base_alpha, ai_alpha, effect_alpha=effect_alpha)

    assert np.isclose(result[0, 0], 0.2)


def test_compose_recovers_effects_with_screen_blend_without_overwriting_subjects():
    composer = GreenScreenLayerComposer(transparency_preserve=0.7)
    base_alpha = np.array([[0.0, 0.0]], dtype=np.float32)
    ai_alpha = np.array([[0.8, 0.2]], dtype=np.float32)
    effect_alpha = np.array([[0.9, 0.9]], dtype=np.float32)
    subject_confidence = np.array([[1.0, 0.0]], dtype=np.float32)

    result = composer.compose(
        base_alpha,
        ai_alpha,
        effect_alpha=effect_alpha,
        subject_confidence=subject_confidence,
    )

    assert np.allclose(result, np.array([[0.8, 0.704]], dtype=np.float32))


def test_semantic_subject_alpha_strengthens_only_non_screen_subject_pixels():
    composer = GreenScreenLayerComposer()
    base_alpha = np.array([[0.0, 0.0]], dtype=np.float32)
    ai_alpha = np.array([[0.1, 0.1]], dtype=np.float32)
    semantic_subject_alpha = np.array([[1.0, 1.0]], dtype=np.float32)
    non_screen_mask = np.array([[True, False]])

    result = composer.compose(
        base_alpha,
        ai_alpha,
        semantic_subject_alpha=semantic_subject_alpha,
        non_screen_mask=non_screen_mask,
    )

    assert float(result[0, 0]) >= 0.90
    assert np.isclose(result[0, 1], 0.1)


def test_compose_sequence_preserves_frame_order_and_validates_lengths():
    composer = GreenScreenLayerComposer()
    base_alphas = [
        np.array([[0.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
    ]
    ai_alphas = [
        np.array([[0.2]], dtype=np.float32),
        np.array([[0.4]], dtype=np.float32),
    ]

    result = composer.compose_sequence(base_alphas, ai_alphas)

    assert np.allclose([alpha[0, 0] for alpha in result], [0.2, 0.4])

    try:
        composer.compose_sequence(base_alphas, ai_alphas[:1])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected mismatched sequence lengths to raise ValueError")
