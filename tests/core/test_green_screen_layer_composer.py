import sys
from pathlib import Path

import numpy as np

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
        "subject_alpha_out",
        "effect_alpha_out",
        "final_alpha",
        "ownership_subject",
        "ownership_effect",
        "ownership_background",
    }
    assert all(layer.shape == (2, 3) for layer in result.debug_layers.values())
