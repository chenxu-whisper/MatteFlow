import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig  # noqa: E402
from matteflow.matte.hybrid_matte import HybridMatte  # noqa: E402


def _apply_fusion_guard(hybrid: HybridMatte, base_alpha: np.ndarray, fused_alpha: np.ndarray):
    subject_ownership = np.ones(base_alpha.shape, dtype=np.float32)
    background_ownership = np.zeros(base_alpha.shape, dtype=np.float32)
    takeover_mask = fused_alpha < base_alpha - 0.50
    subject_ownership[takeover_mask] = 0.0
    background_ownership[takeover_mask] = 1.0
    layer_ownership = SimpleNamespace(
        subject=subject_ownership,
        effect=np.zeros(base_alpha.shape, dtype=np.float32),
        background=background_ownership,
    )
    confidence = np.ones(base_alpha.shape, dtype=np.float32)

    return hybrid._apply_green_screen_fusion_quality_gate(
        fused_alpha=fused_alpha,
        base_alpha=base_alpha,
        ai_alpha=fused_alpha,
        solid_alpha=fused_alpha,
        effect_alpha=np.zeros(base_alpha.shape, dtype=np.float32),
        subject_confidence=confidence,
        subject_competitive_confidence=confidence,
        effect_competitive_confidence=np.zeros(base_alpha.shape, dtype=np.float32),
        background_evidence=background_ownership,
        layer_ownership=layer_ownership,
        frame=None,
    )


def test_green_screen_fusion_guard_rolls_back_composer_hole_risk():
    hybrid = HybridMatte(MattingConfig(use_ai=False))
    base_alpha = np.ones((7, 7), dtype=np.float32)
    fused_alpha = np.ones((7, 7), dtype=np.float32)
    fused_alpha[2:5, 2:5] = 0.0

    guarded = _apply_fusion_guard(hybrid, base_alpha, fused_alpha)

    assert np.allclose(guarded, base_alpha)
    diagnostics = hybrid.last_fusion_quality_gate_diagnostics
    assert diagnostics["risk_guard"]["triggered"] is True
    assert diagnostics["risk_guard"]["fallback_candidate"] == "base_alpha"
    assert "hole_pixels" in diagnostics["risk_guard"]["reasons"]
    assert "edge_takeover" in diagnostics["risk_guard"]["reasons"]
    assert diagnostics["risk_guard"]["selected_hole_pixels"] == 9
    assert diagnostics["risk_guard"]["fallback_hole_pixels"] == 0


def test_green_screen_fusion_guard_triggers_when_hole_and_takeover_risk_cross_threshold():
    hybrid = HybridMatte(MattingConfig(use_ai=False))
    base_alpha = np.ones((7, 7), dtype=np.float32)
    fused_alpha = np.ones((7, 7), dtype=np.float32)
    fused_alpha[2:4, 2:5] = 0.0

    guarded = _apply_fusion_guard(hybrid, base_alpha, fused_alpha)

    assert np.allclose(guarded, base_alpha)
    risk_guard = hybrid.last_fusion_quality_gate_diagnostics["risk_guard"]
    assert risk_guard["triggered"] is True
    assert risk_guard["fallback_candidate"] == "base_alpha"
    assert risk_guard["reasons"] == ["hole_pixels", "overall_score", "edge_takeover"]
    assert risk_guard["selected_hole_pixels"] == 6
    assert risk_guard["fallback_hole_pixels"] == 0
