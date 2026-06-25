import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig  # noqa: E402
from matteflow.matte.hybrid_matte import HybridMatte  # noqa: E402


def test_green_screen_fusion_guard_rolls_back_composer_hole_risk():
    hybrid = HybridMatte(MattingConfig(use_ai=False))
    base_alpha = np.ones((7, 7), dtype=np.float32)
    fused_alpha = np.ones((7, 7), dtype=np.float32)
    fused_alpha[2:5, 2:5] = 0.0
    subject_ownership = np.ones((7, 7), dtype=np.float32)
    background_ownership = np.zeros((7, 7), dtype=np.float32)
    subject_ownership[2:5, 2:5] = 0.0
    background_ownership[2:5, 2:5] = 1.0
    layer_ownership = SimpleNamespace(
        subject=subject_ownership,
        effect=np.zeros((7, 7), dtype=np.float32),
        background=background_ownership,
    )
    confidence = np.ones((7, 7), dtype=np.float32)

    guarded = hybrid._apply_green_screen_fusion_quality_gate(
        fused_alpha=fused_alpha,
        base_alpha=base_alpha,
        ai_alpha=fused_alpha,
        solid_alpha=fused_alpha,
        effect_alpha=np.zeros((7, 7), dtype=np.float32),
        subject_confidence=confidence,
        subject_competitive_confidence=confidence,
        effect_competitive_confidence=np.zeros((7, 7), dtype=np.float32),
        background_evidence=background_ownership,
        layer_ownership=layer_ownership,
        frame=None,
    )

    assert np.allclose(guarded, base_alpha)
    diagnostics = hybrid.last_fusion_quality_gate_diagnostics
    assert diagnostics["risk_guard"]["triggered"] is True
    assert diagnostics["risk_guard"]["fallback_candidate"] == "base_alpha"
    assert "hole_pixels" in diagnostics["risk_guard"]["reasons"]
    assert "edge_takeover" in diagnostics["risk_guard"]["reasons"]
    assert diagnostics["risk_guard"]["selected_hole_pixels"] == 9
    assert diagnostics["risk_guard"]["fallback_hole_pixels"] == 0
