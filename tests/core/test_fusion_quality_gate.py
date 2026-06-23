import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.region_ownership import RegionOwnership  # noqa: E402
from matteflow.matte.fusion_quality_gate import FusionCandidate, FusionQualityGate  # noqa: E402


def _blank_ownership(shape, **overrides):
    values = {
        "subject": np.zeros(shape, dtype=bool),
        "hair_edge": np.zeros(shape, dtype=bool),
        "luminous_prop": np.zeros(shape, dtype=bool),
        "transparent_effect": np.zeros(shape, dtype=bool),
        "background_residue": np.zeros(shape, dtype=bool),
        "uncertain_edge": np.zeros(shape, dtype=bool),
    }
    values.update(overrides)
    return RegionOwnership(**values)


def test_quality_gate_prefers_high_confidence_subject_candidate_in_subject_region():
    subject_mask = np.array([[True, False, False]], dtype=bool)
    ownership = _blank_ownership(subject_mask.shape, subject=subject_mask)
    core = FusionCandidate(
        name="ai_core",
        alpha=np.array([[0.86, 0.10, 0.10]], dtype=np.float32),
        confidence=np.array([[0.95, 0.20, 0.20]], dtype=np.float32),
    )
    fallback = FusionCandidate(
        name="green_fallback",
        alpha=np.array([[0.35, 0.10, 0.10]], dtype=np.float32),
        confidence=np.array([[0.45, 0.20, 0.20]], dtype=np.float32),
    )

    result = FusionQualityGate().fuse([core, fallback], ownership)

    assert float(result.alpha[0, 0]) > 0.80
    assert result.diagnostics["selected_by_region"]["subject"] == "ai_core"


def test_quality_gate_protects_luminous_prop_from_lower_alpha_takeover():
    prop_mask = np.array([[False, True, False]], dtype=bool)
    ownership = _blank_ownership(prop_mask.shape, luminous_prop=prop_mask)
    repaired_prop = FusionCandidate(
        name="effect_prop_repair",
        alpha=np.array([[0.10, 0.98, 0.10]], dtype=np.float32),
        confidence=np.array([[0.20, 0.88, 0.20]], dtype=np.float32),
    )
    ai_candidate = FusionCandidate(
        name="gvm",
        alpha=np.array([[0.10, 0.22, 0.10]], dtype=np.float32),
        confidence=np.array([[0.20, 0.99, 0.20]], dtype=np.float32),
    )

    result = FusionQualityGate().fuse([repaired_prop, ai_candidate], ownership)

    assert float(result.alpha[0, 1]) >= 0.95
    assert result.diagnostics["selected_by_region"]["luminous_prop"] == "effect_prop_repair"
    assert result.diagnostics["rejected_takeovers"]["luminous_prop"] >= 1


def test_quality_gate_uses_fx_candidate_for_transparent_effect_region():
    effect_mask = np.array([[False, False, True]], dtype=bool)
    ownership = _blank_ownership(effect_mask.shape, transparent_effect=effect_mask)
    core = FusionCandidate(
        name="subject_core",
        alpha=np.array([[0.90, 0.10, 0.12]], dtype=np.float32),
        confidence=np.array([[0.90, 0.20, 0.25]], dtype=np.float32),
    )
    fx = FusionCandidate(
        name="fx_detail",
        alpha=np.array([[0.20, 0.10, 0.62]], dtype=np.float32),
        confidence=np.array([[0.20, 0.20, 0.92]], dtype=np.float32),
    )

    result = FusionQualityGate().fuse([core, fx], ownership)

    assert 0.55 <= float(result.alpha[0, 2]) <= 0.70
    assert result.diagnostics["selected_by_region"]["transparent_effect"] == "fx_detail"


def test_quality_gate_rejects_background_residue_false_positive_alpha():
    residue_mask = np.array([[False, True, False]], dtype=bool)
    ownership = _blank_ownership(residue_mask.shape, background_residue=residue_mask)
    base = FusionCandidate(
        name="green_key_base",
        alpha=np.array([[0.90, 0.02, 0.10]], dtype=np.float32),
        confidence=np.array([[0.80, 0.85, 0.30]], dtype=np.float32),
    )
    ai_candidate = FusionCandidate(
        name="ai_core",
        alpha=np.array([[0.90, 0.76, 0.10]], dtype=np.float32),
        confidence=np.array([[0.80, 0.95, 0.30]], dtype=np.float32),
    )

    result = FusionQualityGate().fuse([base, ai_candidate], ownership)

    assert float(result.alpha[0, 1]) <= 0.05
    assert result.diagnostics["selected_by_region"]["background_residue"] == "green_key_base"
    assert result.diagnostics["rejected_takeovers"]["background_residue"] >= 1
