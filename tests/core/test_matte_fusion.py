import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.matte.matte_fusion import MatteFusion


def test_matte_fusion_keeps_legacy_weighted_average_without_confidence_maps():
    fusion = MatteFusion()
    core = np.array([[1.0, 0.0]], dtype=np.float32)
    detail = np.array([[0.0, 1.0]], dtype=np.float32)

    result = fusion.fuse(core, detail_alpha=detail, weights=(0.75, 0.25, 0.0))

    assert np.allclose(result, np.array([[0.75, 0.25]], dtype=np.float32))


def test_matte_fusion_uses_per_pixel_confidence_to_select_regional_strengths():
    fusion = MatteFusion()
    core = np.array([[0.90, 0.20, 0.10]], dtype=np.float32)
    detail = np.array([[0.20, 0.80, 0.10]], dtype=np.float32)
    fx = np.array([[0.10, 0.30, 0.85]], dtype=np.float32)
    core_conf = np.array([[0.95, 0.10, 0.10]], dtype=np.float32)
    detail_conf = np.array([[0.10, 0.90, 0.10]], dtype=np.float32)
    fx_conf = np.array([[0.10, 0.20, 0.95]], dtype=np.float32)

    result = fusion.fuse(
        core,
        detail_alpha=detail,
        fx_alpha=fx,
        core_confidence=core_conf,
        detail_confidence=detail_conf,
        fx_confidence=fx_conf,
    )

    assert float(result[0, 0]) > 0.75
    assert float(result[0, 1]) > 0.65
    assert float(result[0, 2]) > 0.70
