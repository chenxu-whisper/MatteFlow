import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig
from matteflow.matte.hybrid_matte import HybridMatte


def test_soft_fusion_adds_effect_only_into_remaining_alpha_space():
    matte = HybridMatte(MattingConfig(use_ai=False))

    solid = np.array([[0.90, 0.20]], dtype=np.float32)
    effect = np.array([[0.80, 0.60]], dtype=np.float32)

    fused = matte._soft_fuse_layers(solid, effect)

    assert np.isclose(fused[0, 0], 0.98, atol=1e-4)
    assert np.isclose(fused[0, 1], 0.68, atol=1e-4)
    assert np.all(fused >= solid)
    assert np.all(fused <= 1.0)


def test_green_screen_effect_layer_preserves_soft_glow_without_forcing_full_opacity():
    matte = HybridMatte(MattingConfig(use_ai=False, transparency_preserve=0.8))

    frame = np.full((6, 6, 3), [20, 185, 55], dtype=np.uint8)
    frame[2:4, 2:4] = [245, 180, 220]

    base_alpha = np.full((6, 6), 0.10, dtype=np.float32)
    base_alpha[2:4, 2:4] = 0.55
    ai_alpha = np.zeros((6, 6), dtype=np.float32)

    fused = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert fused[2, 2] > 0.20
    assert fused[2, 2] < 0.95
    assert fused[0, 0] < 0.05


def test_black_background_effect_layer_keeps_bright_particles():
    matte = HybridMatte(MattingConfig(use_ai=False))

    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    frame[2:4, 2:4] = [210, 170, 90]

    base_alpha = np.zeros((6, 6), dtype=np.float32)
    base_alpha[2:4, 2:4] = 0.18
    ai_alpha = np.zeros((6, 6), dtype=np.float32)

    fused = matte._merge_black_background_effects([base_alpha], [ai_alpha], [frame])[0]

    assert fused[2, 2] > 0.15
    assert fused[2, 2] < 0.80
    assert fused[0, 0] == 0.0


def test_green_screen_does_not_promote_pink_glow_to_solid_alpha_when_ai_is_overconfident():
    matte = HybridMatte(MattingConfig(use_ai=False, transparency_preserve=0.8))

    frame = np.full((3, 3, 3), [20, 185, 55], dtype=np.uint8)
    frame[1, 1] = [245, 180, 220]
    base_alpha = np.full((3, 3), 0.05, dtype=np.float32)
    base_alpha[1, 1] = 0.35
    ai_alpha = np.zeros((3, 3), dtype=np.float32)
    ai_alpha[1, 1] = 1.0

    fused = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert fused[1, 1] < 0.95
    assert fused[1, 1] > 0.18
    assert fused[0, 0] == 0.0


def test_green_screen_effect_weight_penalizes_green_haze_around_pink_white_glow():
    matte = HybridMatte(MattingConfig(use_ai=False))

    frame = np.array(
        [
            [[245, 180, 220], [220, 225, 200]],
        ],
        dtype=np.uint8,
    )

    weights = matte._green_screen_effect_color_weight(frame)

    assert weights[0, 0] > 0.70
    assert weights[0, 1] < 0.25


def test_green_screen_effect_layer_further_suppresses_green_haze_than_pink_glow():
    matte = HybridMatte(MattingConfig(use_ai=False))

    frame = np.array(
        [
            [[20, 185, 55], [245, 180, 220], [220, 225, 200]],
        ],
        dtype=np.uint8,
    )
    base_alpha = np.array([[0.05, 0.45, 0.45]], dtype=np.float32)

    effect_alpha = matte._green_screen_effect_layer(base_alpha, frame)

    assert effect_alpha[0, 1] > 0.30
    assert effect_alpha[0, 2] < 0.08


def test_green_screen_effect_layer_heart_roi_suppresses_outer_green_haze_on_test_frame():
    matte = HybridMatte(MattingConfig(use_ai=False))

    frame = np.array(
        Image.open(PROJECT_ROOT / "assets" / "frame" / "test_frame_2.png").convert("RGB")
    )
    base_alpha = matte.green_matte.generate(frame)
    effect_alpha = matte._green_screen_effect_layer(base_alpha, frame)

    pink_glow_alpha = float(effect_alpha[449, 376])
    outer_haze_alpha = float(effect_alpha[402, 447])

    assert pink_glow_alpha > 0.70
    assert outer_haze_alpha < 0.14


def test_green_screen_preserves_soft_neutral_subject_edges_like_rabbit_ears():
    matte = HybridMatte(MattingConfig(use_ai=False, transparency_preserve=0.8))

    frame = np.full((3, 3, 3), [30, 127, 59], dtype=np.uint8)
    frame[1, 1] = [140, 169, 160]
    base_alpha = np.full((3, 3), 0.05, dtype=np.float32)
    base_alpha[1, 1] = 0.88
    ai_alpha = np.zeros((3, 3), dtype=np.float32)

    fused = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert fused[1, 1] > 0.55
    assert fused[0, 0] == 0.0


def test_green_screen_effect_layer_preserves_bright_white_ring_on_test_frame_1():
    matte = HybridMatte(MattingConfig(use_ai=False))

    frame = np.array(
        Image.open(PROJECT_ROOT / "assets" / "frame" / "test_frame_1.png").convert("RGB")
    )
    base_alpha = matte.green_matte.generate(frame)
    effect_alpha = matte._green_screen_effect_layer(base_alpha, frame)

    bright_ring_alpha = float(effect_alpha[276, 192])
    green_bg_alpha = float(effect_alpha[40, 40])

    assert bright_ring_alpha > 0.10
    assert green_bg_alpha == 0.0
