import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.errors import ProgressCallbackError
from matteflow.config import BackgroundMode, MattingConfig
from matteflow.matte.hybrid_matte import HybridMatte
from matteflow.refine.color_decontaminate import ColorDecontaminate


def _load_rgb_video_crop(
    video_name: str,
    *,
    frame_index: int,
    top: int,
    left: int,
    height: int,
    width: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(str(PROJECT_ROOT / "assets" / "video" / video_name))
    assert capture.isOpened()
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = capture.read()
        assert ok and frame_bgr is not None
    finally:
        capture.release()

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb[top : top + height, left : left + width]


def test_progress_callback_error_is_exported():
    assert issubclass(ProgressCallbackError, Exception)


def test_green_screen_merge_preserves_high_confidence_non_screen_subject_pixels():
    matte = HybridMatte(MattingConfig(use_ai=False))
    frame = np.array([[[160, 100, 60]]], dtype=np.uint8)
    base_alpha = np.array([[0.96]], dtype=np.float32)
    ai_alpha = np.array([[0.95]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert merged[0, 0] >= 0.90


def test_green_despill_lifts_dark_low_alpha_glow_pixels():
    decontaminate = ColorDecontaminate(MattingConfig())
    frame = np.array([[[78, 62, 112]]], dtype=np.uint8)
    alpha = np.array([[0.05]], dtype=np.float32)

    result = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert float(result.mean()) > float(frame.mean())


def test_green_despill_neutralizes_bright_teal_halo_pixels():
    decontaminate = ColorDecontaminate(MattingConfig(key_color=(0, 255, 0)))
    frame = np.array([[[136, 203, 160]]], dtype=np.uint8)
    alpha = np.array([[0.43]], dtype=np.float32)

    result = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert int(result[0, 0, 1]) - int(result[0, 0, 0]) <= 10


def test_green_despill_neutralizes_teal_pixels_adjacent_to_white_ring():
    decontaminate = ColorDecontaminate(MattingConfig(key_color=(0, 255, 0)))
    frame = np.array(
        [
            [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
            [[0, 255, 0], [255, 255, 255], [96, 176, 180]],
            [[0, 255, 0], [0, 255, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )
    alpha = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.70],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert int(result[1, 2, 1]) - int(result[1, 2, 0]) <= 10


def test_green_despill_unmixes_moderate_alpha_pixels_from_screen_color():
    decontaminate = ColorDecontaminate(MattingConfig(key_color=(0, 255, 0)))
    foreground = np.array([220, 220, 220], dtype=np.float32)
    background = np.array([0, 255, 0], dtype=np.float32)
    alpha_value = 0.70
    mixed = (foreground * alpha_value + background * (1.0 - alpha_value)).astype(np.uint8)
    frame = mixed[None, None, :]
    alpha = np.array([[alpha_value]], dtype=np.float32)

    result = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert int(result[0, 0, 0]) >= 205
    assert int(result[0, 0, 1]) >= 205
    assert int(result[0, 0, 2]) >= 205


def test_white_ring_cleanup_strength_zero_preserves_more_teal_halo():
    frame = np.array([[[170, 182, 210]]], dtype=np.uint8)
    alpha = np.array([[0.43]], dtype=np.float32)

    disabled = ColorDecontaminate(
        MattingConfig(key_color=(0, 255, 0), white_ring_cleanup_strength=0.0)
    ).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]
    enabled = ColorDecontaminate(
        MattingConfig(key_color=(0, 255, 0), white_ring_cleanup_strength=1.0)
    ).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    disabled_gap = int(disabled[0, 0, 1]) - int(disabled[0, 0, 0])
    enabled_gap = int(enabled[0, 0, 1]) - int(enabled[0, 0, 0])

    assert disabled_gap > enabled_gap


def test_green_screen_merge_uses_ai_subject_signal_for_non_screen_subject_pixel():
    matte = HybridMatte(MattingConfig(use_ai=False))
    frame = np.array([[[120, 80, 110]]], dtype=np.uint8)
    base_alpha = np.array([[0.35]], dtype=np.float32)
    ai_alpha = np.array([[0.88]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert merged[0, 0] >= 0.75


def test_green_screen_merge_does_not_promote_white_ring_effect_to_solid_subject():
    matte = HybridMatte(MattingConfig(use_ai=False))
    frame = np.array([[[232, 240, 236]]], dtype=np.uint8)
    base_alpha = np.array([[0.18]], dtype=np.float32)
    ai_alpha = np.array([[0.99]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert 0.01 <= merged[0, 0] <= 0.35


def test_green_screen_merge_falls_back_to_base_when_gvm_subject_signal_is_degenerate():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    frame = np.array([[[190, 120, 200]]], dtype=np.uint8)
    base_alpha = np.array([[0.55]], dtype=np.float32)
    ai_alpha = np.array([[0.0]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert merged[0, 0] >= 0.50


def test_green_screen_merge_does_not_fallback_base_for_real_white_ring_effect():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    frame = np.array([[[232, 240, 236]]], dtype=np.uint8)
    base_alpha = np.array([[0.18]], dtype=np.float32)
    ai_alpha = np.array([[0.0]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert 0.01 <= merged[0, 0] <= 0.35


def test_green_screen_merge_recovers_soft_subject_when_fallback_is_score_blocked():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = np.array([[[170, 160, 175]]], dtype=np.uint8)
    base_alpha = np.array([[0.24]], dtype=np.float32)
    ai_alpha = np.array([[0.35]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert merged[0, 0] >= 0.40


def test_green_screen_merge_score_blocked_rescue_does_not_promote_white_ring_effect():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = np.array([[[232, 240, 236]]], dtype=np.uint8)
    base_alpha = np.array([[0.24]], dtype=np.float32)
    ai_alpha = np.array([[0.35]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert merged[0, 0] <= 0.35


def test_green_screen_merge_score_blocked_rescue_does_not_promote_pale_halo_pixels():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = np.array([[[190, 205, 195]]], dtype=np.uint8)
    base_alpha = np.array([[0.24]], dtype=np.float32)
    ai_alpha = np.array([[0.35]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert merged[0, 0] <= 0.35


def test_green_screen_merge_score_blocked_rescue_recovers_hair_edge_without_lifting_side_halo():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = np.array(
        [[
            [183, 193, 188],
            [176, 166, 181],
            [170, 160, 175],
            [176, 166, 181],
            [183, 193, 188],
        ]],
        dtype=np.uint8,
    )
    base_alpha = np.array([[0.22, 0.24, 0.26, 0.24, 0.22]], dtype=np.float32)
    ai_alpha = np.array([[0.35, 0.35, 0.35, 0.35, 0.35]], dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert merged[0, 2] >= 0.42
    assert merged[0, 0] <= 0.35
    assert merged[0, 4] <= 0.35
    assert merged[0, 2] - max(float(merged[0, 0]), float(merged[0, 4])) >= 0.08


def test_green_screen_merge_score_blocked_rescue_recovers_hair_patch_without_lifting_surrounding_halo():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = np.array(
        [
            [
                [188, 200, 193],
                [182, 193, 187],
                [179, 169, 183],
                [182, 193, 187],
                [188, 200, 193],
            ],
            [
                [184, 196, 189],
                [176, 166, 181],
                [170, 160, 175],
                [176, 166, 181],
                [184, 196, 189],
            ],
            [
                [188, 200, 193],
                [182, 193, 187],
                [179, 169, 183],
                [182, 193, 187],
                [188, 200, 193],
            ],
        ],
        dtype=np.uint8,
    )
    base_alpha = np.array(
        [
            [0.21, 0.22, 0.23, 0.22, 0.21],
            [0.21, 0.24, 0.27, 0.24, 0.21],
            [0.21, 0.22, 0.23, 0.22, 0.21],
        ],
        dtype=np.float32,
    )
    ai_alpha = np.full((3, 5), 0.35, dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert float(merged[1, 2]) >= 0.42
    assert float(merged[1, 2]) - float(merged[0, 0]) >= 0.08
    assert float(merged[1, 2]) - float(merged[0, 4]) >= 0.08
    assert float(merged[0, 0]) <= 0.35
    assert float(merged[0, 4]) <= 0.35


def test_green_screen_merge_score_blocked_rescue_keeps_diagonal_halo_below_hair_patch():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    halo = [179, 191, 184]
    hair = [170, 160, 175]
    frame = np.array(
        [
            [halo, halo, hair],
            [halo, hair, hair],
            [hair, hair, hair],
        ],
        dtype=np.uint8,
    )
    base_alpha = np.array(
        [
            [0.22, 0.22, 0.24],
            [0.22, 0.25, 0.26],
            [0.23, 0.25, 0.27],
        ],
        dtype=np.float32,
    )
    ai_alpha = np.full((3, 3), 0.35, dtype=np.float32)

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert float(merged[1, 1]) >= 0.42
    assert float(merged[0, 0]) <= 0.35
    assert float(merged[0, 1]) <= 0.35
    assert float(merged[1, 0]) <= 0.35
    assert float(merged[1, 1]) - float(merged[0, 0]) >= 0.08


def test_green_screen_score_blocked_rescue_matches_real_test_green_2_crop():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = _load_rgb_video_crop(
        "test_green_2.mp4",
        frame_index=0,
        top=360,
        left=846,
        height=5,
        width=5,
    )
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.full(base_alpha.shape, 0.35, dtype=np.float32)

    solid = matte._green_screen_solid_layer(base_alpha, frame)
    rescue = matte._green_screen_score_blocked_subject_layer(base_alpha, frame)
    merged_with_rescue = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    matte_without_rescue = HybridMatte(MattingConfig(use_ai=False))
    matte_without_rescue.last_active_ai_model = "gvm"
    matte_without_rescue.last_fallback_quality_metrics = None
    merged_without_rescue = matte_without_rescue._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert float(solid[2, 2]) <= 0.01
    assert float(rescue[2, 2]) >= 0.50
    assert float(rescue[4, 0]) >= 0.50
    assert float(rescue[0, 0]) <= 0.01
    assert float(rescue[0, 4]) <= 0.01
    assert float(rescue[4, 4]) <= 0.01
    assert float(merged_with_rescue[2, 2]) >= 0.50
    assert float(merged_with_rescue[2, 2] - merged_without_rescue[2, 2]) >= 0.50
    assert float(merged_with_rescue[4, 0] - merged_without_rescue[4, 0]) >= 0.50
    assert float(merged_with_rescue[0, 0] - merged_without_rescue[0, 0]) <= 0.01


def test_green_screen_score_blocked_rescue_skips_real_crop_when_base_is_already_solid():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = _load_rgb_video_crop(
        "test_green_4.mp4",
        frame_index=45,
        top=656,
        left=451,
        height=5,
        width=5,
    )
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.full(base_alpha.shape, 0.35, dtype=np.float32)

    solid = matte._green_screen_solid_layer(base_alpha, frame)
    rescue = matte._green_screen_score_blocked_subject_layer(base_alpha, frame)
    merged_with_rescue = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    matte_without_rescue = HybridMatte(MattingConfig(use_ai=False))
    matte_without_rescue.last_active_ai_model = "gvm"
    matte_without_rescue.last_fallback_quality_metrics = None
    merged_without_rescue = matte_without_rescue._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert np.all(solid == 1.0)
    assert float(np.max(rescue)) <= 0.01
    assert float(np.max(np.abs(merged_with_rescue - merged_without_rescue))) <= 1e-6


def test_green_screen_score_blocked_rescue_skips_real_bright_cool_effect_crop():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = _load_rgb_video_crop(
        "test_green_2.mp4",
        frame_index=60,
        top=212,
        left=1128,
        height=5,
        width=5,
    )
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.full(base_alpha.shape, 0.35, dtype=np.float32)

    solid = matte._green_screen_solid_layer(base_alpha, frame)
    rescue = matte._green_screen_score_blocked_subject_layer(base_alpha, frame)
    merged_with_rescue = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    matte_without_rescue = HybridMatte(MattingConfig(use_ai=False))
    matte_without_rescue.last_active_ai_model = "gvm"
    matte_without_rescue.last_fallback_quality_metrics = None
    merged_without_rescue = matte_without_rescue._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert float(solid[2, 2]) <= 0.01
    assert float(rescue[2, 2]) <= 0.01
    assert float(rescue[0, 4]) <= 0.01
    assert float(np.max(merged_with_rescue - merged_without_rescue)) <= 0.01


def test_green_screen_score_blocked_rescue_skips_real_cool_gray_transition_effect_crop():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame = _load_rgb_video_crop(
        "test_green_2.mp4",
        frame_index=60,
        top=781,
        left=1085,
        height=5,
        width=5,
    )
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.full(base_alpha.shape, 0.35, dtype=np.float32)

    solid = matte._green_screen_solid_layer(base_alpha, frame)
    rescue = matte._green_screen_score_blocked_subject_layer(base_alpha, frame)
    merged_with_rescue = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    matte_without_rescue = HybridMatte(MattingConfig(use_ai=False))
    matte_without_rescue.last_active_ai_model = "gvm"
    matte_without_rescue.last_fallback_quality_metrics = None
    merged_without_rescue = matte_without_rescue._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert float(solid[2, 2]) <= 0.01
    assert float(rescue[2, 2]) <= 0.01
    assert float(np.max(merged_with_rescue - merged_without_rescue)) <= 0.01


def test_green_screen_subject_integrity_recovers_real_purple_subject_holes():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame_bgr = cv2.imread(str(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"))
    assert frame_bgr is not None
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_f = frame.astype(np.float32, copy=False)
    purple_subject = (
        (frame_f[:, :, 0] > 120.0)
        & (frame_f[:, :, 2] > 130.0)
        & (frame_f[:, :, 1] < 180.0)
    )
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.where(purple_subject & (base_alpha >= 0.45), 1.0, 0.0039).astype(np.float32)
    low_base_subject_holes = (
        purple_subject
        & matte._green_screen_non_screen_mask(frame)
        & (base_alpha < 0.45)
    )

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert int(low_base_subject_holes.sum()) >= 50_000
    assert float((merged[low_base_subject_holes] < 0.20).mean()) <= 0.35
    assert float(merged[low_base_subject_holes].mean()) >= 0.45


def test_green_screen_subject_integrity_does_not_recover_real_blue_background_cloud():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame_bgr = cv2.imread(str(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"))
    assert frame_bgr is not None
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_f = frame.astype(np.float32, copy=False)
    base_alpha = matte.green_matte.generate(frame)
    purple_subject = (
        (frame_f[:, :, 0] > 120.0)
        & (frame_f[:, :, 2] > 130.0)
        & (frame_f[:, :, 1] < 180.0)
    )
    ai_alpha = np.where(purple_subject & (base_alpha >= 0.45), 1.0, 0.0039).astype(np.float32)
    blue_background_cloud = (
        (frame_f[:, :, 2] > 130.0)
        & (frame_f[:, :, 1] > 100.0)
        & (frame_f[:, :, 0] < 150.0)
        & (base_alpha < 0.20)
        & (ai_alpha < 0.20)
    )

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert int(blue_background_cloud.sum()) >= 300_000
    assert float((merged[blue_background_cloud] > 0.35).mean()) <= 0.10
    assert float(merged[blue_background_cloud].mean()) <= 0.18


def test_green_screen_effect_reconstruction_preserves_real_luminous_lightning_bands():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame_bgr = cv2.imread(str(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"))
    assert frame_bgr is not None
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_f = frame.astype(np.float32, copy=False)
    brightness = frame_f.mean(axis=2)
    chroma = frame_f.max(axis=2) - frame_f.min(axis=2)
    purple_subject = (
        (frame_f[:, :, 0] > 120.0)
        & (frame_f[:, :, 2] > 130.0)
        & (frame_f[:, :, 1] < 180.0)
    )
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.where(purple_subject & (base_alpha >= 0.45), 1.0, 0.0039).astype(np.float32)
    luminous_band_core = (
        (brightness > 205.0)
        & (chroma < 70.0)
        & (base_alpha < 0.75)
        & (~purple_subject)
    )

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert int(luminous_band_core.sum()) >= 50_000
    assert float((merged[luminous_band_core] < 0.20).mean()) <= 0.10
    assert float(merged[luminous_band_core].mean()) >= 0.70


def test_green_screen_effect_reconstruction_restores_cyan_halo_near_lightning_only():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame_bgr = cv2.imread(str(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"))
    assert frame_bgr is not None
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_f = frame.astype(np.float32, copy=False)
    red = frame_f[:, :, 0]
    green = frame_f[:, :, 1]
    blue = frame_f[:, :, 2]
    brightness = frame_f.mean(axis=2)
    chroma = frame_f.max(axis=2) - frame_f.min(axis=2)
    purple_subject = (red > 120.0) & (blue > 130.0) & (green < 180.0)
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.where(purple_subject & (base_alpha >= 0.45), 1.0, 0.0039).astype(np.float32)
    luminous_core = (
        (brightness > 205.0)
        & (chroma < 70.0)
        & (base_alpha < 0.75)
        & (~purple_subject)
    )
    core_reach = cv2.dilate(
        luminous_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
        iterations=1,
    ).astype(bool)
    cyan_halo_candidate = (
        (blue > 140.0)
        & (green > 120.0)
        & (red < 150.0)
        & (brightness > 120.0)
        & (base_alpha < 0.45)
        & (~purple_subject)
    )
    near_lightning_cyan_halo = core_reach & (~luminous_core) & cyan_halo_candidate
    far_cyan_background = (~core_reach) & cyan_halo_candidate

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert int(near_lightning_cyan_halo.sum()) >= 100_000
    assert int(far_cyan_background.sum()) >= 300_000
    assert float((merged[near_lightning_cyan_halo] < 0.20).mean()) <= 0.45
    assert float(merged[near_lightning_cyan_halo].mean()) >= 0.24
    assert float((merged[far_cyan_background] > 0.35).mean()) <= 0.08
    assert float(merged[far_cyan_background].mean()) <= 0.08


def test_green_screen_semantic_subject_trimap_guides_gvm_subject_without_lifting_blue_background():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    matte.last_fallback_quality_metrics = {
        "score_blocked": True,
        "effect_damage_blocked": False,
        "accepted": False,
    }
    frame_bgr = cv2.imread(str(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"))
    assert frame_bgr is not None
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_f = frame.astype(np.float32, copy=False)
    red = frame_f[:, :, 0]
    green = frame_f[:, :, 1]
    blue = frame_f[:, :, 2]
    purple_subject = (red > 120.0) & (blue > 130.0) & (green < 180.0)
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.where(purple_subject & (base_alpha >= 0.45), 1.0, 0.0039).astype(np.float32)
    semantic_subject_alpha = np.where(purple_subject, 1.0, 0.0).astype(np.float32)
    purple_holes = purple_subject & matte._green_screen_non_screen_mask(frame) & (base_alpha < 0.45)
    blue_background = (
        (blue > 130.0)
        & (green > 100.0)
        & (red < 150.0)
        & (base_alpha < 0.20)
        & (ai_alpha < 0.20)
        & (~purple_subject)
    )

    merged = matte._merge_green_screen_effects(
        [base_alpha],
        [ai_alpha],
        [frame],
        semantic_subject_alphas=[semantic_subject_alpha],
    )[0]

    assert int(purple_holes.sum()) >= 100_000
    assert int(blue_background.sum()) >= 300_000
    assert float((merged[purple_holes] < 0.20).mean()) <= 0.05
    assert float(merged[purple_holes].mean()) >= 0.78
    assert float((merged[blue_background] > 0.35).mean()) <= 0.08
    assert float(merged[blue_background].mean()) <= 0.12


def test_green_screen_gvm_builds_semantic_subject_trimap_from_birefnet_helper():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.birefnet = _SequenceEngine(0.75)

    semantic_alphas = matte._generate_green_screen_semantic_subject_alphas(
        "gvm",
        [np.zeros((3, 4, 3), dtype=np.uint8)],
        progress_callback=None,
        cancel_check=None,
    )

    assert semantic_alphas is not None
    assert len(semantic_alphas) == 1
    assert semantic_alphas[0].shape == (3, 4)
    assert float(semantic_alphas[0].mean()) == 0.75


def test_green_screen_gvm_merge_uses_competitive_layer_composer_and_records_debug():
    matte = HybridMatte(MattingConfig(use_ai=False, transparency_preserve=1.0))
    matte.last_active_ai_model = "gvm"
    frame = np.full((1, 2, 3), [120, 80, 110], dtype=np.uint8)
    base_alpha = np.array([[0.40, 0.20]], dtype=np.float32)
    ai_alpha = np.array([[0.85, 0.05]], dtype=np.float32)
    subject_gate = np.array([[0.90, 0.0]], dtype=np.float32)
    subject_alpha = np.array([[0.60, 0.0]], dtype=np.float32)
    effect_alpha = np.array([[0.80, 0.60]], dtype=np.float32)
    zero = np.zeros((1, 2), dtype=np.float32)
    matte._green_screen_subject_confidence = lambda *_args, **_kwargs: subject_gate
    matte._green_screen_ai_subject_layer = lambda *_args, **_kwargs: subject_alpha
    matte._green_screen_solid_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_score_blocked_subject_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_subject_integrity_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_semantic_subject_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_effect_layer = lambda *_args, **_kwargs: effect_alpha
    matte._green_screen_luminous_effect_reconstruction_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_gvm_fallback_subject_mask = lambda *_args, **_kwargs: np.zeros((1, 2), dtype=bool)
    matte._should_apply_gvm_subject_fallback = lambda *_args, **_kwargs: False
    matte._recover_degenerate_gvm_subject_alpha = lambda alpha, *_args, **_kwargs: alpha

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert np.allclose(merged, np.array([[0.60, 0.60]], dtype=np.float32))
    assert matte.last_green_screen_layer_debug is not None
    assert len(matte.last_green_screen_layer_debug) == 1
    assert np.allclose(
        matte.last_green_screen_layer_debug[0]["ownership_subject"],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    assert np.allclose(
        matte.last_green_screen_layer_debug[0]["ownership_effect"],
        np.array([[0.0, 1.0]], dtype=np.float32),
    )


def test_green_screen_non_gvm_merge_keeps_legacy_soft_fuse_and_no_layer_debug():
    matte = HybridMatte(MattingConfig(use_ai=False, transparency_preserve=1.0))
    matte.last_active_ai_model = "corridorkey"
    frame = np.full((1, 1, 3), [120, 80, 110], dtype=np.uint8)
    base_alpha = np.array([[0.40]], dtype=np.float32)
    ai_alpha = np.array([[0.85]], dtype=np.float32)
    subject_gate = np.array([[0.0]], dtype=np.float32)
    subject_alpha = np.array([[0.60]], dtype=np.float32)
    effect_alpha = np.array([[0.80]], dtype=np.float32)
    zero = np.zeros((1, 1), dtype=np.float32)
    matte.last_green_screen_layer_debug = [{"stale": zero}]
    matte._green_screen_subject_confidence = lambda *_args, **_kwargs: subject_gate
    matte._green_screen_ai_subject_layer = lambda *_args, **_kwargs: subject_alpha
    matte._green_screen_solid_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_score_blocked_subject_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_subject_integrity_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_semantic_subject_layer = lambda *_args, **_kwargs: zero
    matte._green_screen_effect_layer = lambda *_args, **_kwargs: effect_alpha
    matte._green_screen_luminous_effect_reconstruction_layer = lambda *_args, **_kwargs: zero

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert np.allclose(merged, np.array([[0.92]], dtype=np.float32))
    assert matte.last_green_screen_layer_debug is None


class _ConstantMatte:
    def __init__(self, value: float):
        self.value = value

    def generate(self, frame):
        return np.full(frame.shape[:2], self.value, dtype=np.float32)


class _SequenceEngine:
    def __init__(self, value: float):
        self.model = object()
        self.value = value

    def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
        del progress_callback, cancel_check
        return [np.full(frame.shape[:2], self.value, dtype=np.float32) for frame in frames]


def test_green_screen_sequence_falls_back_from_degenerate_gvm_to_corridorkey():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.config.use_ai = True
    matte.config.ai_model = "auto"
    matte.green_matte = _ConstantMatte(0.55)
    matte.gvm = _SequenceEngine(0.0)
    matte.corridorkey = _SequenceEngine(0.92)
    frames = [np.full((1, 1, 3), [190, 120, 200], dtype=np.uint8) for _ in range(3)]

    result = matte.generate_sequence(frames, BackgroundMode.GREEN_SCREEN)

    assert matte.last_active_ai_model == "corridorkey"
    assert min(float(alpha[0, 0]) for alpha in result) >= 0.80


def test_green_screen_sequence_keeps_healthy_gvm_without_fallback():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.config.use_ai = True
    matte.config.ai_model = "auto"
    matte.green_matte = _ConstantMatte(0.35)
    matte.gvm = _SequenceEngine(0.90)
    matte.corridorkey = _SequenceEngine(0.25)
    frames = [np.full((1, 1, 3), [120, 80, 110], dtype=np.uint8) for _ in range(3)]

    result = matte.generate_sequence(frames, BackgroundMode.GREEN_SCREEN)

    assert matte.last_active_ai_model == "gvm"
    assert min(float(alpha[0, 0]) for alpha in result) >= 0.75


def test_green_screen_sequence_lazy_loads_fallback_engine_after_degenerate_gvm():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.config.use_ai = True
    matte.config.ai_model = "gvm"
    matte.green_matte = _ConstantMatte(0.55)
    matte.gvm = _SequenceEngine(0.0)
    fallback_engine = _SequenceEngine(0.92)

    def _load_corridorkey(_config):
        matte.corridorkey = fallback_engine

    matte._load_corridorkey = _load_corridorkey
    frames = [np.full((1, 1, 3), [190, 120, 200], dtype=np.uint8) for _ in range(3)]

    result = matte.generate_sequence(frames, BackgroundMode.GREEN_SCREEN)

    assert matte.last_active_ai_model == "corridorkey"
    assert matte.corridorkey is fallback_engine
    assert min(float(alpha[0, 0]) for alpha in result) >= 0.80


def test_green_screen_sequence_keeps_gvm_when_fallback_engine_is_not_materially_better():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.config.use_ai = True
    matte.config.ai_model = "auto"
    matte.green_matte = _ConstantMatte(0.35)
    matte.gvm = _SequenceEngine(0.01)
    matte.corridorkey = _SequenceEngine(0.012)
    frames = [np.full((1, 1, 3), [180, 180, 180], dtype=np.uint8) for _ in range(3)]

    result = matte.generate_sequence(frames, BackgroundMode.GREEN_SCREEN)

    assert matte.last_active_ai_model == "gvm"
    assert len(result) == 3


def test_green_screen_sequence_falls_back_when_gvm_is_empty_but_base_has_broad_subject_support():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.config.use_ai = True
    matte.config.ai_model = "auto"
    matte.green_matte = _ConstantMatte(0.35)
    matte.gvm = _SequenceEngine(0.0)
    matte.corridorkey = _SequenceEngine(0.92)
    frames = [np.full((1, 1, 3), [180, 180, 180], dtype=np.uint8) for _ in range(3)]

    result = matte.generate_sequence(frames, BackgroundMode.GREEN_SCREEN)

    assert matte.last_active_ai_model == "corridorkey"
    assert min(float(alpha[0, 0]) for alpha in result) >= 0.80


def test_green_screen_sequence_falls_back_when_gvm_signal_stays_weak_across_broad_support_frames():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.config.use_ai = True
    matte.config.ai_model = "auto"
    matte.green_matte = _ConstantMatte(0.35)
    matte.gvm = _SequenceEngine(0.05)
    matte.corridorkey = _SequenceEngine(0.92)
    frames = [np.full((1, 1, 3), [180, 180, 180], dtype=np.uint8) for _ in range(3)]

    result = matte.generate_sequence(frames, BackgroundMode.GREEN_SCREEN)

    assert matte.last_active_ai_model == "corridorkey"
    assert min(float(alpha[0, 0]) for alpha in result) >= 0.80


def test_green_screen_sequence_falls_back_with_partial_broad_support_when_gvm_stays_weak():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.config.use_ai = True
    matte.config.ai_model = "auto"
    matte.green_matte = _ConstantMatte(0.35)
    matte.gvm = _SequenceEngine(0.05)
    matte.corridorkey = _SequenceEngine(0.92)
    frame = np.array([[[180, 180, 180], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0]]], dtype=np.uint8)
    frames = [frame.copy(), frame.copy(), frame.copy()]

    result = matte.generate_sequence(frames, BackgroundMode.GREEN_SCREEN)

    assert matte.last_active_ai_model == "corridorkey"
    assert min(float(alpha.max()) for alpha in result) >= 0.80


def _configure_region_weighted_quality_gate_masks(matte: HybridMatte) -> None:
    matte._green_screen_subject_confidence = lambda *_args, **_kwargs: np.array([[0.90, 0.20, 0.40]], dtype=np.float32)
    matte._smoothstep = lambda array, _low, _high: np.asarray(array, dtype=np.float32)
    matte._green_screen_effect_color_weight = lambda _frame: np.array([[0.10, 0.80, 0.10]], dtype=np.float32)


def test_region_weighted_fallback_accepts_subject_gain_even_when_global_mean_gain_is_small():
    matte = HybridMatte(MattingConfig(use_ai=False))
    _configure_region_weighted_quality_gate_masks(matte)
    frame = np.full((1, 3, 3), 180, dtype=np.uint8)
    base_alpha = np.array([[0.80, 0.70, 0.30]], dtype=np.float32)
    source_alpha = np.array([[0.40, 0.60, 0.30]], dtype=np.float32)
    fallback_alpha = np.array([[0.50, 0.58, 0.26]], dtype=np.float32)

    assert (float(fallback_alpha.mean()) - float(source_alpha.mean())) < 0.02
    assert matte._is_materially_better_fallback([source_alpha], [fallback_alpha], [base_alpha], [frame]) is True


def test_region_weighted_fallback_rejects_large_effect_damage_even_when_subject_gain_is_high():
    matte = HybridMatte(MattingConfig(use_ai=False))
    _configure_region_weighted_quality_gate_masks(matte)
    frame = np.full((1, 3, 3), 180, dtype=np.uint8)
    base_alpha = np.array([[0.80, 0.70, 0.30]], dtype=np.float32)
    source_alpha = np.array([[0.40, 0.60, 0.30]], dtype=np.float32)
    fallback_alpha = np.array([[0.55, 0.50, 0.30]], dtype=np.float32)

    assert matte._is_materially_better_fallback([source_alpha], [fallback_alpha], [base_alpha], [frame]) is False


def test_region_weighted_fallback_rejects_effect_only_gain_without_subject_improvement():
    matte = HybridMatte(MattingConfig(use_ai=False))
    _configure_region_weighted_quality_gate_masks(matte)
    frame = np.full((1, 3, 3), 180, dtype=np.uint8)
    base_alpha = np.array([[0.80, 0.70, 0.30]], dtype=np.float32)
    source_alpha = np.array([[0.40, 0.50, 0.30]], dtype=np.float32)
    fallback_alpha = np.array([[0.40, 0.60, 0.30]], dtype=np.float32)

    assert (float(fallback_alpha.mean()) - float(source_alpha.mean())) > 0.02
    assert matte._is_materially_better_fallback([source_alpha], [fallback_alpha], [base_alpha], [frame]) is False
