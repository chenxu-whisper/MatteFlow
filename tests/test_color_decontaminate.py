import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, MattingConfig
from matteflow.refine.color_decontaminate import ColorDecontaminate


def test_green_decontaminate_removes_green_from_translucent_glow():
    config = MattingConfig(green_despill_strength=0.8, edge_despill_factor=1.4)
    decontaminate = ColorDecontaminate(config)

    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    frame[:, :] = [245, 210, 225]
    frame[1, 1] = [150, 165, 140]
    alpha = np.full((3, 3), 0.35, dtype=np.float32)

    processed = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert processed[1, 1, 1] < frame[1, 1, 1] - 8
    assert processed[1, 1, 1] <= max(processed[1, 1, 0], processed[1, 1, 2]) + 3
    assert processed[1, 1, 0] >= frame[1, 1, 0]
    assert processed[1, 1, 2] >= frame[1, 1, 2]


def test_green_decontaminate_removes_low_alpha_cyan_green_haze():
    config = MattingConfig(green_despill_strength=0.9, edge_despill_factor=1.6)
    decontaminate = ColorDecontaminate(config)

    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    frame[:, :] = [245, 205, 225]
    frame[1, 1] = [118, 132, 128]
    alpha = np.full((3, 3), 0.025, dtype=np.float32)

    processed = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert processed[1, 1, 1] < frame[1, 1, 1] - 8
    assert processed[1, 1, 1] <= processed[1, 1, 0] + 4
    assert processed[1, 1, 0] >= frame[1, 1, 0]


def test_green_decontaminate_strongly_reduces_mid_alpha_glow_haze():
    config = MattingConfig(green_despill_strength=0.7, edge_despill_factor=1.2)
    decontaminate = ColorDecontaminate(config)

    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    frame[:, :] = [245, 205, 225]
    frame[1, 1] = [58, 103, 66]
    alpha = np.full((3, 3), 0.6, dtype=np.float32)

    processed = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert processed[1, 1, 1] <= max(processed[1, 1, 0], processed[1, 1, 2]) + 6
    assert processed[1, 1, 1] < frame[1, 1, 1] - 24


def test_green_decontaminate_lifts_dark_translucent_glow_halo():
    config = MattingConfig(green_despill_strength=0.7, edge_despill_factor=1.2)
    decontaminate = ColorDecontaminate(config)

    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    frame[:, :] = [245, 205, 225]
    frame[1, 1] = [56, 70, 86]
    alpha = np.full((3, 3), 0.24, dtype=np.float32)

    processed = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert processed[1, 1].mean() > frame[1, 1].mean() + 30
    assert processed[1, 1, 0] > frame[1, 1, 0]
    assert processed[1, 1, 2] >= frame[1, 1, 2]


def test_green_decontaminate_preserves_white_foreground():
    config = MattingConfig(green_despill_strength=0.8, edge_despill_factor=1.4)
    decontaminate = ColorDecontaminate(config)

    frame = np.full((2, 2, 3), 235, dtype=np.uint8)
    alpha = np.full((2, 2), 0.85, dtype=np.float32)

    processed = decontaminate.process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert np.abs(processed.astype(np.int16) - frame.astype(np.int16)).max() <= 2


def test_white_protect_parameters_control_green_despill_protection():
    frame = np.full((2, 2, 3), [205, 230, 210], dtype=np.uint8)
    alpha = np.full((2, 2), 0.45, dtype=np.float32)

    protected_config = MattingConfig(
        green_despill_strength=0.8,
        edge_despill_factor=1.4,
        white_protect_brightness=180,
        white_protect_saturation=90,
    )
    unprotected_config = MattingConfig(
        green_despill_strength=0.8,
        edge_despill_factor=1.4,
        white_protect_brightness=240,
        white_protect_saturation=10,
    )

    protected = ColorDecontaminate(protected_config).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]
    unprotected = ColorDecontaminate(unprotected_config).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert protected[0, 0, 1] > unprotected[0, 0, 1] + 10


def test_green_transparency_rgb_repair_lifts_dark_glow_without_overwriting_solid_subject():
    frame = np.full((3, 3, 3), [20, 150, 35], dtype=np.uint8)
    frame[1, 1] = [55, 70, 65]
    alpha = np.full((3, 3), 0.12, dtype=np.float32)
    alpha[0, 0] = 0.98

    repaired = ColorDecontaminate(MattingConfig()).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert repaired[1, 1, 0] > frame[1, 1, 0]
    assert repaired[1, 1, 2] > frame[1, 1, 2]
    assert np.abs(int(repaired[0, 0, 1]) - int(frame[0, 0, 1])) < 10


def test_black_transparency_rgb_repair_lifts_dim_particle_without_leaving_it_gray():
    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    frame[1, 1] = [30, 36, 33]
    alpha = np.full((3, 3), 0.08, dtype=np.float32)

    repaired = ColorDecontaminate(MattingConfig()).process([frame], [alpha], BackgroundMode.BLACK_BACKGROUND)[0]

    assert repaired[1, 1].mean() > frame[1, 1].mean() + 8
    assert (repaired[1, 1].max() - repaired[1, 1].min()) >= (frame[1, 1].max() - frame[1, 1].min()) + 2


def test_green_decontaminate_removes_green_haze_from_pink_glow_when_green_only_exceeds_blue():
    frame = np.full((3, 3, 3), [245, 185, 225], dtype=np.uint8)
    frame[1, 1] = [225, 215, 200]
    alpha = np.full((3, 3), 0.28, dtype=np.float32)

    processed = ColorDecontaminate(
        MattingConfig(green_despill_strength=0.8, edge_despill_factor=1.3)
    ).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert processed[1, 1, 1] < frame[1, 1, 1] - 8
    assert processed[1, 1, 2] > frame[1, 1, 2]
    assert processed[1, 1, 1] <= processed[1, 1, 0] + 2


def test_green_decontaminate_removes_softer_pink_glow_haze_below_previous_brightness_gate():
    frame = np.full((3, 3, 3), [235, 178, 220], dtype=np.uint8)
    frame[1, 1] = [170, 160, 150]
    alpha = np.full((3, 3), 0.24, dtype=np.float32)

    processed = ColorDecontaminate(
        MattingConfig(green_despill_strength=0.8, edge_despill_factor=1.3)
    ).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert processed[1, 1, 1] < frame[1, 1, 1] - 6
    assert processed[1, 1, 2] >= frame[1, 1, 2]
    assert processed[1, 1, 1] <= processed[1, 1, 0] + 2
