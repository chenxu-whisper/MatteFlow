import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.auto_params import suggest_input_params, suggest_single_frame_params
from matteflow.config import MattingConfig


def test_suggest_single_frame_params_detects_blue_screen(tmp_path):
    image_path = tmp_path / "blue.png"
    frame = np.full((24, 24, 3), [20, 40, 210], dtype=np.uint8)
    frame[8:16, 8:16] = [230, 225, 235]
    Image.fromarray(frame, mode="RGB").save(image_path)

    suggestion = suggest_single_frame_params(image_path, MattingConfig())

    assert suggestion.params["screen_color"] == "blue"
    assert "screen=blue" in suggestion.summary


def test_suggest_single_frame_params_protects_bright_white_subject(tmp_path):
    image_path = tmp_path / "white_subject.png"
    frame = np.full((32, 32, 3), [20, 185, 55], dtype=np.uint8)
    frame[8:24, 8:24] = [225, 226, 238]
    Image.fromarray(frame, mode="RGB").save(image_path)

    suggestion = suggest_single_frame_params(image_path, MattingConfig())

    assert suggestion.params["key_strength"] <= 0.95
    assert suggestion.params["white_protect_brightness"] <= 175
    assert suggestion.params["white_protect_saturation"] >= 40


def test_suggest_single_frame_params_preserves_pink_glow(tmp_path):
    image_path = tmp_path / "pink_glow.png"
    frame = np.full((32, 32, 3), [20, 185, 55], dtype=np.uint8)
    frame[6:26, 6:26] = [245, 180, 220]
    Image.fromarray(frame, mode="RGB").save(image_path)

    suggestion = suggest_single_frame_params(image_path, MattingConfig())

    assert suggestion.params["transparency_preserve"] >= 0.72
    assert suggestion.params["clip_black"] == 0.0


def test_suggest_input_params_uses_middle_video_frame(tmp_path):
    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (16, 16),
    )
    for color in ([20, 40, 210], [20, 185, 55], [20, 40, 210]):
        frame_rgb = np.full((16, 16, 3), color, dtype=np.uint8)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()

    suggestion = suggest_input_params(video_path, MattingConfig())

    assert suggestion.params["screen_color"] == "green"
    assert "sample=video_middle" in suggestion.summary


def test_suggest_input_params_uses_middle_sequence_frame(tmp_path):
    sequence_dir = tmp_path / "seq"
    sequence_dir.mkdir()
    colors = ([20, 40, 210], [20, 185, 55], [20, 40, 210])
    for index, color in enumerate(colors):
        Image.fromarray(np.full((16, 16, 3), color, dtype=np.uint8), mode="RGB").save(
            sequence_dir / f"frame_{index:03d}.png"
        )

    suggestion = suggest_input_params(sequence_dir, MattingConfig())

    assert suggestion.params["screen_color"] == "green"
    assert "sample=sequence_middle" in suggestion.summary
