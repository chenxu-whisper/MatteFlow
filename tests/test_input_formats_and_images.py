import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.input import decoder, formats
from matteflow.pipeline import MattingPipeline


def test_detect_input_kind_supports_video_image_and_sequence(tmp_path):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    (image_dir / "0001.png").write_bytes(b"fake")

    assert formats.detect_input_kind(Path("clip.webm")) == formats.InputKind.VIDEO
    assert formats.detect_input_kind(Path("photo.webp")) == formats.InputKind.IMAGE
    assert formats.detect_input_kind(image_dir) == formats.InputKind.SEQUENCE


def test_supported_extension_sets_cover_mainstream_formats():
    assert {".mp4", ".mov", ".avi", ".mkv", ".webm"} <= formats.VIDEO_EXTENSIONS
    assert {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr"} <= formats.IMAGE_EXTENSIONS


def test_image_decoder_loads_single_image_as_one_frame(tmp_path):
    image_path = tmp_path / "rabbit.webp"
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10
    rgb[:, :, 1] = 20
    rgb[:, :, 2] = 30
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(image_path), bgr)

    frames, meta = decoder.ImageDecoder().decode(image_path)

    assert len(frames) == 1
    assert frames[0].shape == (3, 4, 3)
    assert np.array_equal(frames[0][0, 0], np.array([10, 20, 30], dtype=np.uint8))
    assert meta["input_kind"] == "image"
    assert meta["total_frames"] == 1
    assert meta["source_files"] == ["rabbit.webp"]


def test_image_decoder_logs_path_format_and_shape(tmp_path, caplog):
    image_path = tmp_path / "rabbit.png"
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    caplog.set_level("INFO", logger="matteflow.input.decoder")
    decoder.ImageDecoder().decode(image_path)

    assert "Decoding single image" in caplog.text
    assert "rabbit.png" in caplog.text
    assert "format=.png" in caplog.text
    assert "width=4 height=3" in caplog.text


def test_sequence_decoder_logs_discovered_and_loaded_frames(tmp_path, caplog):
    image_dir = tmp_path / "frames"
    image_dir.mkdir()
    for index in range(2):
        image_path = image_dir / f"{index:04d}.jpg"
        rgb = np.full((2, 3, 3), index, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    caplog.set_level("INFO", logger="matteflow.input.decoder")
    decoder.SequenceDecoder().decode(image_dir)

    assert "Decoding image sequence" in caplog.text
    assert "discovered=2" in caplog.text
    assert "Loaded image sequence" in caplog.text
    assert "frames=2" in caplog.text


def test_video_decoder_logs_failure_for_unopenable_video(tmp_path, caplog):
    video_path = tmp_path / "broken.mp4"
    video_path.write_bytes(b"not a real video")

    caplog.set_level("INFO", logger="matteflow.input.decoder")
    with pytest.raises(ValueError, match="Cannot open video"):
        decoder.VideoDecoder().decode(video_path)

    assert "Decoding video" in caplog.text
    assert "Failed to open video" in caplog.text
    assert "broken.mp4" in caplog.text


def test_pipeline_decode_input_routes_single_image(tmp_path):
    image_path = tmp_path / "input.bmp"
    rgb = np.full((2, 3, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    pipeline = MattingPipeline.__new__(MattingPipeline)
    frames, meta = pipeline._decode_input(image_path)

    assert len(frames) == 1
    assert frames[0].shape == (2, 3, 3)
    assert meta["input_kind"] == "image"
