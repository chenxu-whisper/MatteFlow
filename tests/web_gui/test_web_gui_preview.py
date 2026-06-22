import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import web_gui


def test_create_preview_frames_prefers_processed_rgba_over_comp(tmp_path):
    output_dir = tmp_path / "out"
    processed_dir = output_dir / "Processed"
    comp_dir = output_dir / "Comp"
    processed_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = [255, 200, 200]
    rgba[:, :, 3] = 255
    rgba[0, 0, :3] = [0, 0, 255]
    rgba[0, 0, 3] = 8
    Image.fromarray(rgba, mode="RGBA").save(processed_dir / "processed_000000.png")

    comp = np.zeros((4, 4, 3), dtype=np.uint8)
    comp[:, :, :] = [255, 200, 200]
    comp[0, 0, :] = [0, 0, 8]
    Image.fromarray(comp, mode="RGB").save(comp_dir / "comp_000000.png")
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB").save(tmp_path / "input.png")

    _, output_preview = web_gui._create_preview_frames(output_dir, tmp_path / "input.png")

    assert output_preview is not None
    assert output_preview[0, 0, 2] > output_preview[0, 0, 0]
    assert output_preview[0, 0, 2] > output_preview[0, 0, 1]


def test_create_upload_preview_returns_image_for_image_file(tmp_path):
    image_path = tmp_path / "upload.png"
    Image.fromarray(np.full((3, 4, 3), 128, dtype=np.uint8), mode="RGB").save(image_path)

    image_preview, video_preview, status = web_gui._create_upload_preview(str(image_path))

    assert image_preview["value"] == str(image_path)
    assert image_preview["visible"] is True
    assert video_preview["value"] is None
    assert video_preview["visible"] is False
    assert "图片" in status


def test_create_upload_preview_skips_image_for_video_file(tmp_path):
    video_path = tmp_path / "upload.mp4"
    video_path.write_bytes(b"not a real video")

    image_preview, video_preview, status = web_gui._create_upload_preview(str(video_path))

    assert image_preview["value"] is None
    assert image_preview["visible"] is False
    assert video_preview["value"] == str(video_path)
    assert video_preview["visible"] is True
    assert "视频" in status


def test_find_transparent_png_download_prefers_processed_rgba_first_frame(tmp_path):
    output_dir = tmp_path / "out"
    processed_dir = output_dir / "Processed"
    matte_dir = output_dir / "Matte"
    processed_dir.mkdir(parents=True)
    matte_dir.mkdir(parents=True)

    processed_path = processed_dir / "processed_000000.png"
    matte_path = matte_dir / "matte_000000.png"
    Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8), mode="RGBA").save(processed_path)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(matte_path)

    download_path = web_gui._find_transparent_png_download(output_dir)

    assert download_path == processed_path


def test_find_transparent_png_download_returns_none_without_processed_png(tmp_path):
    output_dir = tmp_path / "out"
    (output_dir / "Matte").mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(output_dir / "Matte" / "matte_000000.png")

    assert web_gui._find_transparent_png_download(output_dir) is None


def test_create_preview_video_closes_imageio_writer_when_frame_write_fails(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "out"
    processed_dir = output_dir / "Processed"
    processed_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8), mode="RGBA").save(
        processed_dir / "processed_000000.png"
    )
    Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8), mode="RGBA").save(
        processed_dir / "processed_000001.png"
    )
    closed = {"value": False}

    class FailingWriter:
        def append_data(self, frame):
            del frame
            raise RuntimeError("write failed")

        def close(self):
            closed["value"] = True

    monkeypatch.setitem(
        sys.modules,
        "imageio",
        SimpleNamespace(get_writer=lambda *args, **kwargs: FailingWriter()),
    )

    with pytest.raises(RuntimeError, match="write failed"):
        web_gui._create_preview_video(output_dir, tmp_path / "preview.mp4")

    assert closed["value"] is True


def test_create_preview_video_cv2_releases_writer_when_frame_write_fails(
    monkeypatch, tmp_path
):
    output_dir = tmp_path / "out"
    processed_dir = output_dir / "Processed"
    processed_dir.mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8), mode="RGBA").save(
        processed_dir / "processed_000000.png"
    )
    released = {"value": False}

    class FailingVideoWriter:
        def isOpened(self):
            return True

        def write(self, frame):
            del frame
            raise RuntimeError("cv2 write failed")

        def release(self):
            released["value"] = True

    monkeypatch.setattr(web_gui.cv2, "VideoWriter", lambda *args, **kwargs: FailingVideoWriter())

    with pytest.raises(RuntimeError, match="cv2 write failed"):
        web_gui._create_preview_video_cv2(output_dir, tmp_path / "preview.mp4")

    assert released["value"] is True
