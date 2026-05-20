import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.ffmpeg_env import discover_ffmpeg, discover_media_tools


def test_discover_ffmpeg_prefers_existing_path():
    result = discover_ffmpeg(
        which=lambda name: "C:/ffmpeg/bin/ffmpeg.exe",
        path_exists=lambda path: True,
        imageio_ffmpeg_getter=lambda: None,
        common_candidate_dirs=[],
    )

    assert result.found is True
    assert result.source == "path"
    assert result.ffmpeg_path == "C:/ffmpeg/bin/ffmpeg.exe"
    assert result.bin_dir == "C:/ffmpeg/bin"


def test_discover_ffmpeg_uses_common_windows_directory():
    result = discover_ffmpeg(
        which=lambda name: None,
        path_exists=lambda path: path == "C:/Program Files/ffmpeg/bin/ffmpeg.exe",
        imageio_ffmpeg_getter=lambda: None,
        common_candidate_dirs=["C:/Program Files/ffmpeg/bin"],
    )

    assert result.found is True
    assert result.source == "common_dir"
    assert result.ffmpeg_path == "C:/Program Files/ffmpeg/bin/ffmpeg.exe"
    assert result.bin_dir == "C:/Program Files/ffmpeg/bin"


def test_discover_ffmpeg_falls_back_to_imageio_binary():
    result = discover_ffmpeg(
        which=lambda name: None,
        path_exists=lambda path: path == "C:/Users/Admin/AppData/Local/imageio/ffmpeg.exe",
        imageio_ffmpeg_getter=lambda: "C:/Users/Admin/AppData/Local/imageio/ffmpeg.exe",
        common_candidate_dirs=[],
    )

    assert result.found is True
    assert result.source == "imageio_ffmpeg"
    assert result.ffmpeg_path == "C:/Users/Admin/AppData/Local/imageio/ffmpeg.exe"
    assert result.bin_dir == "C:/Users/Admin/AppData/Local/imageio"


def test_discover_ffmpeg_reports_not_found():
    result = discover_ffmpeg(
        which=lambda name: None,
        path_exists=lambda path: False,
        imageio_ffmpeg_getter=lambda: None,
        common_candidate_dirs=[],
    )

    assert result.found is False
    assert result.source is None
    assert result.ffmpeg_path is None
    assert result.bin_dir is None


def test_discover_media_tools_prefers_complete_path_pair():
    result = discover_media_tools(
        ffmpeg_which=lambda name: "C:/ffmpeg/bin/ffmpeg.exe",
        ffprobe_which=lambda name: "C:/ffmpeg/bin/ffprobe.exe",
        path_exists=lambda path: True,
        imageio_ffmpeg_getter=lambda: None,
        common_candidate_dirs=[],
    )

    assert result.complete is True
    assert result.download_required is False
    assert result.source == "path"
    assert result.ffmpeg_path == "C:/ffmpeg/bin/ffmpeg.exe"
    assert result.ffprobe_path == "C:/ffmpeg/bin/ffprobe.exe"
    assert result.bin_dir == "C:/ffmpeg/bin"


def test_discover_media_tools_reports_download_required_for_imageio_only():
    result = discover_media_tools(
        ffmpeg_which=lambda name: None,
        ffprobe_which=lambda name: None,
        path_exists=lambda path: path == "C:/Users/Admin/AppData/Local/imageio/ffmpeg.exe",
        imageio_ffmpeg_getter=lambda: "C:/Users/Admin/AppData/Local/imageio/ffmpeg.exe",
        common_candidate_dirs=[],
    )

    assert result.complete is False
    assert result.download_required is True
    assert result.source == "imageio_ffmpeg"
    assert result.ffmpeg_path == "C:/Users/Admin/AppData/Local/imageio/ffmpeg.exe"
    assert result.ffprobe_path is None
    assert result.bin_dir == "C:/Users/Admin/AppData/Local/imageio"
