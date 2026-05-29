import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import web_gui


def test_resolve_gui_output_dir_uses_input_stem(tmp_path):
    output_root = tmp_path / "temp" / "output"

    result = web_gui._resolve_gui_output_dir(
        video_path=Path("assets/video/test green 2.mp4"),
        output_root=output_root,
    )

    assert result == output_root / "test_green_2"
    assert result.exists()


def test_resolve_gui_output_dir_avoids_overwriting_non_empty_directory(tmp_path):
    output_root = tmp_path / "temp" / "output"
    existing = output_root / "test_green_2"
    existing.mkdir(parents=True, exist_ok=True)
    (existing / "done.txt").write_text("keep", encoding="utf-8")

    result = web_gui._resolve_gui_output_dir(
        video_path=Path("assets/video/test_green_2.mp4"),
        output_root=output_root,
    )

    assert result.parent == output_root
    assert result.name.startswith("test_green_2_")
    assert result.exists()
