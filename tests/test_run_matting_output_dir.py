import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_matting


def test_resolve_output_dir_uses_project_temp_output_when_not_provided(tmp_path):
    result = run_matting._resolve_output_dir(
        input_path=Path("assets/video/test_green_2.mp4"),
        output_arg=None,
        project_root=tmp_path,
    )

    assert result == tmp_path / "temp" / "output" / "test_green_2"


def test_resolve_output_dir_preserves_explicit_output_arg(tmp_path):
    explicit = tmp_path / "custom" / "dir"

    result = run_matting._resolve_output_dir(
        input_path=Path("assets/video/test_green_2.mp4"),
        output_arg=str(explicit),
        project_root=tmp_path,
    )

    assert result == explicit
