import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.utils.output_paths import resolve_project_output_dir


def test_resolve_project_output_dir_accepts_job_token(tmp_path):
    input_path = tmp_path / "input frame.png"
    input_path.write_text("x", encoding="utf-8")

    output_dir = resolve_project_output_dir(
        input_path,
        project_root=tmp_path,
        output_root=tmp_path / "out",
        job_token="job123",
    )

    assert output_dir.name == "input_frame_job123"
    assert output_dir.is_dir()
