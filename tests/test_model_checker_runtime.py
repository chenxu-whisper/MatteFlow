import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.utils.model_paths import model_file, models_root, resolve_snapshot_model_dir


def test_model_file_points_into_project_models_dir():
    path = model_file("corridorkey.pth")

    assert path.name == "corridorkey.pth"
    assert path.parent == models_root()


def test_resolve_snapshot_model_dir_prefers_snapshot_layout(tmp_path):
    snapshot = tmp_path / "models--foo--bar" / "snapshots" / "123"
    (snapshot / "unet").mkdir(parents=True)
    (snapshot / "vae").mkdir(parents=True)
    (snapshot / "scheduler").mkdir(parents=True)

    assert resolve_snapshot_model_dir(tmp_path, "foo/bar", ("unet", "vae", "scheduler")) == snapshot
