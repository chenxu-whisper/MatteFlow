import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.utils.model_paths import resolve_snapshot_repo_dir


def test_resolve_snapshot_repo_dir_returns_snapshot_directory(tmp_path):
    snapshot_dir = (
        tmp_path
        / "models--facebook--sam2-hiera-base-plus"
        / "snapshots"
        / "abc123"
    )
    snapshot_dir.mkdir(parents=True)

    result = resolve_snapshot_repo_dir(tmp_path, "facebook/sam2-hiera-base-plus")

    assert result == snapshot_dir


def test_resolve_snapshot_repo_dir_returns_flat_repo_directory(tmp_path):
    flat_dir = tmp_path / "BiRefNet"
    flat_dir.mkdir()

    result = resolve_snapshot_repo_dir(tmp_path, "ZhengPeng7/BiRefNet")

    assert result == flat_dir
