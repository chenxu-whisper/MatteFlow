from pathlib import Path
from typing import Iterable, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def models_root() -> Path:
    return project_root() / "models"


def model_file(filename: str) -> Path:
    return models_root() / filename


def resolve_snapshot_model_dir(
    root: Path, repo_id: str, required_subdirs: Iterable[str]
) -> Optional[Path]:
    flat_dir = root / repo_id.split("/")[-1]
    if all((flat_dir / child).is_dir() for child in required_subdirs):
        return flat_dir

    namespace, name = repo_id.split("/", 1)
    snapshots = root / f"models--{namespace}--{name}" / "snapshots"
    if snapshots.is_dir():
        for path in snapshots.iterdir():
            if path.is_dir() and all((path / child).is_dir() for child in required_subdirs):
                return path

    return None
