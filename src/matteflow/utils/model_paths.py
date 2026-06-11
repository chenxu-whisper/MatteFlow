from pathlib import Path
from typing import Iterable, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _dir_has_entries(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def _main_project_root(current_root: Path) -> Optional[Path]:
    parts = current_root.parts
    if ".worktrees" not in parts:
        return None
    index = parts.index(".worktrees")
    if index <= 0:
        return None
    return Path(*parts[:index])


def models_root() -> Path:
    current_root = project_root()
    local_models = current_root / "models"
    if _dir_has_entries(local_models):
        return local_models

    main_root = _main_project_root(current_root)
    if main_root is not None:
        main_models = main_root / "models"
        if _dir_has_entries(main_models):
            return main_models

    return local_models


def model_file(filename: str) -> Path:
    primary = models_root() / filename
    if primary.exists():
        return primary

    main_root = _main_project_root(project_root())
    if main_root is not None:
        fallback = main_root / "models" / filename
        if fallback.exists():
            return fallback

    return primary


def resolve_snapshot_repo_dir(root: Path, repo_id: str) -> Optional[Path]:
    flat_dir = root / repo_id.split("/")[-1]
    if flat_dir.is_dir():
        return flat_dir

    namespace, name = repo_id.split("/", 1)
    snapshots = root / f"models--{namespace}--{name}" / "snapshots"
    if snapshots.is_dir():
        for path in sorted(snapshots.iterdir()):
            if path.is_dir():
                return path

    current_root = project_root()
    main_root = _main_project_root(current_root)
    if main_root is not None:
        fallback_root = main_root / "models"
        if fallback_root != root:
            return resolve_snapshot_repo_dir(fallback_root, repo_id)

    return None


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

    current_root = project_root()
    main_root = _main_project_root(current_root)
    if main_root is not None:
        fallback_root = main_root / "models"
        if fallback_root != root:
            return resolve_snapshot_model_dir(fallback_root, repo_id, required_subdirs)

    return None
