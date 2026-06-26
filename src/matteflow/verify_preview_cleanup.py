"""Self-check for preview cache cleanup behavior."""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

GRADIO_PREVIEW_MAX_AGE_SECONDS = 3600


def _is_stale(path: Path, now: float) -> bool:
    try:
        return now - path.stat().st_mtime > GRADIO_PREVIEW_MAX_AGE_SECONDS
    except FileNotFoundError:
        return False


def _prune_stale_preview_artifacts(cache_root: Path, active_job_id: str | None = None) -> None:
    if not cache_root.exists():
        return

    now = time.time()
    downloads_dir = cache_root / "downloads"

    for candidate in cache_root.iterdir():
        if candidate.name == "downloads" or not candidate.is_dir():
            continue
        if active_job_id is not None and candidate.name == active_job_id:
            continue
        if _is_stale(candidate, now):
            shutil.rmtree(candidate, ignore_errors=True)

    if downloads_dir.exists():
        for candidate in downloads_dir.iterdir():
            if candidate.is_file() and _is_stale(candidate, now):
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    continue


def _touch_old(path: Path, age_seconds: int) -> None:
    old_time = time.time() - age_seconds
    path.touch()
    Path(path).stat()
    import os

    os.utime(path, (old_time, old_time))


def _build_fixture(cache_root: Path) -> None:
    stale_job = cache_root / "stale-job"
    fresh_job = cache_root / "fresh-job"
    active_job = cache_root / "active-job"
    downloads = cache_root / "downloads"

    stale_job.mkdir(parents=True)
    fresh_job.mkdir(parents=True)
    active_job.mkdir(parents=True)
    downloads.mkdir(parents=True)

    (stale_job / "preview.mp4").write_text("stale", encoding="utf-8")
    (fresh_job / "preview.mp4").write_text("fresh", encoding="utf-8")
    (active_job / "preview.mp4").write_text("active", encoding="utf-8")
    (downloads / "stale.zip").write_text("stale", encoding="utf-8")
    (downloads / "fresh.zip").write_text("fresh", encoding="utf-8")

    old_age = GRADIO_PREVIEW_MAX_AGE_SECONDS + 60
    _touch_old(stale_job, old_age)
    _touch_old(stale_job / "preview.mp4", old_age)
    _touch_old(downloads / "stale.zip", old_age)


def _relative_entries(cache_root: Path) -> list[str]:
    return sorted(str(path.relative_to(cache_root)).replace("\\", "/") for path in cache_root.rglob("*"))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="matteflow-preview-cleanup-") as tmp:
        cache_root = Path(tmp) / "matteflow_previews"
        _build_fixture(cache_root)

        print("Before cleanup:")
        for entry in _relative_entries(cache_root):
            print(f"- {entry}")

        _prune_stale_preview_artifacts(cache_root, active_job_id="active-job")

        print("After cleanup:")
        for entry in _relative_entries(cache_root):
            print(f"- {entry}")

        stale_removed = not (cache_root / "stale-job").exists()
        stale_download_removed = not (cache_root / "downloads" / "stale.zip").exists()
        fresh_kept = (cache_root / "fresh-job").exists()
        active_kept = (cache_root / "active-job").exists()

        if stale_removed and stale_download_removed and fresh_kept and active_kept:
            print("Preview cleanup verification passed.")
            return 0

        print("Preview cleanup verification failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
