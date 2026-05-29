"""CLI verification utility for time-driven preview cleanup."""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from pathlib import Path


def _load_web_gui():
    previous = os.environ.get("MATTEFLOW_SKIP_UI_MODEL_PROBE")
    os.environ["MATTEFLOW_SKIP_UI_MODEL_PROBE"] = "1"
    try:
        return importlib.import_module("scripts.web_gui")
    finally:
        if previous is None:
            os.environ.pop("MATTEFLOW_SKIP_UI_MODEL_PROBE", None)
        else:
            os.environ["MATTEFLOW_SKIP_UI_MODEL_PROBE"] = previous


def run_verification() -> dict[str, object]:
    web_gui = _load_web_gui()
    preview_root = Path(tempfile.mkdtemp(prefix="matteflow-preview-cleanup-")) / "gradio-temp"
    cache_root = preview_root / "gradio" / "matteflow_previews"
    stale_dir = cache_root / "stale-job"
    downloads_dir = cache_root / "downloads"
    stale_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    stale_preview = stale_dir / "preview.mp4"
    stale_download = downloads_dir / "processed_000000.png"
    stale_preview.write_bytes(b"old-video")
    stale_download.write_bytes(b"old-png")

    os.utime(stale_dir, (1000, 1000))
    os.utime(stale_preview, (1000, 1000))
    os.utime(downloads_dir, (1000, 1000))
    os.utime(stale_download, (1000, 1000))

    original_gettempdir = web_gui.tempfile.gettempdir
    original_time = web_gui.time.time
    original_max_age = web_gui.GRADIO_PREVIEW_MAX_AGE_SECONDS

    try:
        web_gui.tempfile.gettempdir = lambda: str(preview_root)
        web_gui.time.time = lambda: 2000.0
        web_gui.GRADIO_PREVIEW_MAX_AGE_SECONDS = 10

        before = {
            "stale_job_dir_exists": stale_dir.exists(),
            "stale_preview_file_exists": stale_preview.exists(),
            "downloads_dir_exists": downloads_dir.exists(),
            "stale_download_exists": stale_download.exists(),
        }

        resolved_root = web_gui._gradio_preview_root()

        after = {
            "stale_job_dir_exists": stale_dir.exists(),
            "stale_preview_file_exists": stale_preview.exists(),
            "downloads_dir_exists": downloads_dir.exists(),
            "stale_download_exists": stale_download.exists(),
        }

        return {
            "resolved_root": str(resolved_root),
            "before": before,
            "after": after,
        }
    finally:
        web_gui.tempfile.gettempdir = original_gettempdir
        web_gui.time.time = original_time
        web_gui.GRADIO_PREVIEW_MAX_AGE_SECONDS = original_max_age
        shutil.rmtree(preview_root.parent, ignore_errors=True)


def main() -> int:
    result = run_verification()

    print("Before cleanup:")
    print(f"  stale job dir exists: {result['before']['stale_job_dir_exists']}")
    print(f"  stale preview file exists: {result['before']['stale_preview_file_exists']}")
    print(f"  downloads dir exists: {result['before']['downloads_dir_exists']}")
    print(f"  stale download exists: {result['before']['stale_download_exists']}")
    print("After cleanup:")
    print(f"  resolved root: {result['resolved_root']}")
    print(f"  stale job dir exists: {result['after']['stale_job_dir_exists']}")
    print(f"  stale preview file exists: {result['after']['stale_preview_file_exists']}")
    print(f"  downloads dir exists: {result['after']['downloads_dir_exists']}")
    print(f"  stale download exists: {result['after']['stale_download_exists']}")
    return 0
