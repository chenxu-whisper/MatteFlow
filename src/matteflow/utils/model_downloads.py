from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Callable

Validator = Callable[[Path], tuple[bool, str | None]]


def download_file_atomically(url: str, destination: Path, validate: Validator) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".partial")

    if temp_path.exists():
        temp_path.unlink()

    try:
        urllib.request.urlretrieve(url, temp_path)
        valid, reason = validate(temp_path)
        if not valid:
            raise RuntimeError(f"Model validation failed: {reason}")
        os.replace(temp_path, destination)
        return destination
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
