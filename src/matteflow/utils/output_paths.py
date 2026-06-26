import re
from datetime import datetime
from pathlib import Path


def sanitize_output_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\s]+', "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "output"


def resolve_project_output_dir(
    input_path: Path,
    project_root: Path,
    output_root: Path | None = None,
    job_token: str | None = None,
) -> Path:
    output_root = output_root or (project_root / "temp" / "output")
    output_root.mkdir(parents=True, exist_ok=True)

    stem = sanitize_output_name(Path(input_path).stem)
    if job_token:
        stem = f"{stem}_{sanitize_output_name(job_token)}"
    candidate = output_root / stem

    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    if candidate.is_dir() and not any(candidate.iterdir()):
        return candidate

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fallback = output_root / f"{stem}_{timestamp}"
    suffix = 2
    while fallback.exists():
        fallback = output_root / f"{stem}_{timestamp}_{suffix}"
        suffix += 1
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback
