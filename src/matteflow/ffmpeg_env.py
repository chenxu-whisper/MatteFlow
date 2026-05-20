from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import which as shutil_which
from typing import Callable, Iterable


@dataclass(frozen=True)
class FFmpegDiscoveryResult:
    found: bool
    ffmpeg_path: str | None
    bin_dir: str | None
    source: str | None


@dataclass(frozen=True)
class MediaToolDiscoveryResult:
    ffmpeg_path: str | None
    ffprobe_path: str | None
    bin_dir: str | None
    source: str | None
    complete: bool
    download_required: bool


def _default_common_candidate_dirs() -> list[str]:
    home = Path.home()
    return [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        r"C:\Program Files (x86)\ffmpeg\bin",
        str(home / "ffmpeg" / "bin"),
        str(home / "scoop" / "apps" / "ffmpeg" / "current" / "bin"),
    ]


def _normalize(path: str | Path) -> str:
    return Path(path).as_posix()


def _normalize_or_none(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return _normalize(path)


def _result(ffmpeg_path: str | None, source: str | None) -> FFmpegDiscoveryResult:
    if ffmpeg_path is None:
        return FFmpegDiscoveryResult(found=False, ffmpeg_path=None, bin_dir=None, source=None)
    return FFmpegDiscoveryResult(
        found=True,
        ffmpeg_path=_normalize(ffmpeg_path),
        bin_dir=_normalize(Path(ffmpeg_path).parent),
        source=source,
    )


def _media_result(
    *,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
    source: str | None,
    download_required: bool,
) -> MediaToolDiscoveryResult:
    normalized_ffmpeg = _normalize_or_none(ffmpeg_path)
    normalized_ffprobe = _normalize_or_none(ffprobe_path)

    bin_dir = None
    if normalized_ffmpeg is not None:
        bin_dir = _normalize(Path(normalized_ffmpeg).parent)
    if normalized_ffprobe is not None and bin_dir is None:
        bin_dir = _normalize(Path(normalized_ffprobe).parent)

    complete = normalized_ffmpeg is not None and normalized_ffprobe is not None
    return MediaToolDiscoveryResult(
        ffmpeg_path=normalized_ffmpeg,
        ffprobe_path=normalized_ffprobe,
        bin_dir=bin_dir,
        source=source,
        complete=complete,
        download_required=download_required,
    )


def _resolve_tool_pair(
    *,
    ffmpeg_path: str | None,
    ffprobe_path: str | None,
    path_exists: Callable[[str], bool],
    source: str | None,
) -> MediaToolDiscoveryResult:
    candidate_ffmpeg = None
    if ffmpeg_path:
        normalized_ffmpeg = _normalize(ffmpeg_path)
        if path_exists(normalized_ffmpeg):
            candidate_ffmpeg = normalized_ffmpeg

    candidate_ffprobe = None
    if ffprobe_path:
        normalized_ffprobe = _normalize(ffprobe_path)
        if path_exists(normalized_ffprobe):
            candidate_ffprobe = normalized_ffprobe

    return _media_result(
        ffmpeg_path=candidate_ffmpeg,
        ffprobe_path=candidate_ffprobe,
        source=source,
        download_required=(candidate_ffmpeg is not None and candidate_ffprobe is None),
    )


def discover_media_tools(
    *,
    ffmpeg_which: Callable[[str], str | None] = shutil_which,
    ffprobe_which: Callable[[str], str | None] = shutil_which,
    path_exists: Callable[[str], bool] | None = None,
    imageio_ffmpeg_getter: Callable[[], str | None] | None = None,
    common_candidate_dirs: Iterable[str] | None = None,
) -> MediaToolDiscoveryResult:
    path_exists = path_exists or (lambda path: Path(path).is_file())
    common_candidate_dirs = list(common_candidate_dirs or _default_common_candidate_dirs())

    ffmpeg_on_path = ffmpeg_which("ffmpeg")
    ffprobe_on_path = ffprobe_which("ffprobe")
    path_result = _resolve_tool_pair(
        ffmpeg_path=ffmpeg_on_path,
        ffprobe_path=ffprobe_on_path,
        path_exists=path_exists,
        source="path",
    )
    if path_result.complete:
        return path_result

    for directory in common_candidate_dirs:
        common_dir_result = _resolve_tool_pair(
            ffmpeg_path=Path(directory) / "ffmpeg.exe",
            ffprobe_path=Path(directory) / "ffprobe.exe",
            path_exists=path_exists,
            source="common_dir",
        )
        if common_dir_result.complete:
            return common_dir_result

    getter = imageio_ffmpeg_getter
    if getter is None:
        try:
            import imageio_ffmpeg

            getter = imageio_ffmpeg.get_ffmpeg_exe
        except Exception:
            getter = None

    if getter is not None:
        try:
            ffmpeg_candidate = getter()
        except Exception:
            ffmpeg_candidate = None
        imageio_result = _resolve_tool_pair(
            ffmpeg_path=ffmpeg_candidate,
            ffprobe_path=None,
            path_exists=path_exists,
            source="imageio_ffmpeg",
        )
        if imageio_result.ffmpeg_path is not None:
            return imageio_result

    if path_result.ffmpeg_path is not None:
        return path_result

    return _media_result(
        ffmpeg_path=None,
        ffprobe_path=None,
        source=None,
        download_required=True,
    )


def discover_ffmpeg(
    *,
    which: Callable[[str], str | None] = shutil_which,
    path_exists: Callable[[str], bool] | None = None,
    imageio_ffmpeg_getter: Callable[[], str | None] | None = None,
    common_candidate_dirs: Iterable[str] | None = None,
) -> FFmpegDiscoveryResult:
    path_exists = path_exists or (lambda path: Path(path).is_file())
    common_candidate_dirs = list(common_candidate_dirs or _default_common_candidate_dirs())

    ffmpeg_on_path = which("ffmpeg")
    if ffmpeg_on_path and path_exists(ffmpeg_on_path):
        return _result(ffmpeg_on_path, "path")

    for directory in common_candidate_dirs:
        candidate = _normalize(Path(directory) / "ffmpeg.exe")
        if path_exists(candidate):
            return _result(candidate, "common_dir")

    getter = imageio_ffmpeg_getter
    if getter is None:
        try:
            import imageio_ffmpeg

            getter = imageio_ffmpeg.get_ffmpeg_exe
        except Exception:
            getter = None

    if getter is not None:
        try:
            candidate = getter()
        except Exception:
            candidate = None
        if candidate and path_exists(candidate):
            return _result(candidate, "imageio_ffmpeg")

    return _result(None, None)


def main() -> int:
    print(json.dumps(asdict(discover_media_tools())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
