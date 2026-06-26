"""Input format detection for videos, image sequences, and single images."""

from enum import Enum
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr"}


class InputKind(Enum):
    VIDEO = "video"
    IMAGE = "image"
    SEQUENCE = "sequence"


def is_video_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: Path | str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def detect_input_kind(path: Path | str) -> InputKind:
    input_path = Path(path)
    if input_path.is_dir():
        return InputKind.SEQUENCE
    if is_video_file(input_path):
        return InputKind.VIDEO
    if is_image_file(input_path):
        return InputKind.IMAGE
    raise ValueError(
        f"Unsupported input format: {input_path}. "
        f"Supported videos: {sorted(VIDEO_EXTENSIONS)}; "
        f"supported images: {sorted(IMAGE_EXTENSIONS)}"
    )
