"""Lightweight per-image parameter suggestions for green-screen matting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import MattingConfig
from .input.formats import IMAGE_EXTENSIONS, InputKind, detect_input_kind, is_image_file


@dataclass(frozen=True)
class AutoParamSuggestion:
    """Suggested config overrides plus a short user-facing summary."""

    params: dict[str, float | int | str]
    summary: str


def suggest_single_frame_params(path: str | Path, base_config: MattingConfig) -> AutoParamSuggestion:
    """Analyze a single image and suggest conservative green-screen parameters."""
    input_path = Path(path)
    if not is_image_file(input_path):
        return AutoParamSuggestion({}, "自动优化跳过: 当前仅支持单帧图片")

    frame = _read_image(input_path)
    if frame is None or frame.size == 0:
        return AutoParamSuggestion({}, "自动优化跳过: 图片为空")

    return _suggest_from_frame(frame, base_config, "image")


def suggest_input_params(path: str | Path, base_config: MattingConfig) -> AutoParamSuggestion:
    """Analyze a representative input frame; videos/sequences use their middle frame."""
    input_path = Path(path)
    try:
        kind = detect_input_kind(input_path)
    except ValueError as exc:
        return AutoParamSuggestion({}, f"自动优化跳过: {exc}")

    frame, sample = _load_representative_frame(input_path, kind)
    if frame is None or frame.size == 0:
        return AutoParamSuggestion({}, "自动优化跳过: 无法读取代表帧")

    return _suggest_from_frame(frame, base_config, sample)


def _suggest_from_frame(
    frame: np.ndarray,
    base_config: MattingConfig,
    sample: str,
) -> AutoParamSuggestion:
    """Suggest parameters from an RGB frame and include sampling context."""
    metrics = _analyze_frame(frame)
    params: dict[str, float | int | str] = {}

    screen_color = "blue" if metrics["blue_excess"] > metrics["green_excess"] * 1.15 else "green"
    params["screen_color"] = screen_color

    green_similarity = float(getattr(base_config, "green_similarity", 0.4))
    if metrics["screen_purity"] > 0.22:
        green_similarity = 0.35
    elif metrics["screen_purity"] < 0.10:
        green_similarity = 0.55
    params["green_similarity"] = float(np.clip(green_similarity, 0.1, 1.0))

    key_strength = float(getattr(base_config, "key_strength", 1.0))
    if metrics["white_subject_ratio"] > 0.08:
        key_strength = min(key_strength, 0.95)
        params["white_protect_brightness"] = 170
        params["white_protect_saturation"] = 45
    else:
        params["white_protect_brightness"] = getattr(base_config, "white_protect_brightness", 180)
        params["white_protect_saturation"] = getattr(base_config, "white_protect_saturation", 25)
    params["key_strength"] = float(np.clip(key_strength, 0.75, 1.15))

    transparency_preserve = float(getattr(base_config, "transparency_preserve", 0.7))
    if metrics["pink_glow_ratio"] > 0.05 or metrics["white_glow_ratio"] > 0.08:
        transparency_preserve = max(transparency_preserve, 0.72)
    params["transparency_preserve"] = float(np.clip(transparency_preserve, 0.55, 0.78))

    green_despill = float(getattr(base_config, "green_despill_strength", 0.7))
    edge_despill = float(getattr(base_config, "edge_despill_factor", 1.2))
    if metrics["green_haze_ratio"] > 0.18:
        green_despill = max(green_despill, 0.78)
        edge_despill = max(edge_despill, 1.3)
    params["green_despill_strength"] = float(np.clip(green_despill, 0.4, 0.9))
    params["edge_despill_factor"] = float(np.clip(edge_despill, 0.8, 1.6))

    params["clip_black"] = 0.0
    params["clip_white"] = 1.0

    summary = (
        f"自动优化: sample={sample}, screen={screen_color}, similarity={params['green_similarity']:.2f}, "
        f"key={params['key_strength']:.2f}, preserve={params['transparency_preserve']:.2f}"
    )
    return AutoParamSuggestion(params, summary)


def apply_suggestion(config: MattingConfig, suggestion: AutoParamSuggestion) -> None:
    """Apply supported suggested overrides to a MattingConfig in-place."""
    for key, value in suggestion.params.items():
        if hasattr(config, key):
            setattr(config, key, value)


def _load_representative_frame(input_path: Path, kind: InputKind) -> tuple[np.ndarray | None, str]:
    if kind == InputKind.IMAGE:
        return _read_image(input_path), "image"

    if kind == InputKind.SEQUENCE:
        image_files: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = sorted(set(image_files))
        if not image_files:
            return None, "sequence_middle"
        middle = image_files[len(image_files) // 2]
        return _read_image(middle), "sequence_middle"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return None, "video_middle"
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = max(total // 2, 0) if total > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame_bgr = cap.read()
        if not ok and target != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = cap.read()
        if not ok:
            return None, "video_middle"
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), "video_middle"
    finally:
        cap.release()


def _read_image(path: Path) -> np.ndarray | None:
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _analyze_frame(frame: np.ndarray) -> dict[str, float]:
    frame_f = frame.astype(np.float32) / 255.0
    r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
    brightness = (r + g + b) / 3.0
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    green_excess_map = np.clip(g - np.maximum(r, b), 0.0, 1.0)
    blue_excess_map = np.clip(b - np.maximum(r, g), 0.0, 1.0)
    green_excess = float(green_excess_map.mean())
    blue_excess = float(blue_excess_map.mean())
    screen_purity = max(green_excess, blue_excess)

    screen_mask = (green_excess_map > 0.12) | (blue_excess_map > 0.12)
    white_subject = (~screen_mask) & (brightness > 0.66) & (chroma < 0.22)
    pink_glow = (~screen_mask) & (r > g + 0.12) & (b > g + 0.04) & (brightness > 0.45)
    white_glow = (~screen_mask) & (brightness > 0.72) & (chroma < 0.35)
    green_haze = (green_excess_map > 0.04) & (green_excess_map < 0.20) & (brightness > 0.22)

    total = float(frame.shape[0] * frame.shape[1])
    return {
        "green_excess": green_excess,
        "blue_excess": blue_excess,
        "screen_purity": screen_purity,
        "white_subject_ratio": float(white_subject.sum() / total),
        "pink_glow_ratio": float(pink_glow.sum() / total),
        "white_glow_ratio": float(white_glow.sum() / total),
        "green_haze_ratio": float(green_haze.sum() / total),
    }
