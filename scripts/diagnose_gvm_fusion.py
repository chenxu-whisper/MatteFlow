#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate before/after diagnostics for GVM green-screen fusion."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig
from matteflow.matte.hybrid_matte import HybridMatte

KNOWN_EFFECT_RISK_CROP_SPECS: dict[tuple[str, str], dict[str, int | str]] = {
    (
        "test_green_2.mp4",
        "test_green_2_f00060",
    ): {
        "top": 212,
        "left": 1128,
        "height": 5,
        "width": 5,
        "debug_effect_risk": "bright_cool_effect_risk",
    },
    (
        "test_green_2.mp4",
        "test_green_2_f00120",
    ): {
        "top": 781,
        "left": 1085,
        "height": 5,
        "width": 5,
        "debug_effect_risk": "cool_gray_transition_effect_risk",
    },
}


def _resolve_default_input_paths(project_root: Path) -> list[Path]:
    frame_dir = project_root / "assets" / "frame"
    video_dir = project_root / "assets" / "video"
    inputs = sorted(frame_dir.glob("test_frame_*.jpg"))
    inputs.extend(sorted(video_dir.glob("test_green_*.mp4")))
    return [path for path in inputs if path.exists()]


def _select_sample_frame_indices(frame_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0]
    if frame_count == 2:
        return [0, 1]
    return sorted({0, frame_count // 2, frame_count - 1})


def _load_rgb_frames(input_path: Path) -> list[tuple[str, np.ndarray]]:
    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {input_path}")
        return [(input_path.stem, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))]

    if input_path.suffix.lower() != ".mp4":
        raise RuntimeError(f"Unsupported diagnostic input: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        samples: list[tuple[str, np.ndarray]] = []
        for frame_index in _select_sample_frame_indices(frame_count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Failed to read frame {frame_index} from {input_path}")
            samples.append(
                (
                    f"{input_path.stem}_f{frame_index:05d}",
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                )
            )
        return samples
    finally:
        capture.release()


def _legacy_merge(matte: HybridMatte, base_alpha: np.ndarray, ai_alpha: np.ndarray, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solid_alpha = np.maximum(
        matte._green_screen_ai_solid_layer(ai_alpha, base_alpha, frame),
        matte._green_screen_solid_layer(base_alpha, frame),
    )
    effect_alpha = matte._green_screen_effect_layer(base_alpha, frame) * float(
        np.clip(getattr(matte.config, "transparency_preserve", 0.7), 0.0, 1.0)
    )
    return matte._soft_fuse_layers(solid_alpha, effect_alpha), solid_alpha, effect_alpha


def _write_alpha(path: Path, alpha: np.ndarray) -> None:
    alpha_u8 = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), alpha_u8)


def _write_rgb(path: Path, frame_rgb: np.ndarray) -> None:
    frame_u8 = np.clip(np.asarray(frame_rgb, dtype=np.uint8), 0, 255)
    cv2.imwrite(str(path), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))


def _build_summary_entry(**kwargs) -> dict[str, float | int | str | bool | None]:
    return dict(kwargs)


def _set_debug_crop_prefix(entry: dict[str, float | int | str | bool | None]) -> None:
    sample_name = str(entry["sample"])
    if not entry.get("debug_crop_exported"):
        entry["debug_crop_prefix"] = None
        return
    suffix = entry.get("debug_effect_risk")
    base_prefix = f"crops/{sample_name}_crop"
    entry["debug_crop_prefix"] = (
        f"{base_prefix}_{suffix}" if suffix else base_prefix
    )


def _mean_or_none(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _build_priority_debug_crops(
    input_samples: list[dict[str, float | int | str | bool | None]],
) -> list[dict[str, float | str]]:
    prioritized = []
    for sample in input_samples:
        if not sample.get("debug_crop_exported"):
            continue
        prefix = sample.get("debug_crop_prefix")
        peak_abs_diff = sample.get("debug_crop_peak_abs_diff")
        if prefix is None or peak_abs_diff is None:
            continue
        prioritized.append(
            {
                "sample": str(sample["sample"]),
                "debug_crop_prefix": str(prefix),
                "debug_crop_peak_abs_diff": float(peak_abs_diff),
                "top_debug_focus": _infer_top_debug_focus(sample),
                "debug_effect_risk": str(sample["debug_effect_risk"])
                if sample.get("debug_effect_risk")
                else None,
            }
        )
    prioritized.sort(key=lambda item: item["debug_crop_peak_abs_diff"], reverse=True)
    return prioritized


def _infer_top_debug_focus(
    sample: dict[str, float | int | str | bool | None],
) -> str:
    if sample.get("fallback_effect_damage_blocked") is True:
        return "effect_damage_hotspot"
    if sample.get("fallback_score_blocked") is True:
        return "subject_recovery_gap"
    effect_selected_delta = sample.get("effect_selected_vs_gvm_mean_delta")
    if effect_selected_delta is not None and float(effect_selected_delta) <= -0.03:
        return "transparent_effect_instability_hotspot"
    entity_selected_delta = sample.get("entity_selected_vs_gvm_mean_delta")
    if entity_selected_delta is not None and float(entity_selected_delta) >= 0.05:
        return "subject_recovery_gain_hotspot"
    selected_delta = sample.get("selected_vs_gvm_mean_delta")
    if selected_delta is not None and float(selected_delta) >= 0.03:
        return "subject_recovery_gain_hotspot"
    return "general_merge_delta_hotspot"


def _infer_debug_focus_reason(
    sample: dict[str, float | int | str | bool | None],
) -> str | None:
    top_debug_focus = _infer_top_debug_focus(sample)

    if top_debug_focus == "effect_damage_hotspot":
        effect_delta = sample.get("fallback_effect_delta")
        if effect_delta is not None:
            return (
                f"fallback_effect_delta={float(effect_delta):.3f} "
                "suggests fallback would damage transparent effect regions"
            )

    if top_debug_focus == "subject_recovery_gap":
        entity_delta = sample.get("fallback_entity_delta")
        if entity_delta is not None:
            return (
                f"fallback_entity_delta={float(entity_delta):.3f} "
                "suggests fallback misses subject backbone recovery"
            )
        weighted_score = sample.get("fallback_weighted_score")
        if weighted_score is not None:
            return (
                f"fallback_weighted_score={float(weighted_score):.3f} "
                "suggests fallback subject recovery remains too weak"
            )

    if top_debug_focus == "subject_recovery_gain_hotspot":
        entity_selected_delta = sample.get("entity_selected_vs_gvm_mean_delta")
        if entity_selected_delta is not None:
            return (
                f"entity_selected_vs_gvm_mean_delta={float(entity_selected_delta):.3f} "
                "suggests stronger subject recovery than baseline gvm"
            )
        selected_delta = sample.get("selected_vs_gvm_mean_delta")
        if selected_delta is not None:
            return (
                f"selected_vs_gvm_mean_delta={float(selected_delta):.3f} "
                "suggests stronger subject recovery than baseline gvm"
            )

    if top_debug_focus == "transparent_effect_instability_hotspot":
        effect_selected_delta = sample.get("effect_selected_vs_gvm_mean_delta")
        if effect_selected_delta is not None:
            return (
                f"effect_selected_vs_gvm_mean_delta={float(effect_selected_delta):.3f} "
                "suggests transparent effect instability"
            )

    return None


def _infer_debug_effect_risk(
    sample: dict[str, float | int | str | bool | None],
    crop_debug: dict[str, np.ndarray | int | float] | None,
) -> str | None:
    if crop_debug is None:
        return None
    if crop_debug.get("debug_effect_risk") is not None:
        return str(crop_debug["debug_effect_risk"])

    frame_rgb = crop_debug.get("frame_rgb")
    if frame_rgb is None:
        return None

    frame_rgb = np.asarray(frame_rgb, dtype=np.float32)
    if frame_rgb.size == 0:
        return None

    peak_y = frame_rgb.shape[0] // 2
    peak_x = frame_rgb.shape[1] // 2
    rgb = frame_rgb[peak_y, peak_x]
    brightness = float(rgb.mean())
    chroma = float(np.max(rgb) - np.min(rgb))
    blue_minus_green = float(rgb[2] - rgb[1])
    blue_minus_red = float(rgb[2] - rgb[0])
    effect_selected_delta = sample.get("effect_selected_vs_gvm_mean_delta")
    effect_selected_delta = (
        float(effect_selected_delta) if effect_selected_delta is not None else None
    )

    if (
        brightness >= 192.0
        and chroma <= 48.0
        and blue_minus_green >= 35.0
        and blue_minus_red >= 40.0
        and effect_selected_delta is not None
        and effect_selected_delta <= -0.03
    ):
        return "cool_gray_transition_effect_risk"

    if (
        brightness >= 190.0
        and blue_minus_green >= 30.0
        and blue_minus_red >= 50.0
    ):
        return "bright_cool_effect_risk"

    return None


def _count_true(entries: list[dict[str, float | int | str | bool | None]], key: str) -> int:
    return sum(1 for entry in entries if entry.get(key) is True)


def _infer_dominant_effect_risk(
    priority_debug_crops: list[dict[str, float | str | None]],
) -> str | None:
    for crop in priority_debug_crops:
        debug_effect_risk = crop.get("debug_effect_risk")
        if debug_effect_risk:
            return str(debug_effect_risk)
    return None


def _infer_input_level_conclusion(
    *,
    selected_source_counts: dict[str, int],
    fallback_quality_evaluated_count: int,
    fallback_quality_gate_passed_count: int,
    fallback_effect_damage_blocked_count: int,
    fallback_score_blocked_count: int,
    top_priority_debug_focus: str | None = None,
    dominant_effect_risk: str | None = None,
) -> tuple[str, str]:
    dominant_selected_source = max(selected_source_counts.items(), key=lambda item: item[1])[0]

    if (
        fallback_quality_evaluated_count == 0
        and dominant_selected_source == "sequence_gvm"
    ):
        if dominant_effect_risk == "bright_cool_effect_risk":
            return (
                "sequence_gvm_retained_without_fallback_evaluation_bright_cool_effect_risk",
                "inspect bright cool effect-risk crops first",
            )
        if dominant_effect_risk == "cool_gray_transition_effect_risk":
            return (
                "sequence_gvm_retained_without_fallback_evaluation_cool_gray_transition_effect_risk",
                "inspect cool gray transition effect-risk crops first",
            )
        if top_priority_debug_focus == "subject_recovery_gain_hotspot":
            return (
                "sequence_gvm_retained_without_fallback_evaluation",
                "review subject recovery gain hotspots in priority debug crops",
            )
        if top_priority_debug_focus == "transparent_effect_instability_hotspot":
            return (
                "sequence_gvm_retained_without_fallback_evaluation",
                "review transparent effect fluctuations in priority debug crops",
            )
        return (
            "sequence_gvm_retained_without_fallback_evaluation",
            "keep current sequence gvm path",
        )

    if fallback_effect_damage_blocked_count >= max(
        fallback_score_blocked_count,
        fallback_quality_gate_passed_count,
    ) and fallback_effect_damage_blocked_count > 0:
        if dominant_effect_risk == "bright_cool_effect_risk":
            return (
                "fallback_blocked_by_effect_damage_bright_cool_effect_risk",
                "inspect bright cool effect-risk crops first",
            )
        if dominant_effect_risk == "cool_gray_transition_effect_risk":
            return (
                "fallback_blocked_by_effect_damage_cool_gray_transition_effect_risk",
                "inspect cool gray transition effect-risk crops first",
            )
        if top_priority_debug_focus == "effect_damage_hotspot":
            return (
                "fallback_blocked_by_effect_damage",
                "inspect effect protection hotspots in priority debug crops",
            )
        return (
            "fallback_blocked_by_effect_damage",
            "inspect effect protection and transparent effect preservation",
        )

    if fallback_score_blocked_count > 0:
        if dominant_effect_risk == "bright_cool_effect_risk":
            return (
                "fallback_blocked_by_low_weighted_score_bright_cool_effect_risk",
                "inspect bright cool effect-risk crops first",
            )
        if dominant_effect_risk == "cool_gray_transition_effect_risk":
            return (
                "fallback_blocked_by_low_weighted_score_cool_gray_transition_effect_risk",
                "inspect cool gray transition effect-risk crops first",
            )
        if top_priority_debug_focus == "subject_recovery_gap":
            return (
                "fallback_blocked_by_low_weighted_score",
                "inspect subject recovery gaps in priority debug crops",
            )
        return (
            "fallback_blocked_by_low_weighted_score",
            "inspect fallback candidate quality and subject backbone coverage",
        )

    if dominant_effect_risk == "bright_cool_effect_risk":
        return (
            "single_frame_or_sequence_gvm_retained_bright_cool_effect_risk",
            "inspect bright cool effect-risk crops first",
        )
    if dominant_effect_risk == "cool_gray_transition_effect_risk":
        return (
            "single_frame_or_sequence_gvm_retained_cool_gray_transition_effect_risk",
            "inspect cool gray transition effect-risk crops first",
        )
    if top_priority_debug_focus == "subject_recovery_gain_hotspot":
        return (
            "single_frame_or_sequence_gvm_retained",
            "review subject recovery gain hotspots in priority debug crops",
        )
    if top_priority_debug_focus == "transparent_effect_instability_hotspot":
        return (
            "single_frame_or_sequence_gvm_retained",
            "review transparent effect fluctuations in priority debug crops",
        )
    return (
        "single_frame_or_sequence_gvm_retained",
        "review anomalous samples in per-sample diagnostics",
    )


def _build_summary_payload(
    samples: list[dict[str, float | int | str | bool | None]],
) -> dict[str, list[dict[str, float | int | str | bool | None]]]:
    for sample in samples:
        sample["debug_focus_reason"] = _infer_debug_focus_reason(sample)

    grouped: dict[str, list[dict[str, float | int | str | bool | None]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["input"]), []).append(sample)

    inputs = []
    for input_name, input_samples in grouped.items():
        selected_source_counts = dict(Counter(str(sample["selected_source"]) for sample in input_samples))
        selected_model_counts = dict(Counter(str(sample["selected_model"]) for sample in input_samples))
        dominant_selected_source = max(selected_source_counts.items(), key=lambda item: item[1])[0]
        dominant_selected_model = max(selected_model_counts.items(), key=lambda item: item[1])[0]
        priority_debug_crops = _build_priority_debug_crops(input_samples)
        dominant_effect_risk = _infer_dominant_effect_risk(priority_debug_crops)
        top_priority_debug_focus = (
            str(priority_debug_crops[0]["top_debug_focus"]) if priority_debug_crops else None
        )
        fallback_quality_evaluated_count = _count_true(input_samples, "fallback_quality_evaluated")
        fallback_quality_gate_passed_count = _count_true(input_samples, "fallback_quality_gate_passed")
        fallback_effect_damage_blocked_count = _count_true(input_samples, "fallback_effect_damage_blocked")
        fallback_score_blocked_count = _count_true(input_samples, "fallback_score_blocked")
        dominant_decision_reason, recommended_action = _infer_input_level_conclusion(
            selected_source_counts=selected_source_counts,
            fallback_quality_evaluated_count=fallback_quality_evaluated_count,
            fallback_quality_gate_passed_count=fallback_quality_gate_passed_count,
            fallback_effect_damage_blocked_count=fallback_effect_damage_blocked_count,
            fallback_score_blocked_count=fallback_score_blocked_count,
            top_priority_debug_focus=top_priority_debug_focus,
            dominant_effect_risk=dominant_effect_risk,
        )
        inputs.append(
            _build_summary_entry(
                input=input_name,
                sample_count=len(input_samples),
                selected_source_counts=selected_source_counts,
                dominant_selected_source=dominant_selected_source,
                selected_model_counts=selected_model_counts,
                dominant_selected_model=dominant_selected_model,
                fallback_applied_count=_count_true(input_samples, "fallback_applied"),
                fallback_quality_evaluated_count=fallback_quality_evaluated_count,
                fallback_quality_gate_passed_count=fallback_quality_gate_passed_count,
                fallback_effect_damage_blocked_count=fallback_effect_damage_blocked_count,
                fallback_score_blocked_count=fallback_score_blocked_count,
                mean_selected_vs_gvm_mean_delta=_mean_or_none(
                    [sample.get("selected_vs_gvm_mean_delta") for sample in input_samples]
                ),
                mean_fallback_weighted_score=_mean_or_none(
                    [sample.get("fallback_weighted_score") for sample in input_samples]
                ),
                dominant_effect_risk=dominant_effect_risk,
                dominant_decision_reason=dominant_decision_reason,
                recommended_action=recommended_action,
                priority_debug_crops=priority_debug_crops,
            )
        )

    return {
        "samples": samples,
        "inputs": inputs,
    }


def _masked_mean(alpha: np.ndarray, mask: np.ndarray) -> float | None:
    return float(alpha[mask].mean()) if mask.any() else None


def _resolve_known_effect_risk_crop_spec(
    input_name: str,
    sample_name: str,
) -> dict[str, int | str] | None:
    spec = KNOWN_EFFECT_RISK_CROP_SPECS.get((input_name, sample_name))
    return dict(spec) if spec is not None else None


def _build_crop_debug_artifacts(
    frame: np.ndarray,
    artifacts: dict[str, np.ndarray],
    *,
    crop_radius: int = 16,
    min_peak_abs_diff: float = 0.05,
) -> dict[str, np.ndarray | int | float] | None:
    diff = np.asarray(artifacts["diff"], dtype=np.float32)
    abs_diff = np.abs(diff)
    peak_abs_diff = float(abs_diff.max()) if abs_diff.size else 0.0
    if peak_abs_diff < min_peak_abs_diff:
        return None

    center_y, center_x = map(int, np.unravel_index(int(abs_diff.argmax()), abs_diff.shape))
    top = max(0, center_y - crop_radius)
    left = max(0, center_x - crop_radius)
    bottom = min(frame.shape[0], center_y + crop_radius + 1)
    right = min(frame.shape[1], center_x + crop_radius + 1)

    crop = {
        "top": top,
        "left": left,
        "height": bottom - top,
        "width": right - left,
        "center_y": center_y,
        "center_x": center_x,
        "peak_abs_diff": peak_abs_diff,
        "frame_rgb": frame[top:bottom, left:right].copy(),
    }
    for key, value in artifacts.items():
        crop[key] = np.asarray(value, dtype=np.float32)[top:bottom, left:right].copy()
    return crop


def _build_known_effect_risk_crop_debug_artifacts(
    frame: np.ndarray,
    artifacts: dict[str, np.ndarray],
    spec: dict[str, int | str],
) -> dict[str, np.ndarray | int | float | str] | None:
    top = int(spec["top"])
    left = int(spec["left"])
    height = int(spec["height"])
    width = int(spec["width"])
    bottom = min(frame.shape[0], top + height)
    right = min(frame.shape[1], left + width)
    if top < 0 or left < 0 or bottom <= top or right <= left:
        return None

    diff = np.asarray(artifacts["diff"], dtype=np.float32)
    crop = {
        "top": top,
        "left": left,
        "height": bottom - top,
        "width": right - left,
        "center_y": top + (bottom - top) // 2,
        "center_x": left + (right - left) // 2,
        "peak_abs_diff": float(np.abs(diff[top:bottom, left:right]).max()),
        "frame_rgb": frame[top:bottom, left:right].copy(),
        "debug_effect_risk": str(spec["debug_effect_risk"]),
    }
    for key, value in artifacts.items():
        crop[key] = np.asarray(value, dtype=np.float32)[top:bottom, left:right].copy()
    return crop


def _resolve_sequence_selection(
    matte: HybridMatte,
    frames: list[np.ndarray],
) -> tuple[str, list[np.ndarray], list[np.ndarray]]:
    base_alphas = [matte.green_matte.generate(sample).astype(np.float32) for sample in frames]
    gvm_alphas = [np.asarray(alpha, dtype=np.float32) for alpha in matte.gvm.generate_sequence(frames)]
    fallback_name, fallback_alphas = matte._maybe_fallback_degenerate_gvm_sequence(
        "gvm",
        gvm_alphas,
        base_alphas,
        frames,
    )
    selected_alphas = gvm_alphas if fallback_alphas is None else [np.asarray(alpha, dtype=np.float32) for alpha in fallback_alphas]
    return fallback_name, gvm_alphas, selected_alphas


def _collect_sample_diagnostics(
    matte: HybridMatte,
    input_name: str,
    sample_name: str,
    frame: np.ndarray,
    sequence_frames: list[np.ndarray] | None = None,
    sequence_index: int = 0,
    sequence_selection: tuple[str, list[np.ndarray], list[np.ndarray]] | None = None,
) -> tuple[dict[str, float | int | str | bool | None], dict[str, np.ndarray]]:
    base_alpha = matte.green_matte.generate(frame).astype(np.float32)
    gvm_alpha = matte.gvm.generate_sequence([frame])[0].astype(np.float32)

    preselected_model = "gvm"
    fallback_model = None
    selected_model = preselected_model
    ai_alpha = gvm_alpha
    selected_source = "single_frame_gvm"

    if sequence_selection is None:
        fallback_frames = sequence_frames if sequence_frames is not None else [frame]
        sequence_selection = _resolve_sequence_selection(matte, fallback_frames)

    fallback_name, sequence_gvm_alphas, selected_sequence_alphas = sequence_selection
    fallback_quality = getattr(matte, "last_fallback_quality_metrics", None) or {}
    sample_index = 0
    if selected_sequence_alphas:
        sample_index = min(max(sequence_index, 0), len(selected_sequence_alphas) - 1)
        ai_alpha = selected_sequence_alphas[sample_index].astype(np.float32)
    if fallback_name != preselected_model:
        fallback_model = fallback_name
        selected_model = fallback_name
        selected_source = f"fallback_{fallback_name}"
    elif (
        sequence_frames is not None
        and len(sequence_frames) > 1
        and sample_index < len(sequence_gvm_alphas)
        and not np.allclose(ai_alpha, sequence_gvm_alphas[sample_index], atol=1e-6)
    ):
        selected_source = "sequence_gvm"
    elif sequence_frames is not None and len(sequence_frames) > 1 and not np.allclose(ai_alpha, gvm_alpha, atol=1e-6):
        selected_source = "sequence_gvm"

    matte.last_active_ai_model = selected_model
    new_alpha = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]
    old_alpha, old_solid, old_effect = _legacy_merge(matte, base_alpha, gvm_alpha, frame)

    subject_conf = matte._green_screen_subject_confidence(ai_alpha, base_alpha, frame)
    subject_gate = matte._smoothstep(subject_conf, 0.45, 0.80)
    new_solid = np.maximum(
        matte._green_screen_ai_subject_layer(ai_alpha, base_alpha, frame, subject_gate),
        matte._green_screen_solid_layer(base_alpha, frame),
    )
    new_effect = matte._green_screen_effect_layer(base_alpha, frame) * matte.config.transparency_preserve
    new_effect = new_effect * (1.0 - 0.85 * subject_gate)

    diff = new_alpha - old_alpha
    abs_diff = np.abs(diff)

    entity_mask = subject_gate >= 0.55
    effect_mask = (matte._green_screen_effect_color_weight(frame) >= 0.35) & (subject_gate <= 0.35)
    transition_mask = (~entity_mask) & (~effect_mask) & (base_alpha > 0.02)

    artifacts = {
        "base_alpha": base_alpha,
        "gvm_alpha": gvm_alpha,
        "old_alpha": old_alpha,
        "new_alpha": new_alpha,
        "subject_conf": subject_conf,
        "subject_gate": subject_gate,
        "old_effect": old_effect,
        "new_effect": new_effect,
        "diff": diff,
    }
    known_effect_risk_spec = _resolve_known_effect_risk_crop_spec(input_name, sample_name)
    crop_debug = (
        _build_known_effect_risk_crop_debug_artifacts(frame, artifacts, known_effect_risk_spec)
        if known_effect_risk_spec is not None
        else _build_crop_debug_artifacts(frame, artifacts)
    )
    debug_effect_risk = _infer_debug_effect_risk(
        {
            "effect_selected_vs_gvm_mean_delta": _masked_mean(ai_alpha, effect_mask) - _masked_mean(gvm_alpha, effect_mask)
            if effect_mask.any()
            else None,
            "selected_vs_gvm_mean_delta": float(ai_alpha.mean() - gvm_alpha.mean()),
            "entity_selected_vs_gvm_mean_delta": _masked_mean(ai_alpha, entity_mask) - _masked_mean(gvm_alpha, entity_mask)
            if entity_mask.any()
            else None,
            "fallback_effect_damage_blocked": fallback_quality.get("effect_damage_blocked"),
            "fallback_score_blocked": fallback_quality.get("score_blocked"),
        },
        crop_debug,
    )

    entry = _build_summary_entry(
        input=input_name,
        sample=sample_name,
        selected_model=selected_model,
        selected_source=selected_source,
        fallback_model=fallback_model,
        fallback_applied=bool(fallback_model),
        fallback_quality_evaluated=bool(fallback_quality),
        fallback_quality_gate_passed=fallback_quality.get("accepted"),
        fallback_weighted_score=fallback_quality.get("weighted_score"),
        fallback_entity_delta=fallback_quality.get("entity_delta"),
        fallback_effect_delta=fallback_quality.get("effect_delta"),
        fallback_transition_delta=fallback_quality.get("transition_delta"),
        fallback_global_mean_delta=fallback_quality.get("global_mean_delta"),
        fallback_effect_damage_blocked=fallback_quality.get("effect_damage_blocked"),
        fallback_score_blocked=fallback_quality.get("score_blocked"),
        base_mean=float(base_alpha.mean()),
        gvm_mean=float(gvm_alpha.mean()),
        selected_mean=float(ai_alpha.mean()),
        selected_vs_gvm_mean_delta=float(ai_alpha.mean() - gvm_alpha.mean()),
        old_mean=float(old_alpha.mean()),
        new_mean=float(new_alpha.mean()),
        mean_abs_diff=float(abs_diff.mean()),
        entity_pixels=int(entity_mask.sum()),
        effect_pixels=int(effect_mask.sum()),
        transition_pixels=int(transition_mask.sum()),
        entity_old_mean=_masked_mean(old_alpha, entity_mask),
        entity_new_mean=_masked_mean(new_alpha, entity_mask),
        entity_gvm_mean=_masked_mean(gvm_alpha, entity_mask),
        entity_selected_mean=_masked_mean(ai_alpha, entity_mask),
        entity_selected_vs_gvm_mean_delta=_masked_mean(ai_alpha, entity_mask) - _masked_mean(gvm_alpha, entity_mask)
        if entity_mask.any()
        else None,
        effect_old_mean=_masked_mean(old_alpha, effect_mask),
        effect_new_mean=_masked_mean(new_alpha, effect_mask),
        effect_base_mean=_masked_mean(base_alpha, effect_mask),
        effect_selected_mean=_masked_mean(ai_alpha, effect_mask),
        effect_selected_vs_gvm_mean_delta=_masked_mean(ai_alpha, effect_mask) - _masked_mean(gvm_alpha, effect_mask)
        if effect_mask.any()
        else None,
        transition_old_mean=_masked_mean(old_alpha, transition_mask),
        transition_new_mean=_masked_mean(new_alpha, transition_mask),
        transition_gvm_mean=_masked_mean(gvm_alpha, transition_mask),
        transition_selected_mean=_masked_mean(ai_alpha, transition_mask),
        transition_selected_vs_gvm_mean_delta=_masked_mean(ai_alpha, transition_mask) - _masked_mean(gvm_alpha, transition_mask)
        if transition_mask.any()
        else None,
        subject_conf_mean=float(subject_conf.mean()),
        subject_conf_entity_mean=_masked_mean(subject_conf, entity_mask),
        subject_conf_effect_mean=_masked_mean(subject_conf, effect_mask),
        old_effect_mean=float(old_effect.mean()),
        new_effect_mean=float(new_effect.mean()),
        old_solid_mean=float(old_solid.mean()),
        new_solid_mean=float(new_solid.mean()),
        debug_crop_exported=bool(crop_debug),
        debug_crop_top=crop_debug["top"] if crop_debug else None,
        debug_crop_left=crop_debug["left"] if crop_debug else None,
        debug_crop_height=crop_debug["height"] if crop_debug else None,
        debug_crop_width=crop_debug["width"] if crop_debug else None,
        debug_crop_center_y=crop_debug["center_y"] if crop_debug else None,
        debug_crop_center_x=crop_debug["center_x"] if crop_debug else None,
        debug_crop_peak_abs_diff=crop_debug["peak_abs_diff"] if crop_debug else None,
        debug_effect_risk=debug_effect_risk,
    )
    artifacts["debug_crop"] = crop_debug
    return entry, artifacts


def _collect_input_diagnostics(
    matte: HybridMatte,
    input_name: str,
    samples: list[tuple[str, np.ndarray]],
) -> list[tuple[dict[str, float | int | str | bool | None], dict[str, np.ndarray]]]:
    sequence_frames = [frame for _sample_name, frame in samples]
    sequence_selection = _resolve_sequence_selection(matte, sequence_frames)
    results = []
    for index, (sample_name, frame) in enumerate(samples):
        results.append(
            _collect_sample_diagnostics(
                matte=matte,
                input_name=input_name,
                sample_name=sample_name,
                frame=frame,
                sequence_frames=sequence_frames,
                sequence_index=index,
                sequence_selection=sequence_selection,
            )
        )
    return results


def main() -> int:
    out_dir = PROJECT_ROOT / ".superpowers" / "diagnostics" / "gvm_fusion_selftest"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_out_dir = out_dir / "crops"
    crop_out_dir.mkdir(parents=True, exist_ok=True)

    input_paths = _resolve_default_input_paths(PROJECT_ROOT)

    config = MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7)
    matte = HybridMatte(config)
    if matte.gvm is None or matte.gvm.model is None:
        raise RuntimeError("GVM not available in current environment")

    summary: list[dict[str, float | int | str | bool | None]] = []
    for input_path in input_paths:
        sample_results = _collect_input_diagnostics(matte, input_path.name, _load_rgb_frames(input_path))
        for entry, artifacts in sample_results:
            sample_name = str(entry["sample"])
            _set_debug_crop_prefix(entry)
            summary.append(entry)

            _write_alpha(out_dir / f"{sample_name}_base.png", artifacts["base_alpha"])
            _write_alpha(out_dir / f"{sample_name}_gvm.png", artifacts["gvm_alpha"])
            _write_alpha(out_dir / f"{sample_name}_old_merge.png", artifacts["old_alpha"])
            _write_alpha(out_dir / f"{sample_name}_new_merge.png", artifacts["new_alpha"])
            _write_alpha(out_dir / f"{sample_name}_subject_conf.png", artifacts["subject_conf"])
            _write_alpha(out_dir / f"{sample_name}_subject_gate.png", artifacts["subject_gate"])
            _write_alpha(out_dir / f"{sample_name}_old_effect.png", artifacts["old_effect"])
            _write_alpha(out_dir / f"{sample_name}_new_effect.png", artifacts["new_effect"])

            diff = artifacts["diff"]
            diff_vis = np.zeros((*diff.shape, 3), dtype=np.uint8)
            pos = np.clip(np.maximum(diff, 0.0) * 255.0 * 3.0, 0, 255).astype(np.uint8)
            neg = np.clip(np.maximum(-diff, 0.0) * 255.0 * 3.0, 0, 255).astype(np.uint8)
            diff_vis[..., 2] = pos
            diff_vis[..., 0] = neg
            cv2.imwrite(str(out_dir / f"{sample_name}_diff.png"), diff_vis)

            crop_debug = artifacts.get("debug_crop")
            if crop_debug:
                crop_prefix_str = str(entry.get("debug_crop_prefix") or f"crops/{sample_name}_crop")
                crop_prefix = out_dir / crop_prefix_str
                _write_rgb(crop_prefix.with_name(f"{crop_prefix.name}_frame.png"), crop_debug["frame_rgb"])
                _write_alpha(crop_prefix.with_name(f"{crop_prefix.name}_base.png"), crop_debug["base_alpha"])
                _write_alpha(crop_prefix.with_name(f"{crop_prefix.name}_gvm.png"), crop_debug["gvm_alpha"])
                _write_alpha(crop_prefix.with_name(f"{crop_prefix.name}_old_merge.png"), crop_debug["old_alpha"])
                _write_alpha(crop_prefix.with_name(f"{crop_prefix.name}_new_merge.png"), crop_debug["new_alpha"])
                _write_alpha(crop_prefix.with_name(f"{crop_prefix.name}_subject_conf.png"), crop_debug["subject_conf"])
                _write_alpha(crop_prefix.with_name(f"{crop_prefix.name}_subject_gate.png"), crop_debug["subject_gate"])

                crop_diff = np.asarray(crop_debug["diff"], dtype=np.float32)
                crop_diff_vis = np.zeros((*crop_diff.shape, 3), dtype=np.uint8)
                crop_pos = np.clip(np.maximum(crop_diff, 0.0) * 255.0 * 3.0, 0, 255).astype(np.uint8)
                crop_neg = np.clip(np.maximum(-crop_diff, 0.0) * 255.0 * 3.0, 0, 255).astype(np.uint8)
                crop_diff_vis[..., 2] = crop_pos
                crop_diff_vis[..., 0] = crop_neg
                cv2.imwrite(str(crop_prefix.with_name(f"{crop_prefix.name}_diff.png")), crop_diff_vis)

    summary_payload = _build_summary_payload(summary)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))
    print(f"Wrote diagnostics to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
