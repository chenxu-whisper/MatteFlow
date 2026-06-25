import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.region_ownership import RegionOwnership  # noqa: E402
from matteflow.config import BackgroundMode, MattingConfig, QualityMode  # noqa: E402
from matteflow.pipeline import MattingPipeline  # noqa: E402


def test_pipeline_writes_processing_report_and_returns_path(monkeypatch, tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.FAST,
        output_fg=False,
        output_matte=False,
        output_comp=False,
        output_processed=False,
        output_debug=False,
    )
    pipeline = MattingPipeline(config)
    frame = np.full((4, 4, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[1:3, 1:3] = 1.0

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: ([frame], {"width": 4, "height": 4, "fps": 1.0}),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda frames_arg, bg_mode, progress_callback, cancel_check=None: [alpha],
    )
    monkeypatch.setattr(
        pipeline.decontaminate,
        "process",
        lambda frames_arg, alphas_arg, bg_mode, context=None: frames_arg,
    )

    result = pipeline.process(tmp_path / "input.png", tmp_path / "out")

    report_path = tmp_path / "out" / "processing_report.json"
    assert result["processing_report_path"] == str(report_path)
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["job"]["background_mode_effective"] == "green_screen"
    assert payload["job"]["frame_count"] == 1
    assert payload["quality"]["frame_count"] == 1
    assert set(payload["p0_risks"]) == {
        "hair_edge_loss",
        "background_residue",
        "light_subject_loss",
        "transparent_effect_loss",
        "temporal_instability",
        "subject_misidentification",
    }
    assert payload["p0_risks"]["background_residue"]["signals"]["quality_background_residue"] >= 0.0
    assert payload["regions"]["subject_pixels"] >= 4
    assert payload["foreground_recovery"] == {}
    assert payload["timings"]["total"] > 0.0
    assert payload["timings"]["processing_report"] > 0.0


def test_pipeline_reuses_matte_stage_base_alpha_when_rebuilding_region_context(monkeypatch, tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.STANDARD,
        output_fg=False,
        output_matte=False,
        output_comp=False,
        output_processed=False,
        output_debug=False,
    )
    pipeline = MattingPipeline(config)
    frame = np.full((4, 4, 3), [0, 220, 40], dtype=np.uint8)
    base_alpha = np.zeros((4, 4), dtype=np.float32)
    base_alpha[1:3, 1:3] = 0.8
    refined_alpha = np.zeros((4, 4), dtype=np.float32)
    captured_base_alphas = []

    def analyze_with_capture(frame_arg, alpha_arg, base_alpha_arg=None):
        captured_base_alphas.append(None if base_alpha_arg is None else base_alpha_arg.copy())
        empty = np.zeros(alpha_arg.shape, dtype=bool)
        return RegionOwnership(
            subject=alpha_arg >= 0.70,
            hair_edge=empty,
            luminous_prop=empty,
            transparent_effect=empty,
            background_residue=empty,
            uncertain_edge=empty,
        )

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: ([frame], {"width": 4, "height": 4, "fps": 1.0}),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda frames_arg, bg_mode, progress_callback, cancel_check=None: [base_alpha],
    )
    monkeypatch.setattr(pipeline.region_analyzer, "analyze", analyze_with_capture)
    monkeypatch.setattr(pipeline.refiner, "refine", lambda frames_arg, alphas_arg, context=None: [refined_alpha])
    monkeypatch.setattr(pipeline.despeckle, "process", lambda alphas_arg, frames=None, context=None: alphas_arg)
    monkeypatch.setattr(
        pipeline.effect_prop_repair,
        "process",
        lambda frames_arg, alphas_arg, bg_mode, active_model=None, context=None: alphas_arg,
    )
    monkeypatch.setattr(
        pipeline.decontaminate,
        "process",
        lambda frames_arg, alphas_arg, bg_mode, context=None: frames_arg,
    )

    pipeline.process(tmp_path / "input.png", tmp_path / "out")

    assert captured_base_alphas
    assert all(item is not None for item in captured_base_alphas)
    assert all(np.array_equal(item, base_alpha) for item in captured_base_alphas)


def test_pipeline_reports_quality_selection_debug_artifacts_when_debug_enabled(monkeypatch, tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.FAST,
        output_fg=False,
        output_matte=False,
        output_comp=False,
        output_processed=False,
        output_debug=True,
    )
    pipeline = MattingPipeline(config)
    frame = np.full((4, 4, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[1:3, 1:3] = 1.0
    pipeline.hybrid_matte.last_quality_selection = {
        "available": True,
        "candidate_count": 1,
        "selected_model_counts": {"traditional": 1},
        "candidate_quality": {"traditional": {"overall_score": 0.9}},
        "skipped_candidates": [],
    }

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: ([frame], {"width": 4, "height": 4, "fps": 1.0}),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda frames_arg, bg_mode, progress_callback, cancel_check=None: [alpha],
    )
    monkeypatch.setattr(
        pipeline.decontaminate,
        "process",
        lambda frames_arg, alphas_arg, bg_mode, context=None: frames_arg,
    )

    pipeline.process(tmp_path / "input.png", tmp_path / "out")

    report_path = tmp_path / "out" / "processing_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["quality_selection_debug_dir"] == "debug\\quality_selection"
    assert payload["artifacts"]["quality_selection_contact_sheet"] == (
        "debug\\quality_selection\\quality_selection_contact_sheet.png"
    )
    assert (tmp_path / "out" / "debug" / "quality_selection" / "quality_selection_contact_sheet.png").exists()
