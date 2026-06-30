import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.alpha_quality import AlphaQualityReport  # noqa: E402
from matteflow.analysis.p0_quality import P0QualityAnalyzer  # noqa: E402
from matteflow.analysis.region_ownership import RegionOwnership  # noqa: E402
from matteflow.config import BackgroundMode, MattingConfig, QualityMode  # noqa: E402
from matteflow.reporting import ProcessingReportBuilder, ProcessingReportWriter  # noqa: E402
from matteflow.reporting.report_view import (  # noqa: E402
    ProcessingReportViewBuilder,
    format_quality_selection_summary,
)


def test_builder_creates_required_sections_and_serializable_values(tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.AUTO,
        quality_mode=QualityMode.HIGH,
        ai_model="auto",
    )
    quality_report = AlphaQualityReport(
        frame_count=2,
        mean_edge_uncertainty=np.float32(0.125),
        speckle_pixels=np.int64(3),
        hole_pixels=np.int64(4),
        background_residue=np.float32(0.01),
        temporal_flicker=np.float32(0.02),
        overall_score=np.float32(0.9),
    )
    ownership = RegionOwnership(
        subject=np.array([[True, False], [True, True]]),
        hair_edge=np.array([[False, True], [False, False]]),
        luminous_prop=np.array([[False, False], [True, False]]),
        transparent_effect=np.array([[False, True], [True, False]]),
        background_residue=np.array([[False, False], [False, True]]),
        uncertain_edge=np.array([[True, True], [False, False]]),
    )
    p0_quality_report = P0QualityAnalyzer().analyze_sequence(
        [np.full((2, 2, 3), 230, dtype=np.uint8)],
        [np.full((2, 2), 0.1, dtype=np.float32)],
        quality_report=quality_report,
        region_context={"region_ownership": [ownership]},
    )
    hybrid_matte = SimpleNamespace(
        last_active_ai_model="gvm",
        black_matte=SimpleNamespace(
            last_effect_enhancement={
                "smoke_pixels": np.int64(12),
                "glow_pixels": np.int64(34),
                "particle_pixels": np.int64(5),
                "subject_edge_pixels": np.int64(6),
                "black_residue_suppressed_pixels": np.int64(78),
                "mean_alpha_delta": np.float32(0.125),
            }
        ),
        last_fallback_quality_metrics={"weighted_score": np.float32(0.82)},
        last_fusion_quality_gate_diagnostics={
            "risk_guard": {
                "triggered": np.bool_(True),
                "reasons": ["hole_pixels"],
            }
        },
        green_screen_layer_debug={"base": object()},
        last_quality_selection={
            "available": True,
            "candidate_count": np.int64(2),
            "selected_model_counts": {"traditional": np.int64(3)},
            "candidate_quality": {"traditional": {"overall_score": np.float32(0.8)}},
            "skipped_candidates": [],
        },
    )
    decontaminate_context = {
        "foreground_recovery": {
            "frames": np.int64(2),
            "attempted_pixels": np.int64(10),
            "accepted_pixels": np.int64(7),
            "screen_rgb": np.array([0.0, 210.0, 40.0], dtype=np.float32),
        },
        "fusion": {
            "available": np.bool_(True),
            "selected_by_region": {"subject": "ai_core"},
            "rejected_takeovers": {"luminous_prop": np.int64(2)},
        },
    }

    report = ProcessingReportBuilder().build(
        input_path=Path("input.mp4"),
        output_dir=tmp_path,
        config=config,
        frame_count=2,
        background_mode_effective=BackgroundMode.GREEN_SCREEN,
        timings={"decode": np.float32(0.1), "total": np.float64(1.2)},
        quality_report=quality_report,
        p0_quality_report=p0_quality_report,
        region_context={"region_ownership": [ownership]},
        hybrid_matte=hybrid_matte,
        decontaminate_context=decontaminate_context,
        artifacts={"processed_output": tmp_path / "processed.png"},
    )

    payload = report.to_dict()
    assert set(payload) == {
        "schema_version",
        "job",
        "timings",
        "quality",
        "p0_risks",
        "regions",
        "region_supervision",
        "model_decisions",
        "fusion",
        "quality_selection",
        "edge_reconstruction",
        "black_effect_enhancement",
        "foreground_recovery",
        "artifacts",
        "warnings",
    }
    assert payload["schema_version"] == 2
    assert payload["job"]["background_mode_requested"] == "auto"
    assert payload["job"]["background_mode_effective"] == "green_screen"
    assert payload["job"]["quality_mode"] == "high"
    assert payload["job"]["ai_model_active"] == "gvm"
    assert payload["quality"]["speckle_pixels"] == 3
    assert payload["p0_risks"]["light_subject_loss"]["level"] == "fail"
    assert payload["p0_risks"]["background_residue"]["signals"]["quality_background_residue"] == 0.01
    assert payload["regions"]["subject_pixels"] == 3
    assert payload["regions"]["transparent_effect_pixels"] == 2
    assert payload["region_supervision"]["region_pixels"]["hair_edge"] == 1
    assert payload["region_supervision"]["failures"] == []
    assert payload["model_decisions"]["fallback_quality_metrics"]["weighted_score"] == 0.82
    assert payload["model_decisions"]["fusion_quality_gate"]["risk_guard"]["triggered"] is True
    assert payload["model_decisions"]["fusion_quality_gate"]["risk_guard"]["reasons"] == ["hole_pixels"]
    assert payload["model_decisions"]["green_screen_layer_debug_available"] is True
    assert payload["quality_selection"]["available"] is True
    assert payload["quality_selection"]["candidate_count"] == 2
    assert payload["quality_selection"]["selected_model_counts"]["traditional"] == 3
    assert payload["fusion"]["available"] is True
    assert payload["fusion"]["rejected_takeovers"]["luminous_prop"] == 2
    assert payload["black_effect_enhancement"] == {
        "smoke_pixels": 12,
        "glow_pixels": 34,
        "particle_pixels": 5,
        "subject_edge_pixels": 6,
        "black_residue_suppressed_pixels": 78,
        "mean_alpha_delta": 0.125,
    }
    assert payload["foreground_recovery"]["screen_rgb"] == [0.0, 210.0, 40.0]
    assert payload["artifacts"]["processed_output"] == "processed.png"

    json.dumps(payload)


def test_writer_writes_stable_processing_report_json(tmp_path):
    report = ProcessingReportBuilder().build(
        input_path=Path("input.png"),
        output_dir=tmp_path,
        config=MattingConfig(),
        frame_count=1,
        background_mode_effective=BackgroundMode.UNKNOWN,
        timings={},
        quality_report=None,
        region_context=None,
        hybrid_matte=None,
        decontaminate_context=None,
        artifacts={},
    )

    report_path = ProcessingReportWriter().write(report, tmp_path)

    assert report_path == tmp_path / "processing_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["job"]["input_path"] == "input.png"
    assert payload["fusion"] == {
        "available": False,
        "selected_by_region": {},
        "rejected_takeovers": {},
    }
    assert payload["quality_selection"] == {
        "available": False,
        "candidate_count": 0,
        "selected_model_counts": {},
        "candidate_quality": {},
        "skipped_candidates": [],
    }
    assert payload["region_supervision"]["region_pixels"]["subject"] == 0
    assert payload["edge_reconstruction"] == {}
    assert report_path.read_text(encoding="utf-8").endswith("\n")


def test_builder_handles_missing_optional_diagnostics(tmp_path):
    config = MattingConfig(quality_mode=QualityMode.FAST, ai_model="birefnet")

    report = ProcessingReportBuilder().build(
        input_path=tmp_path / "input.png",
        output_dir=tmp_path,
        config=config,
        frame_count=0,
        background_mode_effective=BackgroundMode.UNKNOWN,
        timings=None,
        quality_report=None,
        region_context={},
        hybrid_matte=SimpleNamespace(),
        decontaminate_context={},
        artifacts={"debug_dir": None},
    )

    payload = report.to_dict()
    assert payload["quality"] == {
        "frame_count": 0,
        "overall_score": None,
        "mean_edge_uncertainty": None,
        "speckle_pixels": 0,
        "hole_pixels": 0,
        "background_residue": None,
        "temporal_flicker": None,
        "edge_temporal_flicker": None,
        "transparent_temporal_flicker": None,
        "max_frame_delta": None,
    }
    assert set(payload["p0_risks"]) == {
        "hair_edge_loss",
        "background_residue",
        "light_subject_loss",
        "transparent_effect_loss",
        "temporal_instability",
        "subject_misidentification",
    }
    assert payload["p0_risks"]["hair_edge_loss"]["level"] == "pass"
    assert payload["regions"] == {
        "subject_pixels": 0,
        "hair_edge_pixels": 0,
        "luminous_prop_pixels": 0,
        "transparent_effect_pixels": 0,
        "background_residue_pixels": 0,
        "uncertain_edge_pixels": 0,
    }
    assert payload["region_supervision"]["total_pixels"] == 0
    assert payload["edge_reconstruction"] == {}
    assert payload["black_effect_enhancement"] == {}
    assert payload["model_decisions"]["active_ai_model"] is None
    assert payload["quality_selection"]["available"] is False
    assert payload["foreground_recovery"] == {}
    assert payload["artifacts"] == {}
    assert payload["warnings"] == []


def test_builder_aggregates_black_effect_enhancement_history(tmp_path):
    config = MattingConfig(background_mode=BackgroundMode.BLACK_BACKGROUND)
    hybrid_matte = SimpleNamespace(
        black_matte=SimpleNamespace(
            effect_enhancement_history=[
                {
                    "smoke_pixels": 2,
                    "glow_pixels": 3,
                    "particle_pixels": 1,
                    "subject_edge_pixels": 4,
                    "black_residue_suppressed_pixels": 5,
                    "mean_alpha_delta": 0.10,
                },
                {
                    "smoke_pixels": 7,
                    "glow_pixels": 11,
                    "particle_pixels": 13,
                    "subject_edge_pixels": 17,
                    "black_residue_suppressed_pixels": 19,
                    "mean_alpha_delta": 0.30,
                },
            ]
        )
    )

    report = ProcessingReportBuilder().build(
        input_path=Path("input.mp4"),
        output_dir=tmp_path,
        config=config,
        frame_count=2,
        background_mode_effective=BackgroundMode.BLACK_BACKGROUND,
        timings={},
        quality_report=None,
        hybrid_matte=hybrid_matte,
    )

    assert report.to_dict()["black_effect_enhancement"] == {
        "frames": 2,
        "smoke_pixels": 9,
        "glow_pixels": 14,
        "particle_pixels": 14,
        "subject_edge_pixels": 21,
        "black_residue_suppressed_pixels": 24,
        "mean_alpha_delta": 0.2,
    }


def test_report_view_formats_quality_selection_summary():
    summary = format_quality_selection_summary(
        {
            "quality_selection": {
                "available": True,
                "candidate_count": 2,
                "selected_model_counts": {"matanyone2": 4},
                "skipped_candidates": [{"name": "sam2", "reason": "guidance_missing"}],
            }
        }
    )

    assert "质量选择: 已启用" in summary
    assert "候选数量: 2" in summary
    assert "matanyone2: 4" in summary
    assert "sam2: guidance_missing" in summary


def test_report_view_includes_quality_selection_summary_in_markdown():
    view = ProcessingReportViewBuilder().from_payload(
        {
            "quality": {},
            "job": {},
            "timings": {},
            "regions": {},
            "foreground_recovery": {},
            "fusion": {},
            "quality_selection": {
                "available": True,
                "candidate_count": 1,
                "selected_model_counts": {"traditional": 3},
            },
        }
    )

    markdown = view.to_markdown()
    assert "质量选择: 已启用" in markdown
    assert "traditional: 3" in markdown


def test_report_view_includes_black_effect_enhancement_summary():
    view = ProcessingReportViewBuilder().from_payload(
        {
            "quality": {},
            "job": {},
            "timings": {},
            "regions": {},
            "foreground_recovery": {},
            "fusion": {},
            "quality_selection": {},
            "black_effect_enhancement": {
                "smoke_pixels": 12,
                "glow_pixels": 34,
                "particle_pixels": 5,
                "subject_edge_pixels": 6,
                "black_residue_suppressed_pixels": 78,
                "mean_alpha_delta": 0.125,
            },
        }
    )

    markdown = view.to_markdown()
    assert "黑底增强" in markdown
    assert "烟雾像素：12" in markdown
    assert "平均 alpha 变化：0.125" in markdown


def test_report_view_includes_temporal_detail_quality_metrics():
    view = ProcessingReportViewBuilder().from_payload(
        {
            "quality": {
                "edge_temporal_flicker": 0.12,
                "transparent_temporal_flicker": 0.08,
                "max_frame_delta": 0.20,
            },
            "job": {},
            "timings": {},
            "regions": {},
            "foreground_recovery": {},
            "fusion": {},
            "quality_selection": {},
        }
    )

    markdown = view.to_markdown()
    assert "边缘时序闪烁：0.120" in markdown
    assert "半透明时序闪烁：0.080" in markdown
    assert "最大帧差：0.200" in markdown


def test_report_view_includes_region_supervision_and_edge_reconstruction():
    view = ProcessingReportViewBuilder().from_payload(
        {
            "quality": {},
            "job": {},
            "timings": {},
            "regions": {},
            "region_supervision": {
                "region_pixels": {"hair_edge": 3},
                "region_ratios": {"hair_edge": 0.125},
                "failures": ["required_region transparent_effect missing"],
            },
            "edge_reconstruction": {
                "changed_pixels": 4,
                "protected_pixels": 2,
                "mean_delta": 0.05,
            },
            "foreground_recovery": {},
            "fusion": {},
            "quality_selection": {},
        }
    )

    markdown = view.to_markdown()
    assert "区域弱监督" in markdown
    assert "hair_edge：3 px / 0.125" in markdown
    assert "required_region transparent_effect missing" in markdown
    assert "边缘重建" in markdown
    assert "变更像素：4" in markdown
