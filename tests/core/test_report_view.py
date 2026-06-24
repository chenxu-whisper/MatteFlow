import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.reporting import ProcessingReportViewBuilder  # noqa: E402


def _report_payload() -> dict:
    return {
        "schema_version": 1,
        "job": {
            "frame_count": 12,
            "background_mode_requested": "auto",
            "background_mode_effective": "green_screen",
            "quality_mode": "high",
            "ai_model_requested": "auto",
            "ai_model_active": "gvm",
        },
        "timings": {"total": 3.25, "decode": 0.1},
        "quality": {
            "overall_score": 0.913245,
            "mean_edge_uncertainty": 0.0412,
            "speckle_pixels": 7,
            "hole_pixels": 12,
            "background_residue": 0.0034,
            "temporal_flicker": 0.015,
        },
        "regions": {
            "subject_pixels": 123456,
            "hair_edge_pixels": 2345,
            "luminous_prop_pixels": 420,
            "transparent_effect_pixels": 188,
            "background_residue_pixels": 33,
            "uncertain_edge_pixels": 987,
        },
        "model_decisions": {
            "active_ai_model": "gvm",
            "fallback_quality_metrics": {"weighted_score": 0.82},
            "green_screen_layer_debug_available": True,
        },
        "fusion": {
            "available": True,
            "selected_by_region": {"subject": "ai_core", "luminous_prop": "effect_safe"},
            "rejected_takeovers": {"luminous_prop": 2},
        },
        "foreground_recovery": {
            "frames": 12,
            "attempted_pixels": 1024,
            "accepted_pixels": 735,
            "screen_rgb": [0.0, 210.0, 40.0],
        },
        "artifacts": {"processed_dir": "Processed"},
        "warnings": ["edge uncertainty is above target"],
    }


def test_report_view_builder_summarizes_processing_report_payload():
    view = ProcessingReportViewBuilder().from_payload(_report_payload(), report_path=Path("report.json"))

    assert view.report_path == Path("report.json")
    assert view.title == "处理诊断报告"
    assert "质量评分：0.913" in view.quality_summary
    assert "边缘不确定性：0.041" in view.quality_summary
    assert "孔洞像素：12" in view.quality_summary
    assert "请求模型：auto" in view.model_summary
    assert "实际模型：gvm" in view.model_summary
    assert "主体：123456 px" in view.region_summary
    assert "发光道具：420 px" in view.region_summary
    assert "接受像素：735 / 1024" in view.recovery_summary
    assert view.warnings == ("edge uncertainty is above target",)

    markdown = view.to_markdown()
    assert "### 处理诊断报告" in markdown
    assert "融合诊断" in markdown
    assert "subject: ai_core" in markdown
    assert "luminous_prop: 2" in markdown


def test_report_view_builder_loads_report_from_path(tmp_path):
    report_path = tmp_path / "processing_report.json"
    report_path.write_text(json.dumps(_report_payload()), encoding="utf-8")

    view = ProcessingReportViewBuilder().from_path(report_path)

    assert view.report_path == report_path
    assert "质量评分：0.913" in view.to_markdown()


def test_report_view_builder_returns_unavailable_view_for_missing_report(tmp_path):
    missing_path = tmp_path / "missing_report.json"

    view = ProcessingReportViewBuilder().from_path(missing_path)

    assert view.report_path == missing_path
    assert view.quality_summary == "处理诊断报告暂不可用。"
    assert view.model_summary == ""
    assert view.region_summary == ""
    assert view.recovery_summary == ""
    assert view.warnings == ()
    assert view.to_markdown() == "处理诊断报告暂不可用。"
