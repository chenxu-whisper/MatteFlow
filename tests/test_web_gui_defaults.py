import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.diagnostics import DiagnosticCode, DiagnosticItem, DiagnosticReport, DiagnosticSeverity
from matteflow.job_queue import GPUJobQueue
from matteflow.service import ProcessResult
from scripts import web_gui


def test_gui_defaults_match_checker_000040_preset():
    defaults = web_gui.GUI_DEFAULTS

    assert defaults["mode"] == "green"
    assert defaults["quality"] == "standard"
    assert defaults["preferred_ai"] == "gvm"
    assert defaults["pure_color"] is True
    assert defaults["use_filter"] is False
    assert defaults["green_similarity"] == 0.4
    assert defaults["green_despill"] == 0.7
    assert defaults["green_hair"] == 0.8
    assert defaults["white_protect_brightness"] == 180
    assert defaults["white_protect_saturation"] == 25
    assert defaults["edge_despill_factor"] == 1.2
    assert defaults["screen_color"] == "auto"
    assert defaults["key_strength"] == 1.0
    assert defaults["clip_black"] == 0.0
    assert defaults["clip_white"] == 1.0
    assert defaults["shrink_grow"] == 0
    assert defaults["edge_blur"] == 0
    assert defaults["despill_enable"] is True
    assert defaults["despill_strength"] == 0.7
    assert defaults["despill_color"] == "green"
    assert defaults["despeckle_enable"] is True
    assert defaults["despeckle_radius"] == 2
    assert defaults["despeckle_threshold"] == 0.0
    assert defaults["transparency_preserve"] == 0.7
    assert defaults["gvm_max_internal_size"] == 768
    assert defaults["auto_optimize"] is False
    assert defaults["generate_zip"] is False
    assert defaults["output_matte"] is True
    assert defaults["output_comp"] is False
    assert defaults["output_processed"] is True


def test_gui_default_ai_choice_prefers_gvm_when_available(monkeypatch):
    monkeypatch.setattr(
        web_gui,
        "_ui_choices",
        [("MatAnyone2", "matanyone2"), ("GVM", "gvm"), ("传统算法", "traditional")],
    )

    assert web_gui._default_ai_choice() == "gvm"


def test_gui_default_ai_choice_falls_back_when_gvm_unavailable(monkeypatch):
    monkeypatch.setattr(
        web_gui,
        "_ui_choices",
        [("MatAnyone2", "matanyone2"), ("传统算法", "traditional")],
    )

    assert web_gui._default_ai_choice() == "matanyone2"


def test_apply_recommended_preset_uses_checker_000040_values(monkeypatch):
    monkeypatch.setattr(
        web_gui,
        "_ui_choices",
        [("MatAnyone2", "matanyone2"), ("GVM", "gvm"), ("传统算法", "traditional")],
    )

    preset = web_gui._apply_recommended_preset()

    assert preset["mode"] == "green"
    assert preset["quality"] == "standard"
    assert preset["use_ai"] == "gvm"
    assert preset["pure_color_mode"] is True
    assert preset["use_guided_filter"] is False
    assert preset["green_similarity"] == 0.4
    assert preset["green_despill"] == 0.7
    assert preset["green_hair"] == 0.8
    assert preset["white_protect_thresh"] == 180
    assert preset["white_protect_sat"] == 25
    assert preset["edge_despill_factor"] == 1.2
    assert preset["screen_color"] == "auto"
    assert preset["key_strength"] == 1.0
    assert preset["clip_black"] == 0.0
    assert preset["clip_white"] == 1.0
    assert preset["shrink_grow"] == 0
    assert preset["edge_blur"] == 0
    assert preset["despill_enable"] is True
    assert preset["despill_strength"] == 0.7
    assert preset["despill_color"] == "green"
    assert preset["despeckle_enable"] is True
    assert preset["despeckle_radius"] == 2
    assert preset["despeckle_threshold"] == 0.0
    assert preset["transparency_preserve"] == 0.7
    assert preset["gvm_max_internal_size"] == 768
    assert preset["auto_optimize"] is False
    assert preset["generate_zip"] is False
    assert preset["output_fg"] is False
    assert preset["output_matte"] is True
    assert preset["output_comp"] is False
    assert preset["output_processed"] is True


def test_recommended_preset_updates_include_all_target_controls(monkeypatch):
    monkeypatch.setattr(
        web_gui,
        "_ui_choices",
        [("MatAnyone2", "matanyone2"), ("GVM", "gvm"), ("传统算法", "traditional")],
    )

    updates = web_gui._recommended_preset_updates()

    assert len(updates) == len(web_gui.RECOMMENDED_PRESET_OUTPUT_KEYS)
    assert updates[0]["value"] == "green"
    assert updates[1]["value"] == "standard"
    assert updates[2]["value"] == "gvm"
    assert any(update["value"] is False for update in updates)


def test_gui_primary_controls_are_limited_to_core_effect_controls():
    assert web_gui.GUI_PRIMARY_CONTROL_KEYS == [
        "use_ai",
        "quality",
        "key_strength",
        "transparency_preserve",
        "green_despill",
        "edge_despill_factor",
        "shrink_grow",
        "edge_blur",
        "gvm_max_internal_size",
    ]

    assert "clip_black" not in web_gui.GUI_PRIMARY_CONTROL_KEYS
    assert "clip_white" not in web_gui.GUI_PRIMARY_CONTROL_KEYS
    assert "white_protect_brightness" not in web_gui.GUI_PRIMARY_CONTROL_KEYS
    assert "despeckle_threshold" not in web_gui.GUI_PRIMARY_CONTROL_KEYS


def test_gui_fixed_low_value_controls_keep_safe_defaults():
    fixed = web_gui.GUI_FIXED_PARAMETER_DEFAULTS

    assert fixed["pure_color"] is True
    assert fixed["use_filter"] is False
    assert fixed["edge_softness"] == 0.0
    assert fixed["temporal_strength"] == 0.5
    assert fixed["color_space"] == "sRGB"
    assert fixed["despill_enable"] is True
    assert fixed["despill_color"] == "green"


def test_format_diagnostic_report_prioritizes_errors():
    report = DiagnosticReport(
        items=(
            DiagnosticItem(
                code=DiagnosticCode.GPU_OUT_OF_MEMORY,
                severity=DiagnosticSeverity.ERROR,
                title="GPU 显存不足",
                summary="当前任务执行时显存不足。",
                actions=("关闭其他 GPU 程序",),
                blocking=True,
            ),
            DiagnosticItem(
                code=DiagnosticCode.MODEL_MISSING,
                severity=DiagnosticSeverity.WARNING,
                title="模型缺失",
                summary="部分模型当前不可用。",
                actions=("检查模型目录",),
                blocking=False,
            ),
        )
    )

    text = web_gui._format_diagnostic_report(report)

    assert "GPU 显存不足" in text
    assert "关闭其他 GPU 程序" in text
    assert "模型缺失" in text


def test_collect_environment_diagnostics_returns_blocking_report(monkeypatch):
    from matteflow.ffmpeg_env import MediaToolDiscoveryResult

    monkeypatch.setattr(
        web_gui,
        "discover_media_tools",
        lambda: MediaToolDiscoveryResult(
            ffmpeg_path=None,
            ffprobe_path=None,
            bin_dir=None,
            source=None,
            complete=False,
            download_required=True,
        ),
    )

    class FakeChecker:
        def collect_model_facts(self):
            return {}

    report = web_gui._collect_environment_diagnostics(model_checker=FakeChecker())

    assert report.ok is False
    assert report.blocking_count == 1


def test_process_video_signature_only_exposes_effective_advanced_controls():
    params = set(inspect.signature(web_gui.process_video).parameters)

    assert "auto_optimize" in params
    assert "transparency_preserve" in params
    assert "gvm_max_internal_size" in params
    assert "generate_zip" in params

    assert "edge_refinement" not in params
    assert "edge_radius" not in params
    assert "edge_threshold" not in params
    assert "temporal_enable" not in params
    assert "temporal_window" not in params
    assert "temporal_threshold" not in params
    assert "output_format" not in params
    assert "output_premultiply" not in params
    assert "output_invert" not in params
    assert "bgm2_background" not in params


def test_process_video_applies_effective_advanced_controls(monkeypatch, tmp_path):
    captured = {}

    class FakeService:
        def process(self, params, progress_callback=None, cancel_check=None):
            captured["params"] = params
            captured["progress_callback"] = progress_callback
            captured["cancel_check"] = cancel_check
            processed_dir = Path(params.output_dir) / "Processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8), mode="RGBA").save(
                processed_dir / "processed_000000.png"
            )
            return ProcessResult(
                success=True,
                input_path=params.input_path,
                output_dir=params.output_dir,
                background_mode="green_screen",
                frame_count=3,
                processing_time=1.5,
                timings={"fps": 2.0},
            )

    monkeypatch.setattr(web_gui, "_resolve_gui_output_dir", lambda video_path: tmp_path)
    monkeypatch.setattr(web_gui, "_create_preview_video", lambda output_dir, preview_path: None)
    monkeypatch.setattr(web_gui, "_create_preview_frames", lambda output_dir: (None, None))

    def fail_if_zip_created(output_dir, zip_path):
        raise AssertionError("ZIP should not be generated when generate_zip is False")

    monkeypatch.setattr(web_gui, "_create_zip", fail_if_zip_created)

    result = web_gui.process_video(
        video_path=str(tmp_path / "input.png"),
        mode="green",
        quality="standard",
        use_ai="gvm",
        pure_color_mode=True,
        use_guided_filter=False,
        green_similarity=0.4,
        green_despill=0.7,
        green_hair=0.8,
        white_protect_thresh=180,
        white_protect_sat=25,
        edge_despill_factor=1.2,
        black_threshold=0.03,
        black_glow=0.9,
        black_particle=0.7,
        edge_softness=0.0,
        temporal_strength=0.5,
        transparency_preserve=0.65,
        gvm_max_internal_size=1024,
        auto_optimize=False,
        screen_color="auto",
        key_strength=1.0,
        clip_black=0.0,
        clip_white=1.0,
        shrink_grow=0,
        edge_blur=0,
        despill_enable=True,
        despill_strength=0.7,
        despill_color="green",
        despeckle_enable=True,
        despeckle_radius=2,
        despeckle_threshold=0.0,
        color_space="sRGB",
        output_fg=False,
        output_matte=True,
        output_comp=False,
        output_processed=True,
        generate_zip=False,
        ai_gamma=0.8,
        ai_threshold=0.1,
        ai_gain=1.2,
        ai_sharpen=0.0,
        service_factory=lambda: FakeService(),
    )
    preview, zip_file, status, *_ = result

    overrides = captured["params"].config_overrides
    assert overrides["green_similarity"] == 0.4
    assert overrides["green_hair_detail"] == 0.8
    assert overrides["white_protect_brightness"] == 180
    assert overrides["white_protect_saturation"] == 25
    assert overrides["edge_despill_factor"] == 1.2
    assert overrides["screen_color"] == "auto"
    assert overrides["key_strength"] == 1.0
    assert overrides["clip_black"] == 0.0
    assert overrides["clip_white"] == 1.0
    assert overrides["shrink_grow"] == 0
    assert overrides["edge_blur"] == 0
    assert overrides["transparency_preserve"] == 0.65
    assert overrides["gvm_max_internal_size"] == 1024
    assert overrides["generate_zip_by_default"] is False
    assert zip_file is None
    assert preview is None
    assert "完成" in status
    assert "3帧" in status
    assert "耗时 1.5s" in status
    assert "2.0 fps" in status
    assert result[-1].endswith("processed_000000.png")
    assert result[5] == 3


def test_process_video_applies_auto_optimized_input_params(monkeypatch, tmp_path):
    captured = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def process(self, video_path, output_dir, progress_callback):
            processed_dir = Path(output_dir) / "Processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8), mode="RGBA").save(
                processed_dir / "processed_000000.png"
            )
            return {"frames_processed": 1, "fps": 1.0, "elapsed_time": 1.0}

    monkeypatch.setattr(web_gui, "MattingPipeline", FakePipeline)
    monkeypatch.setattr(web_gui, "_resolve_gui_output_dir", lambda video_path: tmp_path)
    monkeypatch.setattr(web_gui, "_create_preview_video", lambda output_dir, preview_path: None)
    monkeypatch.setattr(web_gui, "_create_preview_frames", lambda output_dir: (None, None))

    image_path = tmp_path / "input.png"
    Image.fromarray(np.full((4, 4, 3), [20, 185, 55], dtype=np.uint8), mode="RGB").save(image_path)

    result = web_gui.process_video(
        video_path=str(image_path),
        mode="green",
        quality="standard",
        use_ai="gvm",
        pure_color_mode=True,
        use_guided_filter=False,
        green_similarity=0.4,
        green_despill=0.7,
        green_hair=0.8,
        white_protect_thresh=180,
        white_protect_sat=25,
        edge_despill_factor=1.2,
        black_threshold=0.03,
        black_glow=0.9,
        black_particle=0.7,
        edge_softness=0.0,
        temporal_strength=0.5,
        transparency_preserve=0.7,
        gvm_max_internal_size=768,
        auto_optimize=True,
        screen_color="auto",
        key_strength=1.0,
        clip_black=0.0,
        clip_white=1.0,
        shrink_grow=0,
        edge_blur=0,
        despill_enable=True,
        despill_strength=0.7,
        despill_color="green",
        despeckle_enable=True,
        despeckle_radius=2,
        despeckle_threshold=0.0,
        color_space="sRGB",
        output_fg=False,
        output_matte=True,
        output_comp=False,
        output_processed=True,
        generate_zip=False,
        ai_gamma=0.8,
        ai_threshold=0.1,
        ai_gain=1.2,
        ai_sharpen=0.0,
    )

    assert captured["config"].screen_color == "green"
    status = result[2]
    assert "自动优化" in status
    assert "本次实际参数" in status
    assert "screen=green" in status
    assert "similarity=0.35" in status
    assert "key=1.00" in status
    assert "preserve=0.70" in status
    assert "despill=0.70" in status
    assert "edge_despill=1.20" in status
    assert "clip=0.00/1.00" in status
    assert "white_protect=180/25" in status
    assert "shrink_grow=0" in status
    assert "edge_blur=0" in status
    assert "gvm_size=768" in status


def test_process_video_formats_failures_with_diagnostics(monkeypatch, tmp_path):
    class FailingService:
        def process(self, params, progress_callback=None, cancel_check=None):
            raise RuntimeError("decoder exploded")

    local_queue = GPUJobQueue()
    monkeypatch.setattr(web_gui, "_resolve_gui_output_dir", lambda video_path: tmp_path)

    result = web_gui.process_video(
        video_path=str(tmp_path / "input.png"),
        mode="green",
        quality="standard",
        use_ai="gvm",
        pure_color_mode=True,
        use_guided_filter=False,
        green_similarity=0.4,
        green_despill=0.7,
        green_hair=0.8,
        white_protect_thresh=180,
        white_protect_sat=25,
        edge_despill_factor=1.2,
        black_threshold=0.03,
        black_glow=0.9,
        black_particle=0.7,
        edge_softness=0.0,
        temporal_strength=0.5,
        transparency_preserve=0.65,
        gvm_max_internal_size=1024,
        auto_optimize=False,
        screen_color="auto",
        key_strength=1.0,
        clip_black=0.0,
        clip_white=1.0,
        shrink_grow=0,
        edge_blur=0,
        despill_enable=True,
        despill_strength=0.7,
        despill_color="green",
        despeckle_enable=True,
        despeckle_radius=2,
        despeckle_threshold=0.0,
        color_space="sRGB",
        output_fg=False,
        output_matte=True,
        output_comp=False,
        output_processed=True,
        generate_zip=False,
        ai_gamma=0.8,
        ai_threshold=0.1,
        ai_gain=1.2,
        ai_sharpen=0.0,
        service_factory=lambda: FailingService(),
        queue_factory=lambda: local_queue,
    )

    status = result[2]
    assert "**ERROR** 处理失败" in status
    assert "MatteFlow 在处理任务时发生未分类错误。" in status
    assert "- 查看日志输出" in status
