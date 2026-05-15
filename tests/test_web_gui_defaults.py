import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def test_process_video_signature_only_exposes_effective_advanced_controls():
    params = set(inspect.signature(web_gui.process_video).parameters)

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

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def process(self, video_path, output_dir, progress_callback):
            captured["video_path"] = video_path
            captured["output_dir"] = output_dir
            captured["progress_callback"] = progress_callback
            processed_dir = Path(output_dir) / "Processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8), mode="RGBA").save(
                processed_dir / "processed_000000.png"
            )
            return {
                "frames_processed": 1,
                "fps": 1.0,
                "elapsed_time": 1.0,
            }

    monkeypatch.setattr(web_gui, "MattingPipeline", FakePipeline)
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
    preview, zip_file, status, *_ = result

    config = captured["config"]
    assert config.green_similarity == 0.4
    assert config.green_hair_detail == 0.8
    assert config.white_protect_brightness == 180
    assert config.white_protect_saturation == 25
    assert config.edge_despill_factor == 1.2
    assert config.screen_color == "auto"
    assert config.key_strength == 1.0
    assert config.clip_black == 0.0
    assert config.clip_white == 1.0
    assert config.shrink_grow == 0
    assert config.edge_blur == 0
    assert config.transparency_preserve == 0.65
    assert config.gvm_max_internal_size == 1024
    assert config.generate_zip_by_default is False
    assert zip_file is None
    assert preview is None
    assert "完成" in status
    assert result[-1].endswith("processed_000000.png")
