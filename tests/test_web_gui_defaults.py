import sys
from pathlib import Path


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
