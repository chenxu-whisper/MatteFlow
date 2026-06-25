import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig  # noqa: E402
from scripts import web_gui  # noqa: E402


def test_gui_quality_selection_defaults_are_off():
    assert web_gui.GUI_DEFAULTS["quality_selection_enable"] is False
    assert web_gui.GUI_DEFAULTS["quality_birefnet_auto_load"] is False


def test_process_video_accepts_quality_selection_controls():
    signature = inspect.signature(web_gui.process_video)

    assert "quality_selection_enable" in signature.parameters
    assert "quality_birefnet_auto_load" in signature.parameters


def test_gui_process_job_params_preserve_quality_selection_flags(tmp_path):
    config = MattingConfig()
    config.quality_selection_enable = True
    config.quality_birefnet_auto_load = True

    params = web_gui._build_process_job_params(
        tmp_path / "input.png",
        tmp_path / "out",
        config,
    )

    assert params.quality_selection_enable is True
    assert params.quality_birefnet_auto_load is True
