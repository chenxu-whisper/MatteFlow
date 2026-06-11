import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _drop_gvm_wrapper_modules():
    for name in list(sys.modules):
        if name.startswith("matteflow.vendor.gvm_core"):
            sys.modules.pop(name)


def test_gvm_wrapper_imports_without_vendored_models_package():
    _drop_gvm_wrapper_modules()

    wrapper = importlib.import_module("matteflow.vendor.gvm_core.wrapper")

    assert wrapper.UNetSpatioTemporalConditionModel is not None


def test_model_checker_detects_gvm_runtime_when_wrapper_imports(monkeypatch, tmp_path):
    import torch
    from matteflow.utils import model_checker

    model_dir = tmp_path / "gvm"
    for child in ("unet", "vae", "scheduler"):
        (model_dir / child).mkdir(parents=True)

    _drop_gvm_wrapper_modules()
    monkeypatch.setattr(model_checker, "resolve_snapshot_model_dir", lambda *args: model_dir)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    status = model_checker.ModelChecker()._check_gvm()

    assert status["available"] is True
    assert status["reason"] is None
