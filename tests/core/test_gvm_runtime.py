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


def test_gvm_processor_disables_low_cpu_mem_usage_for_diffusers_unet_fallback(
    monkeypatch, tmp_path
):
    _drop_gvm_wrapper_modules()
    wrapper = importlib.import_module("matteflow.vendor.gvm_core.wrapper")

    captured = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            captured[cls.__name__] = {"args": args, "kwargs": kwargs}
            return cls()

        def to(self, *args, **kwargs):
            return self

    class FakeVAE(FakeModel):
        pass

    class FakeScheduler(FakeModel):
        pass

    class FakeDiffusersUNet(FakeModel):
        pass

    class FakePipe:
        def __init__(self, vae, unet, scheduler):
            self.vae = vae
            self.unet = unet
            self.scheduler = scheduler

        def to(self, *args, **kwargs):
            return self

    model_dir = tmp_path / "gvm"
    for child in ("unet", "vae", "scheduler"):
        (model_dir / child).mkdir(parents=True)

    monkeypatch.setattr(wrapper, "AutoencoderKLTemporalDecoder", FakeVAE)
    monkeypatch.setattr(wrapper, "FlowMatchEulerDiscreteScheduler", FakeScheduler)
    monkeypatch.setattr(wrapper, "UNetSpatioTemporalConditionModel", FakeDiffusersUNet)
    monkeypatch.setattr(wrapper, "GVMPipeline", FakePipe)

    wrapper.GVMProcessor(model_base=str(model_dir), device="cpu", seed=123)

    assert captured["FakeDiffusersUNet"]["kwargs"]["low_cpu_mem_usage"] is False
