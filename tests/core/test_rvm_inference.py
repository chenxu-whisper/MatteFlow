import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RVM_ROOT = PROJECT_ROOT / "src" / "matteflow" / "matte" / "rvm"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(RVM_ROOT))


class _FakeRVMModel:
    def eval(self):
        return self

    def to(self, _device):
        return self

    def load_state_dict(self, _state_dict):
        return None


def _load_rvm_inference_module():
    return importlib.import_module("matteflow.matte.rvm.inference")


def test_rvm_converter_uses_torch_compile_without_torchscript(monkeypatch):
    inference = _load_rvm_inference_module()
    model = _FakeRVMModel()
    compiled_model = object()
    compile_calls = []

    monkeypatch.setattr(inference, "MattingNetwork", lambda _variant: model, raising=False)
    monkeypatch.setattr(inference.torch, "load", lambda *_args, **_kwargs: {})

    def _compile(candidate, **kwargs):
        compile_calls.append({"model": candidate, "kwargs": kwargs})
        return compiled_model

    monkeypatch.setattr(inference.torch, "compile", _compile, raising=False)
    monkeypatch.setattr(
        inference.torch.jit,
        "script",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("torch.jit.script should not be used")),
    )
    monkeypatch.setattr(
        inference.torch.jit,
        "freeze",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("torch.jit.freeze should not be used")),
    )

    converter = inference.Converter("mobilenetv3", "checkpoint.pth", "cpu")

    assert converter.model is compiled_model
    assert compile_calls == [{"model": model, "kwargs": {"fullgraph": False}}]


def test_rvm_converter_falls_back_to_eager_model_when_torch_compile_fails(monkeypatch):
    inference = _load_rvm_inference_module()
    model = _FakeRVMModel()

    monkeypatch.setattr(inference, "MattingNetwork", lambda _variant: model, raising=False)
    monkeypatch.setattr(inference.torch, "load", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        inference.torch,
        "compile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("compile unavailable")),
        raising=False,
    )
    monkeypatch.setattr(
        inference.torch.jit,
        "script",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("torch.jit.script should not be used")),
    )
    monkeypatch.setattr(
        inference.torch.jit,
        "freeze",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("torch.jit.freeze should not be used")),
    )

    converter = inference.Converter("mobilenetv3", "checkpoint.pth", "cpu")

    assert converter.model is model
