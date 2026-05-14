import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATTEFLOW_ROOT = PROJECT_ROOT / "src" / "matteflow"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return

    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load_gvm_matte_module():
    _ensure_package("matteflow", MATTEFLOW_ROOT)
    _ensure_package("matteflow.matte", MATTEFLOW_ROOT / "matte")

    module_name = "matteflow.matte.gvm_matte"
    module_path = MATTEFLOW_ROOT / "matte" / "gvm_matte.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_gvm_model_base_prefers_hf_snapshot_layout(tmp_path):
    module = _load_gvm_matte_module()
    snapshot_dir = tmp_path / "models--geyongtao--gvm" / "snapshots" / "abc123"
    (snapshot_dir / "unet").mkdir(parents=True)
    (snapshot_dir / "vae").mkdir(parents=True)
    (snapshot_dir / "scheduler").mkdir(parents=True)

    resolved = module.resolve_gvm_model_base(tmp_path)

    assert resolved == snapshot_dir


def test_gvm_source_uses_vendored_processor():
    module_path = MATTEFLOW_ROOT / "matte" / "gvm_matte.py"
    contents = module_path.read_text(encoding="utf-8")

    assert "from ..vendor.gvm_core.wrapper import GVMProcessor" in contents
    assert "from gvm_core.wrapper import GVMProcessor" not in contents
    assert "sys.path.insert" not in contents


def test_gvm_matte_loads_processor_from_vendored_package(monkeypatch):
    module = _load_gvm_matte_module()

    _ensure_package("matteflow.vendor", MATTEFLOW_ROOT / "vendor")
    _ensure_package("matteflow.vendor.gvm_core", MATTEFLOW_ROOT / "vendor" / "gvm_core")

    captured = {}

    class _FakeProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    fake_wrapper = types.ModuleType("matteflow.vendor.gvm_core.wrapper")
    fake_wrapper.GVMProcessor = _FakeProcessor
    sys.modules["matteflow.vendor.gvm_core.wrapper"] = fake_wrapper

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "resolve_gvm_model_base", lambda _root: Path("E:/fake-model"))

    matte = module.GVMMatte(SimpleNamespace())

    assert isinstance(matte.model, _FakeProcessor)
    assert Path(captured["kwargs"]["model_base"]) == Path("E:/fake-model")
    assert captured["kwargs"]["device"] == "cuda"
