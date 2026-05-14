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


def _load_corridorkey_matte_module():
    _ensure_package("matteflow", MATTEFLOW_ROOT)
    _ensure_package("matteflow.matte", MATTEFLOW_ROOT / "matte")

    module_name = "matteflow.matte.corridorkey_matte"
    module_path = MATTEFLOW_ROOT / "matte" / "corridorkey_matte.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_corridorkey_source_uses_vendored_engine():
    module_path = MATTEFLOW_ROOT / "matte" / "corridorkey_matte.py"
    contents = module_path.read_text(encoding="utf-8")

    assert "matteflow.vendor.corridorkey_module.inference_engine" in contents
    assert "from CorridorKeyModule.inference_engine import CorridorKeyEngine" not in contents
    assert "sys.path.insert" not in contents


def test_corridorkey_matte_loads_engine_from_vendored_package(monkeypatch):
    module = _load_corridorkey_matte_module()

    captured = {}

    class _FakeEngine:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    fake_inference_engine = types.ModuleType("matteflow.vendor.corridorkey_module.inference_engine")
    fake_inference_engine.CorridorKeyEngine = _FakeEngine
    sys.modules["matteflow.vendor.corridorkey_module.inference_engine"] = fake_inference_engine

    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_inference_engine)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    matte = module.CorridorKeyMatte(SimpleNamespace())

    assert isinstance(matte.model, _FakeEngine)
    assert Path(captured["kwargs"]["checkpoint_path"]).name == "corridorkey.pth"
    assert captured["kwargs"]["device"] == "cuda"


def test_extract_alpha_array_rejects_missing_output():
    module = _load_corridorkey_matte_module()
    matte = module.CorridorKeyMatte.__new__(module.CorridorKeyMatte)

    try:
        matte._extract_alpha_array({})
    except RuntimeError as exc:
        assert "CorridorKey returned no alpha output" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing CorridorKey alpha output")
