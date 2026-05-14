import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATTEFLOW_ROOT = PROJECT_ROOT / "src" / "matteflow"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return

    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load_matanyone2_matte_module():
    _ensure_package("matteflow", MATTEFLOW_ROOT)
    _ensure_package("matteflow.matte", MATTEFLOW_ROOT / "matte")

    module_name = "matteflow.matte.matanyone2_matte"
    module_path = MATTEFLOW_ROOT / "matte" / "matanyone2_matte.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_matanyone2_source_uses_vendored_processor():
    module_path = MATTEFLOW_ROOT / "matte" / "matanyone2_matte.py"
    contents = module_path.read_text(encoding="utf-8")

    assert "matteflow.vendor.matanyone2_module.wrapper" in contents
    assert "from modules.MatAnyone2Module.wrapper import MatAnyone2Processor" not in contents
    assert "sys.path.insert" not in contents


def test_matanyone2_matte_loads_processor_from_vendored_package(monkeypatch):
    module = _load_matanyone2_matte_module()

    captured = {}

    class _FakeProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    fake_wrapper = types.ModuleType("matteflow.vendor.matanyone2_module.wrapper")
    fake_wrapper.MatAnyone2Processor = _FakeProcessor
    sys.modules["matteflow.vendor.matanyone2_module.wrapper"] = fake_wrapper

    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_wrapper)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)

    matte = module.MatAnyone2Matte(SimpleNamespace())

    assert isinstance(matte.model, _FakeProcessor)
    assert Path(captured["kwargs"]["ckpt_path"]).name == "matanyone2.pth"
    assert captured["kwargs"]["device"] == "cuda"


def test_matanyone2_generate_sequence_uses_processor_output_dir(monkeypatch, tmp_path):
    module = _load_matanyone2_matte_module()

    calls = {}

    class _FakeProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def process_frames(self, input_frames, mask_frame, output_dir, frame_names, **kwargs):
            calls["input_frames"] = input_frames
            calls["mask_frame"] = mask_frame
            calls["frame_names"] = frame_names
            calls["kwargs"] = kwargs
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, _frame_name in enumerate(frame_names):
                alpha = np.full((3, 4), 64 + idx, dtype=np.uint8)
                cv2.imwrite(str(out_dir / f"{frame_names[idx]}.png"), alpha)
            return len(frame_names)

    fake_wrapper = types.ModuleType("matteflow.vendor.matanyone2_module.wrapper")
    fake_wrapper.MatAnyone2Processor = _FakeProcessor
    sys.modules["matteflow.vendor.matanyone2_module.wrapper"] = fake_wrapper

    class _FakeTemporaryDirectory:
        def __init__(self, path: Path):
            self.path = path

        def __enter__(self):
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_wrapper)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix=None: _FakeTemporaryDirectory(tmp_path))

    matte = module.MatAnyone2Matte(SimpleNamespace())
    frames = [np.zeros((3, 4, 3), dtype=np.uint8), np.ones((3, 4, 3), dtype=np.uint8) * 255]

    alphas = matte.generate_sequence(frames)

    assert len(alphas) == 2
    assert alphas[0].shape == (3, 4)
    assert np.isclose(alphas[0][0, 0], 64 / 255.0)
    assert np.isclose(alphas[1][0, 0], 65 / 255.0)
    assert calls["mask_frame"].shape == (3, 4)
    assert calls["frame_names"] == ["frame_000000", "frame_000001"]
