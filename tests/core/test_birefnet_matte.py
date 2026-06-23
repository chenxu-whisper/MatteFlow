import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig  # noqa: E402
from matteflow.matte.birefnet_matte import BiRefNetMatte  # noqa: E402


def test_birefnet_postprocess_accepts_list_outputs():
    matte = BiRefNetMatte.__new__(BiRefNetMatte)
    pred = [torch.full((1, 1, 2, 3), 0.75, dtype=torch.float32)]

    alpha = matte._postprocess(pred, (4, 6))

    assert alpha.shape == (4, 6)
    assert alpha.dtype == np.float32
    assert float(alpha.mean()) == 0.75


def test_birefnet_load_suppresses_known_kornia_torchscript_deprecation(monkeypatch):
    captured_kwargs = {}

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

    class FakeAutoModelForImageSegmentation:
        @classmethod
        def from_pretrained(cls, *_args, **kwargs):
            captured_kwargs.update(kwargs)
            warnings.warn(
                "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.",
                DeprecationWarning,
            )
            return FakeModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageSegmentation = FakeAutoModelForImageSegmentation
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matte = BiRefNetMatte(MattingConfig())

    assert matte.model is not None
    assert "dtype" in captured_kwargs
    assert "torch_dtype" not in captured_kwargs
    assert not [
        warning
        for warning in caught
        if "torch.jit.script" in str(warning.message)
    ]
