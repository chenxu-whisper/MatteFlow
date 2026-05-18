import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.matte.gvm_matte import GVMMatte
from matteflow.vendor.gvm_core import wrapper


class _FakeGVMProcessor:
    def process_sequence(
        self,
        input_path: str,
        output_dir: str,
        direct_output_dir: str,
        mode: str,
        write_video: bool,
        base_res: int | None = None,
        scale_cap: int | None = None,
    ) -> None:
        del input_path, output_dir, mode, write_video, base_res, scale_cap
        alpha = np.full((1024, 1024), 255, dtype=np.uint8)
        out_path = Path(direct_output_dir) / "00000.png"
        assert cv2.imwrite(str(out_path), alpha)


def test_gvm_sequence_resizes_alpha_back_to_input_frame_shape() -> None:
    matte = GVMMatte.__new__(GVMMatte)
    matte.model = _FakeGVMProcessor()

    frames = [np.zeros((960, 960, 3), dtype=np.uint8)]

    alphas = matte._run_sequence_inference(frames)

    assert len(alphas) == 1
    assert alphas[0].shape == (960, 960)


def test_gvm_sequence_passes_configured_internal_size_to_processor() -> None:
    class _CapturingGVMProcessor(_FakeGVMProcessor):
        def process_sequence(self, *args, **kwargs) -> None:
            self.kwargs = kwargs
            super().process_sequence(*args, **kwargs)

    matte = GVMMatte.__new__(GVMMatte)
    matte.config = type("Config", (), {"gvm_max_internal_size": 512})()
    matte.model = _CapturingGVMProcessor()

    frames = [np.zeros((960, 960, 3), dtype=np.uint8)]

    matte._run_sequence_inference(frames)

    assert matte.model.kwargs["base_res"] == 512
    assert matte.model.kwargs["scale_cap"] == 512


def test_gvm_wrapper_preserves_soft_alpha_values(monkeypatch, tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image_path = input_dir / "00000.png"
    assert cv2.imwrite(str(image_path), np.zeros((4, 4, 3), dtype=np.uint8))

    written = {}

    class _FakeReader:
        frame_rate = 24.0

        def __init__(self, path, transform):
            del path, transform

        def __getitem__(self, index):
            del index
            return torch.zeros((3, 4, 4), dtype=torch.float32)

    class _FakeWriter:
        extension = "png"

        def __init__(self, path, extension="png"):
            del path, extension
            self.counter = 0

        def write(self, alpha, filenames=None):
            del filenames
            written["alpha"] = alpha.detach().cpu().clone()

        def close(self):
            pass

    soft_alpha = torch.tensor(
        [[[[0.08, 0.12], [0.50, 0.96]]]],
        dtype=torch.float32,
    )

    processor = wrapper.GVMProcessor.__new__(wrapper.GVMProcessor)
    processor.device = torch.device("cpu")
    processor._dtype = torch.float32
    processor.pipe = lambda *args, **kwargs: SimpleNamespace(
        image=torch.zeros_like(soft_alpha).repeat(1, 3, 1, 1),
        alpha=soft_alpha.clone(),
    )

    monkeypatch.setattr(wrapper, "ImageSequenceReader", _FakeReader)
    monkeypatch.setattr(
        wrapper,
        "DataLoader",
        lambda reader, batch_size, collate_fn=None: [
            {
                "rgb_names": ["00000.png"],
                "rgb_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
            }
        ],
    )
    monkeypatch.setattr(wrapper, "ImageSequenceWriter", _FakeWriter)
    monkeypatch.setattr(wrapper, "Compose", lambda ops: ops)
    monkeypatch.setattr(wrapper, "ToTensor", lambda: None)
    monkeypatch.setattr(wrapper, "Resize", lambda *args, **kwargs: None)
    monkeypatch.setattr(wrapper, "impad_multi", lambda batch: (batch, (0, 0, 0, 0)))

    processor.process_sequence(
        input_path=str(input_dir),
        output_dir=str(tmp_path / "out"),
        direct_output_dir=str(tmp_path / "alpha"),
        num_frames_per_batch=1,
        write_video=False,
    )

    assert "alpha" in written
    actual = written["alpha"][0, 0, :2, :2].numpy()
    expected = F.interpolate(soft_alpha, (4, 4), mode="bilinear")[0, 0, :2, :2].numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_gvm_preserve_internal_swirl_content_repairs_low_alpha_holes() -> None:
    matte = GVMMatte.__new__(GVMMatte)
    frame = np.full((9, 9, 3), [20, 200, 20], dtype=np.uint8)
    frame[2:7, 2:7] = [170, 135, 240]
    alpha = np.zeros((9, 9), dtype=np.float32)
    alpha[2:7, 2:7] = 0.52
    alpha[4, 4] = 0.08

    repaired = matte._preserve_internal_swirl_content(frame, alpha)

    assert repaired[4, 4] > alpha[4, 4]
    assert repaired[4, 4] <= 0.55


def test_gvm_preserve_internal_swirl_content_keeps_pure_green_background_flat() -> None:
    matte = GVMMatte.__new__(GVMMatte)
    frame = np.full((9, 9, 3), [20, 200, 20], dtype=np.uint8)
    alpha = np.zeros((9, 9), dtype=np.float32)

    repaired = matte._preserve_internal_swirl_content(frame, alpha)

    assert np.allclose(repaired, alpha)


def test_gvm_preserve_internal_swirl_content_can_recover_zero_alpha_hole_with_support() -> None:
    matte = GVMMatte.__new__(GVMMatte)
    frame = np.full((9, 9, 3), [20, 200, 20], dtype=np.uint8)
    frame[1:8, 1:8] = [185, 155, 240]
    alpha = np.zeros((9, 9), dtype=np.float32)
    alpha[1:8, 1:8] = 0.52
    alpha[4, 4] = 0.0

    repaired = matte._preserve_internal_swirl_content(frame, alpha)

    assert repaired[4, 4] > 0.2
    assert repaired[4, 4] <= 0.55
