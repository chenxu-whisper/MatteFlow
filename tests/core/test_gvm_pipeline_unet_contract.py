import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.vendor.gvm_core.gvm.pipelines.pipeline_gvm import GVMPipeline


class _Scheduler:
    config = type("Config", (), {"num_train_timesteps": 1000})()

    def set_timesteps(self, num_inference_steps, device):
        self.timesteps = torch.arange(num_inference_steps, device=device)


class _UNetOutput:
    def __init__(self, sample):
        self.sample = sample


class _StockDiffusersUNet:
    def __init__(self):
        self.added_time_ids = None

    def __call__(self, sample, timestep, *, encoder_hidden_states, added_time_ids):
        self.added_time_ids = added_time_ids
        return _UNetOutput(torch.zeros_like(sample[:, :, :4]))


class _VendoredUNet:
    def __init__(self):
        self.position_ids = object()
        self.class_labels = object()

    def __call__(
        self,
        sample,
        timestep,
        *,
        encoder_hidden_states,
        position_ids=None,
        class_labels=None,
    ):
        self.position_ids = position_ids
        self.class_labels = class_labels
        return _UNetOutput(torch.zeros_like(sample[:, :, :4]))


class _WrappedStockUNet:
    def __init__(self):
        self.model = _StockDiffusersUNet()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def _pipeline_with(unet):
    pipe = object.__new__(GVMPipeline)
    pipe.unet = unet
    pipe.scheduler = _Scheduler()
    pipe.encode = lambda rgb: rgb.new_zeros((rgb.shape[0], rgb.shape[1], 4, 2, 2))
    return pipe


def test_single_infer_passes_added_time_ids_to_stock_diffusers_unet():
    unet = _StockDiffusersUNet()
    pipe = _pipeline_with(unet)

    pipe.single_infer(
        torch.zeros((2, 3, 3, 4, 4)),
        num_inference_steps=1,
        position_ids=torch.ones((2, 3), dtype=torch.long),
        class_labels=torch.ones((2,), dtype=torch.long),
        noise_type="zeros",
    )

    assert unet.added_time_ids.shape == (2, 3)
    assert unet.added_time_ids.device.type == "cpu"
    assert unet.added_time_ids.dtype == torch.float32


def test_single_infer_detects_added_time_ids_on_wrapped_stock_unet():
    unet = _WrappedStockUNet()
    pipe = _pipeline_with(unet)

    pipe.single_infer(
        torch.zeros((2, 3, 3, 4, 4)),
        num_inference_steps=1,
        position_ids=torch.ones((2, 3), dtype=torch.long),
        class_labels=torch.ones((2,), dtype=torch.long),
        noise_type="zeros",
    )

    assert unet.model.added_time_ids.shape == (2, 3)


def test_single_infer_falls_back_to_vendored_unet_contract():
    unet = _VendoredUNet()
    pipe = _pipeline_with(unet)
    position_ids = torch.ones((2, 3), dtype=torch.long)
    class_labels = torch.ones((2,), dtype=torch.long)

    pipe.single_infer(
        torch.zeros((2, 3, 3, 4, 4)),
        num_inference_steps=1,
        position_ids=position_ids,
        class_labels=class_labels,
        noise_type="zeros",
    )

    assert unet.position_ids is position_ids
    assert unet.class_labels is class_labels
