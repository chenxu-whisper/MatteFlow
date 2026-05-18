import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.gpu_monitor import GPUInfo, GPUMonitor, get_gpu_info


class FakeNVML:
    NVML_TEMPERATURE_GPU = 0

    def __init__(self):
        self.initialized = False

    def nvmlInit(self):
        self.initialized = True

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle-{index}"

    def nvmlDeviceGetName(self, handle):
        return b"RTX Test"

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=8 * 1024**3, used=2 * 1024**3, free=6 * 1024**3)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=42)

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 55

    def nvmlShutdown(self):
        self.initialized = False


def test_gpu_monitor_uses_nvml_when_available():
    fake_nvml = FakeNVML()
    monitor = GPUMonitor(import_module=lambda name: fake_nvml if name == "pynvml" else None)

    info = monitor.get_info()

    assert info == GPUInfo(
        available=True,
        backend="nvml",
        device_count=1,
        name="RTX Test",
        memory_total_mb=8192,
        memory_used_mb=2048,
        memory_free_mb=6144,
        utilization_percent=42,
        temperature_c=55,
    )


def test_gpu_monitor_falls_back_to_torch_cuda_when_nvml_missing():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(index):
            return "Torch GPU"

        @staticmethod
        def mem_get_info(index):
            return 3 * 1024**3, 4 * 1024**3

    fake_torch = SimpleNamespace(cuda=FakeCuda())

    def import_module(name):
        if name == "pynvml":
            raise ModuleNotFoundError(name)
        if name == "torch":
            return fake_torch
        raise AssertionError(name)

    info = GPUMonitor(import_module=import_module).get_info()

    assert info.available is True
    assert info.backend == "torch"
    assert info.device_count == 1
    assert info.name == "Torch GPU"
    assert info.memory_total_mb == 4096
    assert info.memory_used_mb == 1024
    assert info.memory_free_mb == 3072


def test_gpu_monitor_logs_nvml_failure_and_torch_fallback_success(caplog):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def get_device_name(index):
            return "Fallback GPU"

        @staticmethod
        def mem_get_info(index):
            return 1 * 1024**3, 2 * 1024**3

    fake_torch = SimpleNamespace(cuda=FakeCuda())

    def import_module(name):
        if name == "pynvml":
            raise RuntimeError("NVML driver not loaded")
        if name == "torch":
            return fake_torch
        raise AssertionError(name)

    with caplog.at_level(logging.INFO, logger="matteflow.gpu_monitor"):
        info = GPUMonitor(import_module=import_module).get_info()

    assert info.backend == "torch"
    assert "NVML GPU monitor unavailable: NVML driver not loaded" in caplog.text
    assert "torch.cuda GPU monitor succeeded: device_count=1 name=Fallback GPU" in caplog.text


def test_gpu_monitor_reports_unavailable_when_no_backend_is_usable():
    def import_module(name):
        raise ModuleNotFoundError(name)

    info = GPUMonitor(import_module=import_module).get_info()

    assert info.available is False
    assert info.backend == "none"
    assert info.device_count == 0
    assert "No GPU monitor backend available" in info.error


def test_gpu_monitor_reports_unavailable_when_backend_raises():
    def import_module(name):
        if name == "pynvml":
            raise RuntimeError("driver unavailable")
        raise ModuleNotFoundError(name)

    info = GPUMonitor(import_module=import_module).get_info()

    assert info.available is False
    assert info.backend == "none"
    assert "driver unavailable" in info.error


def test_gpu_monitor_logs_final_failure_reason(caplog):
    def import_module(name):
        raise ModuleNotFoundError(name)

    with caplog.at_level(logging.INFO, logger="matteflow.gpu_monitor"):
        info = GPUMonitor(import_module=import_module).get_info()

    assert info.available is False
    assert "No GPU monitor backend available" in caplog.text
    assert "pynvml" in caplog.text
    assert "torch" in caplog.text


def test_get_gpu_info_uses_default_monitor():
    info = get_gpu_info(monitor=GPUMonitor(import_module=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name))))

    assert isinstance(info, GPUInfo)
    assert info.available is False


def test_importing_gpu_monitor_does_not_import_torch_or_pynvml(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "pynvml", raising=False)

    import matteflow.gpu_monitor as gpu_monitor

    importlib.reload(gpu_monitor)

    assert "torch" not in sys.modules
    assert "pynvml" not in sys.modules
