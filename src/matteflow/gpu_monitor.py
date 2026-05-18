"""Lightweight GPU/VRAM monitoring helpers.

This module intentionally avoids importing GPU libraries at module import time.
Callers can import it from CLI, tests, or GUI setup code without creating CUDA
contexts or requiring optional monitoring dependencies to be installed.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPUInfo:
    """Snapshot of GPU availability and memory state."""

    available: bool
    backend: str
    device_count: int = 0
    name: Optional[str] = None
    memory_total_mb: Optional[int] = None
    memory_used_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    utilization_percent: Optional[int] = None
    temperature_c: Optional[int] = None
    error: Optional[str] = None


class GPUMonitor:
    """Query GPU status using NVML first, then torch.cuda as a fallback."""

    def __init__(self, import_module: Callable[[str], object] = importlib.import_module) -> None:
        self._import_module = import_module

    def get_info(self) -> GPUInfo:
        """Return a best-effort GPU info snapshot without raising."""
        errors: list[str] = []
        logger.debug("Starting GPU monitor query")

        nvml_info = self._try_nvml(errors)
        if nvml_info is not None:
            return nvml_info

        torch_info = self._try_torch(errors)
        if torch_info is not None:
            return torch_info

        message = "; ".join(errors) if errors else "No GPU monitor backend available"
        logger.error("No GPU monitor backend available: %s", message)
        return GPUInfo(
            available=False,
            backend="none",
            device_count=0,
            error=f"No GPU monitor backend available: {message}",
        )

    def _try_nvml(self, errors: list[str]) -> Optional[GPUInfo]:
        logger.debug("Trying NVML GPU monitor backend")
        try:
            nvml = self._import_module("pynvml")
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            errors.append(f"pynvml: {message}")
            logger.warning("NVML GPU monitor unavailable: %s", message)
            return None

        try:
            nvml.nvmlInit()
            count = int(nvml.nvmlDeviceGetCount())
            if count <= 0:
                logger.info("NVML GPU monitor found no devices")
                return GPUInfo(
                    available=False,
                    backend="nvml",
                    device_count=0,
                    error="NVML found no GPU devices",
                )

            handle = nvml.nvmlDeviceGetHandleByIndex(0)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = self._safe_nvml_utilization(nvml, handle)
            temperature = self._safe_nvml_temperature(nvml, handle)
            name = self._decode_name(nvml.nvmlDeviceGetName(handle))
            total_mb = self._bytes_to_mb(memory.total)
            used_mb = self._bytes_to_mb(memory.used)
            free_mb = self._bytes_to_mb(memory.free)
            logger.info(
                "NVML GPU monitor succeeded: device_count=%s name=%s memory_used_mb=%s memory_total_mb=%s",
                count,
                name,
                used_mb,
                total_mb,
            )
            return GPUInfo(
                available=True,
                backend="nvml",
                device_count=count,
                name=name,
                memory_total_mb=total_mb,
                memory_used_mb=used_mb,
                memory_free_mb=free_mb,
                utilization_percent=utilization,
                temperature_c=temperature,
            )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            errors.append(f"pynvml: {message}")
            logger.warning("NVML GPU monitor failed during query: %s", message)
            return None
        finally:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass

    def _try_torch(self, errors: list[str]) -> Optional[GPUInfo]:
        logger.debug("Trying torch.cuda GPU monitor backend")
        try:
            torch = self._import_module("torch")
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            errors.append(f"torch: {message}")
            logger.warning("torch.cuda GPU monitor unavailable: %s", message)
            return None

        try:
            cuda = torch.cuda
            if not cuda.is_available():
                logger.info("torch.cuda GPU monitor found CUDA unavailable")
                return GPUInfo(
                    available=False,
                    backend="torch",
                    device_count=0,
                    error="torch.cuda is not available",
                )

            count = int(cuda.device_count())
            free_bytes, total_bytes = cuda.mem_get_info(0)
            total_mb = self._bytes_to_mb(total_bytes)
            free_mb = self._bytes_to_mb(free_bytes)
            name = str(cuda.get_device_name(0))
            used_mb = max(total_mb - free_mb, 0)
            logger.info(
                "torch.cuda GPU monitor succeeded: device_count=%s name=%s memory_used_mb=%s memory_total_mb=%s",
                count,
                name,
                used_mb,
                total_mb,
            )
            return GPUInfo(
                available=True,
                backend="torch",
                device_count=count,
                name=name,
                memory_total_mb=total_mb,
                memory_free_mb=free_mb,
                memory_used_mb=used_mb,
            )
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            errors.append(f"torch: {message}")
            logger.warning("torch.cuda GPU monitor failed during query: %s", message)
            return None

    @staticmethod
    def _safe_nvml_utilization(nvml: object, handle: object) -> Optional[int]:
        try:
            return int(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            return None

    @staticmethod
    def _safe_nvml_temperature(nvml: object, handle: object) -> Optional[int]:
        try:
            sensor = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
            return int(nvml.nvmlDeviceGetTemperature(handle, sensor))
        except Exception:
            return None

    @staticmethod
    def _decode_name(name: object) -> str:
        if isinstance(name, bytes):
            return name.decode("utf-8", errors="replace")
        return str(name)

    @staticmethod
    def _bytes_to_mb(value: int) -> int:
        return int(round(int(value) / 1024 / 1024))


def get_gpu_info(monitor: Optional[GPUMonitor] = None) -> GPUInfo:
    """Convenience function for callers that do not need a monitor instance."""
    return (monitor or GPUMonitor()).get_info()
