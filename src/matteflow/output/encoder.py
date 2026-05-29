"""Image output helpers for MatteFlow."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class RGBAEncoder:
    """Encode RGB/RGBA and grayscale images to disk."""

    def encode_image(self, image: np.ndarray, output_path: str | Path) -> None:
        """Write an RGB or RGBA image.

        Float inputs are interpreted as 0..1 and converted to uint8. OpenCV
        expects BGR/BGRA channel order, while the pipeline works in RGB/RGBA.
        """
        arr = self._prepare_for_write(image, output_path)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Expected RGB/RGBA image, got shape {arr.shape}")

        if arr.shape[2] == 4:
            encoded = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        else:
            encoded = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self._write(encoded, output_path)

    def encode_grayscale(self, image: np.ndarray, output_path: str | Path) -> None:
        """Write a single-channel grayscale image."""
        arr = self._prepare_for_write(image, output_path)
        if arr.ndim != 2:
            raise ValueError(f"Expected grayscale image, got shape {arr.shape}")
        self._write(arr, output_path)

    @staticmethod
    def _prepare_for_write(image: np.ndarray, output_path: str | Path) -> np.ndarray:
        path = Path(output_path)
        if path.suffix.lower() == ".exr":
            return image.astype(np.float32, copy=False)
        return RGBAEncoder._to_uint8(image)

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        return (np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    @staticmethod
    def _write(image: np.ndarray, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to write image: {path}")
