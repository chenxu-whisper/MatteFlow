"""MatAnyone2 模块.

实用的人物视频抠图框架，保留精细细节，增强真实世界条件下的鲁棒性。
"""

import importlib
import inspect
import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from ..config import MattingConfig
from ..errors import ModelLoadError
from ..utils.model_paths import model_file

logger = logging.getLogger(__name__)


def load_matanyone2_processor_class():
    """Lazy-load the vendored MatAnyone2 processor."""
    module = importlib.import_module("matteflow.vendor.matanyone2_module.wrapper")
    return module.MatAnyone2Processor


class MatAnyone2Matte:
    """MatAnyone2 人物视频抠图引擎"""

    def __init__(self, config: MattingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 MatAnyone2 模型"""
        try:
            MatAnyone2Processor = load_matanyone2_processor_class()

            model_path = self._get_model_path()
            if not model_path.exists():
                logger.info("MatAnyone2 model not found, triggering download: %s", model_path)
                self._download_model()

            logger.info("Loading MatAnyone2 model from %s on device=%s", model_path, self.device)
            self.model = MatAnyone2Processor(
                device=str(self.device),
                ckpt_path=str(model_path),
                max_internal_size=1080,
            )
            logger.info("Loaded MatAnyone2 on %s", self.device)

        except Exception:
            logger.exception("Failed to load MatAnyone2")
            logger.warning("MatAnyone2 will fall back to traditional processing")
            self.model = None

    def _download_model(self):
        """下载 MatAnyone2 模型"""
        import urllib.request

        model_dir = self._get_model_path().parent
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "matanyone2.pth"

        url = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"

        logger.info("Downloading MatAnyone2 weights from %s", url)

        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info("Saved MatAnyone2 weights to %s", model_path)
        except Exception as e:
            raise RuntimeError(f"MatAnyone2 download failed: {e}") from e

    def _get_model_path(self) -> Path:
        return model_file("matanyone2.pth")

    def generate(
        self,
        frames: List[np.ndarray],
        first_frame_mask: Optional[np.ndarray] = None,
        cancel_check=None,
    ) -> List[np.ndarray]:
        """
        生成 Alpha Matte — 注意：后处理由调用方统一处理

        Args:
            frames: RGB 帧列表
            first_frame_mask: 首帧分割 mask（可选）

        Returns:
            Alpha 列表
        """
        if self.model is None:
            logger.info("Model not available, returning empty alphas for %s frames", len(frames))
            raise ModelLoadError("MatAnyone2 model is not loaded")

        try:
            logger.info("Starting MatAnyone2 sequence inference for %s frames", len(frames))
            return self._run_sequence_inference(frames, first_frame_mask, cancel_check=cancel_check)

        except Exception as exc:
            logger.exception("MatAnyone2 inference failed")
            raise RuntimeError("MatAnyone2 inference failed") from exc

    def _apply_chroma_key_postprocess(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """应用 Chroma Key 后处理参数 — 对齐 EZ-CorridorKey"""
        from .chroma_key_postprocess import apply_chroma_key_postprocess
        return apply_chroma_key_postprocess(alphas, self.config)

    def _run_sequence_inference(
        self,
        frames: List[np.ndarray],
        first_frame_mask: Optional[np.ndarray],
        cancel_check=None,
    ) -> List[np.ndarray]:
        if not frames:
            return []
        if self.model is None:
            raise RuntimeError("MatAnyone2 model is not loaded")

        # 上游 decoder 已统一输出 RGB，这里直接透传，避免重复变换导致颜色偏移
        rgb_frames = frames
        mask = self._resolve_first_frame_mask(frames[0], first_frame_mask)
        frame_names = [f"frame_{index:06d}" for index in range(len(frames))]

        with tempfile.TemporaryDirectory(prefix="matanyone2-") as temp_dir:
            output_dir = Path(temp_dir) / "AlphaHint"
            logger.info(
                "Prepared MatAnyone2 temporary workspace: output_dir=%s frame_count=%s",
                output_dir,
                len(frame_names),
            )
            process_kwargs = {
                "input_frames": rgb_frames,
                "mask_frame": mask,
                "output_dir": str(output_dir),
                "frame_names": frame_names,
                "clip_name": "matteflow",
            }
            if "cancel_check" in inspect.signature(self.model.process_frames).parameters:
                process_kwargs["cancel_check"] = cancel_check
            self.model.process_frames(**process_kwargs)

            alphas = []
            for frame_name in frame_names:
                alpha_path = output_dir / f"{frame_name}.png"
                alpha = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
                if alpha is None:
                    raise RuntimeError(f"Failed to read MatAnyone2 output matte: {alpha_path}")
                if alpha.ndim == 3:
                    alpha = alpha[..., 0]
                alphas.append(alpha.astype(np.float32) / 255.0)

            logger.info("Completed MatAnyone2 inference with %s matte frames", len(alphas))
            return alphas

    def _resolve_first_frame_mask(
        self, first_frame: np.ndarray, first_frame_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        if first_frame_mask is None:
            return np.full(first_frame.shape[:2], 255, dtype=np.uint8)

        mask = np.asarray(first_frame_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        return mask

    def generate_sequence(self, frames: List[np.ndarray], cancel_check=None) -> List[np.ndarray]:
        """序列处理"""
        return self.generate(frames, cancel_check=cancel_check)
