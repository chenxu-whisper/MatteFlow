"""CorridorKey AI 抠图模块 - 物理分离前景/背景

基于 Corridor Digital 的 CorridorKey 算法
通过物理光场分离，保留毛发、运动模糊、半透明
"""

import importlib
import logging
from typing import List, Optional

import cv2
import numpy as np
import torch

from ..config import MattingConfig
from ..errors import ModelLoadError
from ..utils.model_checker import validate_direct_model_file
from ..utils.model_downloads import download_file_atomically
from ..utils.model_paths import model_file, models_root

logger = logging.getLogger(__name__)


def load_corridorkey_engine_class():
    """Lazy-load the vendored CorridorKey engine to avoid eager heavy imports."""
    module = importlib.import_module("matteflow.vendor.corridorkey_module.inference_engine")
    return module.CorridorKeyEngine


class CorridorKeyMatte:
    """基于 CorridorKey 的物理分离抠图引擎"""

    def __init__(self, config: MattingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 CorridorKey 模型"""
        try:
            CorridorKeyEngine = load_corridorkey_engine_class()

            checkpoint_path = model_file("corridorkey.pth")

            if not checkpoint_path.exists():
                logger.info("CorridorKey model not found, triggering download: %s", checkpoint_path)
                self._download_model()

            logger.info("Loading CorridorKey model from %s on device=%s", checkpoint_path, self.device)
            self.model = CorridorKeyEngine(
                checkpoint_path=str(checkpoint_path),
                device=str(self.device),
                img_size=2048,
                use_refiner=True,
                optimization_mode="auto",
            )
            logger.info("Loaded CorridorKey on %s", self.device)

        except Exception:
            logger.exception("Failed to load CorridorKey")
            logger.warning("CorridorKey will fall back to traditional processing")
            self.model = None

    def _download_model(self):
        """下载 CorridorKey 模型 (383MB)"""
        model_dir = models_root()
        model_dir.mkdir(parents=True, exist_ok=True)

        # 当前使用的官方权重来源与本项目加载路径。
        url = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
        model_path = model_dir / "corridorkey.pth"

        logger.info("Downloading CorridorKey weights from %s", url)
        try:
            download_file_atomically(
                url,
                model_path,
                validate=lambda path: validate_direct_model_file(path, "corridorkey.pth"),
            )
            logger.info("Saved CorridorKey weights to %s", model_path)
        except Exception as e:
            raise RuntimeError(f"CorridorKey download failed: {e}") from e

    def generate(self, frame: np.ndarray, background: Optional[np.ndarray] = None) -> np.ndarray:
        """单帧抠图

        Args:
            frame: 输入帧 (H, W, 3), uint8
            background: 可选背景帧，用于物理分离

        Returns:
            alpha: Alpha 通道 (H, W), float32 [0, 1]
        """
        if self.model is None:
            logger.info("Model unavailable")
            raise ModelLoadError("CorridorKey model is not loaded")

        try:
            logger.info("Running CorridorKey process_frame on resolution=%sx%s", frame.shape[1], frame.shape[0])
            # 上游 decoder 已统一输出 RGB，这里不再重复做 BGR->RGB 变换
            frame_rgb = frame

            # 创建默认 mask（全白，让模型自己检测）
            h, w = frame.shape[:2]
            mask = np.ones((h, w), dtype=np.float32)

            # 调用 process_frame
            result = self.model.process_frame(
                image=frame_rgb,
                mask_linear=mask,
                refiner_scale=getattr(self.config, 'refiner_scale', 1.0),
                input_is_linear=getattr(self.config, 'input_is_linear', False),
                screen_color=getattr(self.config, 'screen_color', 'auto')
            )

            alpha = self._extract_alpha_array(result)

            # 应用后处理参数
            alpha = self._apply_postprocess(alpha)

            return alpha

        except Exception as exc:
            logger.exception("CorridorKey process_frame failed")
            raise RuntimeError("CorridorKey inference failed") from exc

    def _extract_alpha_array(self, result) -> np.ndarray:
        """Extract alpha output and normalize it to a 2D float32 numpy array."""
        alpha = None
        if isinstance(result, dict):
            alpha = result.get("alpha", result.get("matte"))
        elif isinstance(result, tuple):
            alpha = result[0] if result else None
        else:
            alpha = result

        if alpha is None:
            raise RuntimeError("CorridorKey returned no alpha output")

        if isinstance(alpha, torch.Tensor):
            alpha_array = alpha.detach().cpu().numpy()
        else:
            alpha_array = np.asarray(alpha, dtype=np.float32)

        if alpha_array.ndim == 3:
            alpha_array = alpha_array[..., 0]

        alpha_array = alpha_array.astype(np.float32, copy=False)
        if alpha_array.max(initial=0.0) > 1.0:
            alpha_array = alpha_array / 255.0

        return np.clip(alpha_array, 0.0, 1.0).astype(np.float32, copy=False)

    def _apply_postprocess(self, alpha: np.ndarray) -> np.ndarray:
        """风格后处理"""

        # 1. Shrink/Grow
        if self.config.shrink_grow != 0:
            alpha = self._apply_shrink_grow(alpha)

        # 2. Edge Blur
        if self.config.edge_blur > 0:
            alpha = self._apply_edge_blur(alpha)

        # 3. Clip Black/White
        clip_black = self.config.clip_black
        clip_white = self.config.clip_white
        if clip_black > 0 or clip_white < 1.0:
            cw = max(clip_white, clip_black + 0.001)
            alpha = np.clip((alpha - clip_black) / (cw - clip_black), 0.0, 1.0)

        return alpha

    def _apply_shrink_grow(self, alpha: np.ndarray) -> np.ndarray:
        """应用 Shrink/Grow"""
        shrink_grow = self.config.shrink_grow
        abs_px = abs(shrink_grow)
        if abs_px == 0:
            return alpha

        alpha_u8 = (alpha * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (abs_px * 2 + 1, abs_px * 2 + 1)
        )
        if shrink_grow < 0:
            alpha_u8 = cv2.erode(alpha_u8, kernel)
        else:
            alpha_u8 = cv2.dilate(alpha_u8, kernel)
        return alpha_u8.astype(np.float32) / 255.0

    def _apply_edge_blur(self, alpha: np.ndarray) -> np.ndarray:
        """应用 Edge Blur"""
        edge_blur = self.config.edge_blur
        if edge_blur <= 0:
            return alpha

        alpha_u8 = (alpha * 255).astype(np.uint8)
        k = edge_blur * 2 + 1
        alpha_u8 = cv2.GaussianBlur(alpha_u8, (k, k), 0)
        return alpha_u8.astype(np.float32) / 255.0

    def generate_sequence(self, frames: List[np.ndarray], background: Optional[np.ndarray] = None, progress_callback=None) -> List[np.ndarray]:
        """序列抠图"""
        logger.info("Starting CorridorKey sequence inference for %s frames", len(frames))
        alphas = []
        total = len(frames)

        for i, frame in enumerate(frames):
            alpha = self.generate(frame, background)
            alphas.append(alpha)

            if progress_callback and i % max(1, total // 20) == 0:
                progress_callback(i, total)

        logger.info("Completed CorridorKey sequence inference with %s matte frames", len(alphas))
        return alphas
