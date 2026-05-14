"""GVM (Generative Video Matting) 模块

基于 Stable Video Diffusion 的生成式视频抠图模型
支持人物和动物的细粒度视频抠图

模型信息:
- 大小: ~6 GB
- 来源: https://github.com/aim-uofa/GVM
- 论文: SIGGRAPH 2025
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

from ..config import MattingConfig
from ..utils.model_paths import models_root, resolve_snapshot_model_dir

logger = logging.getLogger(__name__)


def resolve_gvm_model_base(models_root: Path) -> Optional[Path]:
    """Resolve a usable GVM weights directory from either flat or HF-cache layouts."""
    return resolve_snapshot_model_dir(
        models_root,
        "geyongtao/gvm",
        ("unet", "vae", "scheduler"),
    )


class GVMMatte:
    """Generative Video Matting 引擎"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self._ensure_cuda_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _ensure_cuda_available(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("GVM 仅支持 CUDA GPU。当前 PyTorch 环境未检测到可用 CUDA，请安装 CUDA 版 PyTorch。")
    
    def _load_model(self):
        """加载 GVM 模型"""
        try:
            from ..vendor.gvm_core.wrapper import GVMProcessor

            model_base = resolve_gvm_model_base(models_root())

            if model_base is None:
                logger.warning("Weights not found under models root: %s", models_root())
                raise RuntimeError("GVM weights not found")

            logger.info("Loading GVM model from %s on device=%s", model_base, self.device)
            self.model = GVMProcessor(
                model_base=str(model_base),
                device=str(self.device),
            )
            self._force_cpu_float32_runtime()
            logger.info("Loaded GVM from vendored MatteFlow package on %s", self.device)

        except Exception as e:
            logger.exception("Failed to load GVM runtime")
            logger.warning("GVM requires the vendored runtime and diffusers dependencies")
            self.model = None
    
    def generate(self, frames: List[np.ndarray], hints: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        生成 Alpha Matte — 注意：后处理由调用方统一处理
        
        Args:
            frames: RGB 帧列表
            hints: 可选的 hint mask (首帧)
        
        Returns:
            Alpha 列表
        """
        if self.model is None:
            logger.info("Model not available, returning empty alphas for %s frames", len(frames))
            return [np.ones(f.shape[:2], dtype=np.float32) * 0.5 for f in frames]
        
        try:
            del hints  # Current GVM integration does not consume per-frame hints.
            logger.info("Starting GVM sequence inference for %s frames", len(frames))
            return self._run_sequence_inference(frames)
            
        except Exception as e:
            logger.exception("GVM inference failed")
            return [np.ones(f.shape[:2], dtype=np.float32) * 0.5 for f in frames]
    
    def _apply_chroma_key_postprocess(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """应用 Chroma Key 后处理参数 — 对齐 EZ-CorridorKey"""
        from .chroma_key_postprocess import apply_chroma_key_postprocess
        return apply_chroma_key_postprocess(alphas, self.config)
    
    def _force_cpu_float32_runtime(self) -> None:
        """Diffusers pipelines cannot reliably run in float16 on CPU."""
        if self.model is None or self.device.type != "cpu":
            return

        self.model._dtype = torch.float32
        if getattr(self.model, "pipe", None) is not None:
            self.model.pipe = self.model.pipe.to(self.device, dtype=torch.float32)
        if getattr(self.model, "vae", None) is not None:
            self.model.vae = self.model.vae.to(self.device, dtype=torch.float32)
        if getattr(self.model, "unet", None) is not None:
            self.model.unet = self.model.unet.to(self.device, dtype=torch.float32)

    def _run_sequence_inference(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Run the upstream GVM sequence API and load the generated mattes."""
        if not frames:
            return []

        with tempfile.TemporaryDirectory(prefix="matteflow_gvm_") as temp_dir:
            temp_root = Path(temp_dir)
            input_dir = temp_root / "input_frames"
            alpha_dir = temp_root / "alpha_seq"
            input_dir.mkdir(parents=True, exist_ok=True)
            alpha_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Prepared GVM temporary workspace: input_dir=%s alpha_dir=%s frame_count=%s",
                input_dir,
                alpha_dir,
                len(frames),
            )

            for index, frame in enumerate(frames):
                frame_path = input_dir / f"{index:05d}.png"
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(frame_path), frame_bgr):
                    raise RuntimeError(f"Failed to write GVM input frame: {frame_path}")

            self.model.process_sequence(
                input_path=str(input_dir),
                output_dir=str(temp_root),
                direct_output_dir=str(alpha_dir),
                mode="matte",
                write_video=False,
            )

            alpha_files = sorted(alpha_dir.glob("*.png"))
            if len(alpha_files) != len(frames):
                raise RuntimeError(
                    f"GVM output frame count mismatch: expected {len(frames)}, got {len(alpha_files)}"
                )

            alphas: List[np.ndarray] = []
            for frame, alpha_file in zip(frames, alpha_files):
                alpha = cv2.imread(str(alpha_file), cv2.IMREAD_UNCHANGED)
                if alpha is None:
                    raise RuntimeError(f"Failed to read GVM output matte: {alpha_file}")
                if alpha.ndim == 3:
                    alpha = alpha[..., 0]
                expected_h, expected_w = frame.shape[:2]
                if alpha.shape != (expected_h, expected_w):
                    alpha = cv2.resize(alpha, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                alphas.append(alpha.astype(np.float32) / 255.0)

            logger.info("Completed GVM sequence inference with %s matte frames", len(alphas))
            return alphas
    
    def generate_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """序列处理"""
        return self.generate(frames)
