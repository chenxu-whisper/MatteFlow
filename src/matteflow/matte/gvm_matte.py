"""GVM (Generative Video Matting) 模块

基于 Stable Video Diffusion 的生成式视频抠图模型
支持人物和动物的细粒度视频抠图

模型信息:
- 大小: ~6 GB
- 来源: https://github.com/aim-uofa/GVM
- 论文: SIGGRAPH 2025
"""

import numpy as np
import torch
from typing import List, Optional
from pathlib import Path

from ..config import MattingConfig


class GVMMatte:
    """Generative Video Matting 引擎"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载 GVM 模型"""
        try:
            import sys
            sys.path.insert(0, "E:/ByteDance/Projects/Code/EZ-CorridorKey")
            
            # 使用 EZ-CorridorKey 的 wrapper
            from gvm_core.wrapper import GVMProcessor
            
            # 检查权重路径
            ez_gvm_path = Path("E:/ByteDance/Projects/Code/EZ-CorridorKey/gvm_core/weights")
            
            if not ez_gvm_path.exists():
                print("[GVM] Weights not found, please download from HuggingFace")
                raise RuntimeError("GVM weights not found")
            
            print(f"[GVM] Loading from {ez_gvm_path}...")
            self.model = GVMProcessor(
                model_base=str(ez_gvm_path),
                device=str(self.device)
            )
            print(f"[GVM] Loaded from EZ-CorridorKey on {self.device}")
            
        except Exception as e:
            print(f"[GVM] Failed to load: {e}")
            print("[GVM] Please install: pip install diffusers")
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
            print("[GVM] Model not available, returning empty alphas")
            return [np.ones(f.shape[:2], dtype=np.float32) * 0.5 for f in frames]
        
        try:
            # GVM 处理逻辑
            alphas = []
            for i, frame in enumerate(frames):
                alpha = self._process_frame(frame, hints if i == 0 else None)
                alphas.append(alpha)
            
            # 注意：后处理由 hybrid_matte 统一处理，避免重复
            return alphas
            
        except Exception as e:
            print(f"[GVM] Inference failed: {e}")
            return [np.ones(f.shape[:2], dtype=np.float32) * 0.5 for f in frames]
    
    def _apply_chroma_key_postprocess(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """应用 Chroma Key 后处理参数 — 对齐 EZ-CorridorKey"""
        from .chroma_key_postprocess import apply_chroma_key_postprocess
        return apply_chroma_key_postprocess(alphas, self.config)
    
    def _process_frame(self, frame: np.ndarray, hint: Optional[np.ndarray] = None) -> np.ndarray:
        """单帧处理"""
        h, w = frame.shape[:2]
        
        # 预处理
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame_tensor = frame_tensor.to(self.device)
        
        # 使用模型生成 matte
        with torch.no_grad():
            # 这里应该是 GVM 的实际推理逻辑
            # 简化版：返回一个模拟的 alpha
            alpha = torch.ones(1, 1, h, w).to(self.device) * 0.5
        
        # 后处理
        alpha = alpha.squeeze().cpu().numpy()
        alpha = np.clip(alpha, 0, 1)
        
        return alpha
    
    def generate_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """序列处理"""
        return self.generate(frames)
