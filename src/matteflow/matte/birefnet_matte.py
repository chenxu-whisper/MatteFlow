"""BiRefNet AI 抠图模块 - 2024 SOTA 单图抠图"""

import numpy as np
import cv2
import torch
from typing import List
from pathlib import Path

from ..config import MattingConfig


class BiRefNetMatte:
    """基于 BiRefNet 的高精度抠图引擎"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载 BiRefNet 模型（从 HuggingFace）"""
        try:
            from transformers import AutoModelForImageSegmentation
            
            print("[BiRefNet] Loading model from HuggingFace...")
            
            # 尝试加载最新版本，指定 revision 避免兼容性问题
            try:
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet",
                    trust_remote_code=True,
                    revision="main",
                    torch_dtype=torch.float32
                )
            except Exception as e1:
                print(f"[BiRefNet] Main revision failed: {e1}")
                print("[BiRefNet] Trying legacy revision...")
                # 尝试旧版本
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet",
                    trust_remote_code=True,
                    revision="v1.0",
                    torch_dtype=torch.float32
                )
            
            self.model.eval().to(self.device)
            print(f"[BiRefNet] Loaded on {self.device}")
            
        except Exception as e:
            print(f"[BiRefNet] Failed to load: {e}")
            print("[BiRefNet] Fallback to RVM or traditional")
            self.model = None
    
    def generate(self, frame: np.ndarray) -> np.ndarray:
        """单帧抠图"""
        if self.model is None:
            # 回退到传统算法
            from .green_screen_matte import GreenScreenMatte
            return GreenScreenMatte(self.config).generate(frame)
        
        # 预处理
        tensor, orig_size = self._preprocess(frame)
        
        # 推理
        with torch.no_grad():
            pred = self.model(tensor)
        
        # 后处理
        alpha = self._postprocess(pred, orig_size)
        
        return alpha
    
    def generate_sequence(self, frames: List[np.ndarray], progress_callback=None) -> List[np.ndarray]:
        """序列抠图（逐帧处理，适合批量）"""
        alphas = []
        total = len(frames)
        
        for i, frame in enumerate(frames):
            alpha = self.generate(frame)
            alphas.append(alpha)
            
            if progress_callback and i % max(1, total // 20) == 0:
                progress_callback(i, total)
        
        # 注意：后处理由 hybrid_matte 统一处理，避免重复
        return alphas
    
    def _apply_chroma_key_postprocess(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """应用 Chroma Key 后处理参数 — 对齐 EZ-CorridorKey"""
        from .chroma_key_postprocess import apply_chroma_key_postprocess
        return apply_chroma_key_postprocess(alphas, self.config)
    
    def _preprocess(self, frame: np.ndarray):
        """预处理 - BiRefNet 需要 1024x1024 输入"""
        from torchvision import transforms
        
        h, w = frame.shape[:2]
        
        # 转换为 PIL Image 格式
        if frame.dtype == np.uint8:
            frame_rgb = frame
        else:
            frame_rgb = (frame * 255).astype(np.uint8)
        
        # 标准化变换
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        tensor = transform(frame_rgb).unsqueeze(0).to(self.device)
        
        return tensor, (h, w)
    
    def _postprocess(self, pred, orig_size) -> np.ndarray:
        """后处理 - 恢复原始尺寸"""
        h, w = orig_size
        
        # 提取 alpha
        if isinstance(pred, dict):
            alpha = pred.get('out', pred.get('alpha', None))
        else:
            alpha = pred
        
        if alpha is None:
            # 如果输出结构不对，返回全 1
            return np.ones((h, w), dtype=np.float32)
        
        # 取第一个 batch 和第一个通道
        if alpha.ndim == 4:
            alpha = alpha[0, 0]
        elif alpha.ndim == 3:
            alpha = alpha[0]
        
        # 缩放回原始尺寸
        alpha = alpha.cpu().numpy()
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 归一化到 [0, 1]
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        
        return alpha
