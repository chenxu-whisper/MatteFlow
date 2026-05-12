"""RMBG-2 AI 抠图模块 - 商业级开源抠图"""

import numpy as np
import cv2
import torch
from typing import List
from pathlib import Path

from ..config import MattingConfig


class RMBGMatte:
    """基于 RMBG-2 的高精度抠图引擎"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载 RMBG-2 模型"""
        try:
            from transformers import AutoModelForImageSegmentation
            
            print("[RMBG] Loading model from HuggingFace...")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "briaai/RMBG-2-Studio",
                trust_remote_code=True,
                revision="main"
            )
            self.model.eval().to(self.device)
            print(f"[RMBG] Loaded on {self.device}")
            
        except Exception as e:
            print(f"[RMBG] Failed to load: {e}")
            print("[RMBG] Fallback to traditional")
            self.model = None
    
    def generate(self, frame: np.ndarray) -> np.ndarray:
        """单帧抠图"""
        if self.model is None:
            from .green_screen_matte import GreenScreenMatte
            return GreenScreenMatte(self.config).generate(frame)
        
        tensor, orig_size = self._preprocess(frame)
        
        with torch.no_grad():
            pred = self.model(tensor)
        
        alpha = self._postprocess(pred, orig_size)
        return alpha
    
    def generate_sequence(self, frames: List[np.ndarray], progress_callback=None) -> List[np.ndarray]:
        """序列抠图"""
        alphas = []
        total = len(frames)
        
        for i, frame in enumerate(frames):
            alpha = self.generate(frame)
            alphas.append(alpha)
            
            if progress_callback and i % max(1, total // 20) == 0:
                progress_callback(i, total)
        
        return alphas
    
    def _preprocess(self, frame: np.ndarray):
        """预处理"""
        from torchvision import transforms
        
        h, w = frame.shape[:2]
        
        if frame.dtype == np.uint8:
            frame_rgb = frame
        else:
            frame_rgb = (frame * 255).astype(np.uint8)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        tensor = transform(frame_rgb).unsqueeze(0).to(self.device)
        return tensor, (h, w)
    
    def _postprocess(self, pred, orig_size) -> np.ndarray:
        """后处理"""
        h, w = orig_size
        
        if isinstance(pred, dict):
            alpha = pred.get('out', pred.get('alpha', None))
        else:
            alpha = pred
        
        if alpha is None:
            return np.ones((h, w), dtype=np.float32)
        
        if alpha.ndim == 4:
            alpha = alpha[0, 0]
        elif alpha.ndim == 3:
            alpha = alpha[0]
        
        alpha = alpha.cpu().numpy()
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        
        return alpha
