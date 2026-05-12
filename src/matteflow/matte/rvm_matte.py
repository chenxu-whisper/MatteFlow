"""RVM (Robust Video Matting) AI 抠图模块"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import List, Optional
from pathlib import Path

from ..config import MattingConfig


class RVMMatte:
    """基于 RVM 的 AI 视频抠图引擎"""
    
    def __init__(self, config: MattingConfig, model_type: str = "mobilenetv3"):
        self.config = config
        self.model_type = model_type
        # 尝试 GPU，ECC 错误时回退 CPU
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                test = torch.zeros(1).cuda()
                self.device = torch.device("cuda")
                print("[RVM] Using CUDA")
            except Exception as e:
                print(f"[RVM] CUDA error ({e}), falling back to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self.model = None
        self._rec = [None] * 4  # RVM 循环状态
        
        self._load_model()
    
    def _load_model(self):
        """加载 RVM 模型"""
        try:
            rvm_path = Path(__file__).parent / "rvm"
            if str(rvm_path.parent) not in sys.path:
                sys.path.insert(0, str(rvm_path.parent))
            
            import importlib.util
            spec = importlib.util.spec_from_file_location('rvm', str(rvm_path / '__init__.py'))
            rvm_pkg = importlib.util.module_from_spec(spec)
            sys.modules['rvm'] = rvm_pkg
            
            from rvm.model import MattingNetwork
            
            print("[RVM] Creating model...")
            self.model = MattingNetwork(self.model_type).eval().to(self.device)
            
            cache_dir = Path.home() / ".cache" / "matteflow" / "models"
            model_path = cache_dir / f"rvm_{self.model_type}.pth"
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            print(f"[RVM] Loading weights from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            print(f"[RVM] Loaded {self.model_type} on {self.device}")
            print(f"[RVM] Params: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"[RVM] Failed to load: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def generate(self, frame: np.ndarray) -> np.ndarray:
        """单帧抠图"""
        if self.model is None:
            from .green_screen_matte import GreenScreenMatte
            return GreenScreenMatte(self.config).generate(frame)
        
        # 单帧模式：重置循环状态，避免时序污染
        self.reset()
        
        tensor, pad_info = self._preprocess(frame)
        
        with torch.no_grad():
            fgr, pha, *self._rec = self.model(tensor, *self._rec)
        
        return self._postprocess(pha, pad_info)
    
    def generate_sequence(self, frames: List[np.ndarray], progress_callback=None) -> List[np.ndarray]:
        """序列抠图（利用时序一致性）"""
        if self.model is None:
            from .green_screen_matte import GreenScreenMatte
            matte = GreenScreenMatte(self.config)
            return [matte.generate(f) for f in frames]
        
        alphas = []
        self.reset()
        
        for i, frame in enumerate(frames):
            tensor, pad_info = self._preprocess(frame)
            
            with torch.no_grad():
                fgr, pha, *self._rec = self.model(tensor, *self._rec)
            
            alpha = self._postprocess(pha, pad_info)
            alphas.append(alpha)
            
            if progress_callback and i % max(1, len(frames) // 20) == 0:
                progress_callback(i, len(frames))
        
        # 注意：后处理由 hybrid_matte 统一处理，避免重复
        return alphas
    
    def _apply_chroma_key_postprocess(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """应用 Chroma Key 后处理参数 — 对齐 EZ-CorridorKey"""
        from .chroma_key_postprocess import apply_chroma_key_postprocess
        return apply_chroma_key_postprocess(alphas, self.config)
    
    def reset(self):
        """重置循环状态"""
        self._rec = [None] * 4
    
    def _preprocess(self, frame: np.ndarray):
        """
        预处理 - 修复：RVM 只需要 [0,1] 归一化，不需要 ImageNet 标准化
        """
        h, w = frame.shape[:2]
        
        # 1. 确保尺寸是 32 的倍数（RVM 要求）
        new_h = ((h - 1) // 32 + 1) * 32
        new_w = ((w - 1) // 32 + 1) * 32
        
        pad_bottom = new_h - h
        pad_right = new_w - w
        
        if pad_bottom > 0 or pad_right > 0:
            frame_padded = np.pad(frame, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='constant')
        else:
            frame_padded = frame
        
        # 2. RVM 模型只需要 [0, 1] 归一化
        # 不要做 ImageNet 标准化！RVM 是在原始 RGB [0,1] 上训练的
        tensor = torch.from_numpy(frame_padded.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        pad_info = (h, w, pad_bottom, pad_right)
        return tensor, pad_info
    
    def _postprocess(self, pha: torch.Tensor, pad_info) -> np.ndarray:
        """后处理"""
        h, w, pad_bottom, pad_right = pad_info
        
        alpha = pha[0, 0].cpu().numpy()
        
        # 去掉 padding
        if pad_bottom > 0:
            alpha = alpha[:-pad_bottom, :]
        if pad_right > 0:
            alpha = alpha[:, :-pad_right]
        
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        
        # 如果启用了 AI 增强，对 alpha 做后处理
        if getattr(self.config, 'ai_enhance', False):
            alpha = self._enhance_alpha(alpha)
        
        return alpha
    
    def _enhance_alpha(self, alpha: np.ndarray) -> np.ndarray:
        """
        Alpha 增强 - 修复：保守增强，避免破坏原有质量
        
        策略：
        1. 轻微 Gamma 校正 - 只提升极暗部
        2. 对比度拉伸 - 增强前景/背景区分
        3. 可选边缘锐化
        """
        # 1. 保守 Gamma 校正（默认 gamma=0.8，轻微提升暗部）
        gamma = getattr(self.config, 'ai_enhance_gamma', 0.8)
        if gamma != 1.0:
            alpha = np.power(alpha, gamma)
        
        # 2. 对比度拉伸（只拉伸两端，保护中间过渡）
        threshold = getattr(self.config, 'ai_enhance_threshold', 0.1)
        gain = getattr(self.config, 'ai_enhance_gain', 1.2)
        
        # 极低值区域（背景泄漏）提升
        bg_leak = alpha < threshold
        alpha = np.where(bg_leak, alpha * gain, alpha)
        
        # 高置信度前景增强
        fg_strong = alpha > 0.85
        alpha = np.where(fg_strong, 0.85 + (alpha - 0.85) * 1.1, alpha)
        
        # 3. 边缘锐化（可选，默认关闭）
        sharpen = getattr(self.config, 'ai_enhance_sharpen', 0.0)
        if sharpen > 0:
            blurred = cv2.GaussianBlur(alpha, (0, 0), 2)
            detail = alpha - blurred
            alpha = alpha + detail * sharpen
        
        return np.clip(alpha, 0, 1).astype(np.float32)
