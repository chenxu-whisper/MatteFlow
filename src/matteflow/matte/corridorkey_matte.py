"""CorridorKey AI 抠图模块 - 物理分离前景/背景

基于 Corridor Digital 的 CorridorKey 算法
通过物理光场分离，保留毛发、运动模糊、半透明
"""

import numpy as np
import cv2
import torch
from typing import List, Optional, Tuple
from pathlib import Path

from ..config import MattingConfig


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
            import sys
            sys.path.insert(0, "E:/ByteDance/Projects/Code/EZ-CorridorKey")
            
            # 使用 EZ-CorridorKey 的 inference_engine
            from CorridorKeyModule.inference_engine import CorridorKeyEngine
            
            # 检查模型路径
            ez_ck_path = Path("E:/ByteDance/Projects/Code/EZ-CorridorKey/CorridorKeyModule/checkpoints/CorridorKey_v1.0.pth")
            
            if not ez_ck_path.exists():
                print("[CorridorKey] Model not found, downloading...")
                self._download_model()
                ez_ck_path = Path(__file__).parent / "corridorkey" / "model.pth"
            
            print(f"[CorridorKey] Loading from {ez_ck_path}...")
            self.model = CorridorKeyEngine(
                checkpoint_path=str(ez_ck_path),
                device=str(self.device),
                img_size=2048,
                use_refiner=True,
                optimization_mode='auto'
            )
            print(f"[CorridorKey] Loaded on {self.device}")
            
        except Exception as e:
            print(f"[CorridorKey] Failed to load: {e}")
            print("[CorridorKey] Fallback to traditional")
            self.model = None
    
    def _download_model(self):
        """下载 CorridorKey 模型 (383MB)"""
        import urllib.request
        
        model_dir = Path(__file__).parent / "corridorkey"
        model_dir.mkdir(exist_ok=True)
        
        # CorridorKey 模型下载链接
        # 尝试多个镜像
        urls = [
            "https://github.com/nikopueringer/CorridorKey/releases/download/v1.0.0/corridorkey_model.pth",
            "https://huggingface.co/nikopueringer/CorridorKey/resolve/main/model.pth",
        ]
        model_path = model_dir / "model.pth"
        
        for url in urls:
            try:
                print(f"[CorridorKey] Downloading from {url}...")
                urllib.request.urlretrieve(url, model_path)
                print(f"[CorridorKey] Model saved to {model_path}")
                return
            except Exception as e:
                print(f"[CorridorKey] Failed to download from {url}: {e}")
        
        raise RuntimeError("All download URLs failed")
    
    def generate(self, frame: np.ndarray, background: Optional[np.ndarray] = None) -> np.ndarray:
        """单帧抠图 — 对齐 EZ-CorridorKey 参数
        
        Args:
            frame: 输入帧 (H, W, 3), uint8
            background: 可选背景帧，用于物理分离
        
        Returns:
            alpha: Alpha 通道 (H, W), float32 [0, 1]
        """
        if self.model is None:
            from .green_screen_matte import GreenScreenMatte
            return GreenScreenMatte(self.config).generate(frame)
        
        # 直接使用 EZ-CorridorKey 的 process_frame 方法
        try:
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
            
            # 提取 alpha
            if isinstance(result, dict):
                alpha = result.get('alpha', result.get('matte', None))
            elif isinstance(result, tuple):
                alpha = result[0]
            else:
                alpha = result
            
            # 确保是 numpy 数组
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.cpu().numpy()
            
            # 确保单通道
            if alpha.ndim == 3:
                alpha = alpha[..., 0] if alpha.shape[2] > 1 else alpha[..., 0]
            
            # 归一化
            if alpha.max() > 1.0:
                alpha = alpha / 255.0
            alpha = np.clip(alpha, 0, 1).astype(np.float32)
            
            # 应用后处理参数
            alpha = self._apply_postprocess(alpha)
            
            return alpha
            
        except Exception as e:
            print(f"[CorridorKey] process_frame failed: {e}")
            # 回退到传统算法
            from .green_screen_matte import GreenScreenMatte
            return GreenScreenMatte(self.config).generate(frame)
    
    def _apply_postprocess(self, alpha: np.ndarray) -> np.ndarray:
        """应用 EZ-CorridorKey 风格后处理"""
        
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
        alphas = []
        total = len(frames)
        
        for i, frame in enumerate(frames):
            alpha = self.generate(frame, background)
            alphas.append(alpha)
            
            if progress_callback and i % max(1, total // 20) == 0:
                progress_callback(i, total)
        
        return alphas
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """预处理 - CorridorKey 需要 4 通道输入 (RGBA)"""
        h, w = frame.shape[:2]
        
        # 缩放到模型输入尺寸
        new_w, new_h = 512, 288
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        # 归一化到 [0, 1]
        if frame_resized.dtype == np.uint8:
            frame_f = frame_resized.astype(np.float32) / 255.0
        else:
            frame_f = frame_resized.astype(np.float32)
        
        # 添加 alpha 通道（CorridorKey 需要 4 通道）
        if frame_f.shape[2] == 3:
            alpha = np.ones((new_h, new_w, 1), dtype=np.float32)
            frame_f = np.concatenate([frame_f, alpha], axis=2)
        
        # 转换格式 (B, C, H, W)
        tensor = torch.from_numpy(frame_f).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor, (h, w)
    
    def _physical_separation(self, fg_tensor: torch.Tensor, bg_tensor: torch.Tensor) -> torch.Tensor:
        """物理分离前景/背景
        
        核心算法：通过光场物理模型分离前景和背景
        保留毛发、运动模糊、半透明效果
        """
        # 简单的物理分离实现（基于光场差异）
        # 实际 CorridorKey 有更复杂的模型
        
        diff = torch.abs(fg_tensor - bg_tensor)
        
        # 计算 alpha：差异大的区域 = 前景
        alpha = torch.mean(diff, dim=1, keepdim=True)
        
        # 增强对比度
        alpha = torch.pow(alpha, 0.5)
        
        # 归一化
        alpha = torch.clamp(alpha, 0, 1)
        
        return alpha
    
    def _postprocess(self, pha: torch.Tensor, orig_size: Tuple[int, int]) -> np.ndarray:
        """后处理"""
        h, w = orig_size
        
        # 提取 alpha
        if pha.dim() == 4:
            alpha = pha[0, 0]
        elif pha.dim() == 3:
            alpha = pha[0]
        else:
            alpha = pha
        
        # 缩放回原始尺寸
        alpha = alpha.cpu().numpy()
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 归一化
        alpha = np.clip(alpha, 0, 1).astype(np.float32)
        
        return alpha
