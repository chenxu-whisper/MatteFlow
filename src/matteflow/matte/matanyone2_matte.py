"""MatAnyone2 模块

实用的人物视频抠图框架，保留精细细节
避免分割式边界，增强真实世界条件下的鲁棒性

模型信息:
- 大小: ~135 MB
- 来源: https://github.com/pq-yang/MatAnyone2
- 论文: CVPR 2026
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional
from pathlib import Path

from ..config import MattingConfig


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
            import sys
            sys.path.insert(0, "E:/ByteDance/Projects/Code/EZ-CorridorKey")
            
            # 使用 EZ-CorridorKey 的 wrapper
            from modules.MatAnyone2Module.wrapper import MatAnyone2Processor
            
            # 检查模型路径
            ez_ma2_path = Path("E:/ByteDance/Projects/Code/EZ-CorridorKey/modules/MatAnyone2Module/checkpoints/matanyone2.pth")
            
            if not ez_ma2_path.exists():
                print("[MatAnyone2] Model not found, downloading...")
                self._download_model()
                ez_ma2_path = Path.home() / ".cache" / "matteflow" / "models" / "matanyone2.pth"
            
            print(f"[MatAnyone2] Loading from {ez_ma2_path}...")
            self.model = MatAnyone2Processor(
                device=str(self.device),
                ckpt_path=str(ez_ma2_path),
                max_internal_size=1080
            )
            print(f"[MatAnyone2] Loaded on {self.device}")
            
        except Exception as e:
            print(f"[MatAnyone2] Failed to load: {e}")
            print("[MatAnyone2] Fallback to traditional")
            self.model = None
    
    def _download_model(self):
        """下载 MatAnyone2 模型"""
        import urllib.request
        
        model_dir = Path.home() / ".cache" / "matteflow" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "matanyone2.pth"
        
        # HuggingFace 模型链接
        url = "https://huggingface.co/pq-yang/MatAnyone2/resolve/main/matanyone2.pth"
        
        print(f"[MatAnyone2] Downloading from {url}")
        print("[MatAnyone2] This may take a few minutes...")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"[MatAnyone2] Saved to {model_path}")
        except Exception as e:
            print(f"[MatAnyone2] Download failed: {e}")
            print("[MatAnyone2] Please download manually from:")
            print("  https://huggingface.co/pq-yang/MatAnyone2")
    
    def generate(self, frames: List[np.ndarray], first_frame_mask: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        生成 Alpha Matte — 注意：后处理由调用方统一处理
        
        Args:
            frames: RGB 帧列表
            first_frame_mask: 首帧分割 mask（可选）
        
        Returns:
            Alpha 列表
        """
        if self.model is None:
            print("[MatAnyone2] Model not available")
            return [np.ones(f.shape[:2], dtype=np.float32) * 0.5 for f in frames]
        
        try:
            # MatAnyone2 处理逻辑
            alphas = []
            prev_alpha = None
            
            for i, frame in enumerate(frames):
                if i == 0 and first_frame_mask is not None:
                    alpha = self._process_first_frame(frame, first_frame_mask)
                else:
                    alpha = self._process_frame(frame, prev_alpha)
                
                alphas.append(alpha)
                prev_alpha = alpha
            
            # 注意：后处理由 hybrid_matte 统一处理，避免重复
            return alphas
            
        except Exception as e:
            print(f"[MatAnyone2] Inference failed: {e}")
            return [np.ones(f.shape[:2], dtype=np.float32) * 0.5 for f in frames]
    
    def _apply_chroma_key_postprocess(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """应用 Chroma Key 后处理参数 — 对齐 EZ-CorridorKey"""
        from .chroma_key_postprocess import apply_chroma_key_postprocess
        return apply_chroma_key_postprocess(alphas, self.config)
    
    def _process_first_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """处理首帧"""
        # 预处理
        frame_tensor = self._preprocess(frame)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        # 推理
        with torch.no_grad():
            alpha = self.model(frame_tensor, mask_tensor)
        
        return self._postprocess(alpha)
    
    def _process_frame(self, frame: np.ndarray, prev_alpha: Optional[np.ndarray] = None) -> np.ndarray:
        """处理后续帧"""
        frame_tensor = self._preprocess(frame)
        
        # 使用前一帧 alpha 作为记忆
        if prev_alpha is not None:
            prev_tensor = torch.from_numpy(prev_alpha).unsqueeze(0).unsqueeze(0).float().to(self.device)
        else:
            prev_tensor = None
        
        with torch.no_grad():
            alpha = self.model(frame_tensor, prev_tensor)
        
        return self._postprocess(alpha)
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """预处理帧"""
        # 归一化
        frame = frame.astype(np.float32) / 255.0
        
        # 转换为 tensor
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device)
        
        return frame_tensor
    
    def _postprocess(self, alpha: torch.Tensor) -> np.ndarray:
        """后处理 alpha"""
        alpha = alpha.squeeze().cpu().numpy()
        alpha = np.clip(alpha, 0, 1)
        return alpha
    
    def generate_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """序列处理"""
        return self.generate(frames)
