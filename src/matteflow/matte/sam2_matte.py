"""SAM2 (Segment Anything Model 2) 模块

Meta 的 SAM2 视频分割模型，支持视频跟踪和分割
用于生成首帧 mask，配合 MatAnyone2 使用

模型信息:
- 大小: ~324 MB (Base+)
- 来源: https://github.com/facebookresearch/segment-anything-2
"""

import numpy as np
import torch
from typing import List, Optional, Tuple
from pathlib import Path

from ..config import MattingConfig


class SAM2Matte:
    """SAM2 视频分割引擎"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """加载 SAM2 模型"""
        try:
            print("[SAM2] Loading model...")
            
            # SAM2 使用 HuggingFace 或本地缓存
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_id = "facebook/sam2-hiera-base-plus"
            
            # 检查本地缓存
            local_path = cache_dir / "models--facebook--sam2.1-hiera-base-plus"
            if local_path.exists():
                print(f"[SAM2] Found local model at {local_path}")
            
            # 尝试使用 transformers 加载
            try:
                from transformers import AutoModel
                self.model = AutoModel.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    trust_remote_code=True
                )
                self.model.to(self.device)
                self.model.eval()
                print(f"[SAM2] Loaded from HuggingFace on {self.device}")
                return
            except Exception as e:
                print(f"[SAM2] transformers loading failed: {e}")
            
            # 回退到 torch.hub
            try:
                self.model = torch.hub.load(
                    "facebookresearch/segment-anything-2",
                    "sam2_hiera_base_plus",
                    trust_repo=True
                )
                self.model.to(self.device)
                self.model.eval()
                print(f"[SAM2] Loaded from torch.hub on {self.device}")
            except Exception as e:
                print(f"[SAM2] torch.hub loading failed: {e}")
                
        except Exception as e:
            print(f"[SAM2] Failed to load: {e}")
            print("[SAM2] Please install: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            self.model = None
            self.predictor = None
    
    def _download_model(self):
        """下载 SAM2 模型"""
        import urllib.request
        
        model_dir = Path.home() / ".cache" / "matteflow" / "models" / "sam2"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "sam2_hiera_base_plus.pt"
        
        # HuggingFace 模型链接
        url = "https://huggingface.co/facebook/sam2-hiera-base-plus/resolve/main/sam2_hiera_base_plus.pt"
        
        print(f"[SAM2] Downloading from {url}")
        print("[SAM2] This may take a few minutes...")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"[SAM2] Saved to {model_path}")
        except Exception as e:
            print(f"[SAM2] Download failed: {e}")
            print("[SAM2] Please download manually from:")
            print("  https://huggingface.co/facebook/sam2-hiera-base-plus")
    
    def generate_mask(self, frame: np.ndarray, points: Optional[List[Tuple[int, int]]] = None,
                     labels: Optional[List[int]] = None) -> np.ndarray:
        """
        生成单帧分割 mask
        
        Args:
            frame: RGB 帧
            points: 点击点坐标列表 [(x, y), ...]
            labels: 点标签 (1=前景, 0=背景)
        
        Returns:
            分割 mask
        """
        if self.predictor is None and self.model is None:
            print("[SAM2] Model not available")
            return np.ones(frame.shape[:2], dtype=np.uint8)
        
        try:
            if self.predictor is not None:
                # 使用视频 predictor
                return self._predict_with_predictor(frame, points, labels)
            else:
                # 使用普通模型
                return self._predict_with_model(frame, points, labels)
                
        except Exception as e:
            print(f"[SAM2] Inference failed: {e}")
            return np.ones(frame.shape[:2], dtype=np.uint8)
    
    def _predict_with_predictor(self, frame: np.ndarray, points, labels) -> np.ndarray:
        """使用视频 predictor"""
        # 初始化状态
        self.predictor.init_state(frame)
        
        # 添加点提示
        if points is not None and labels is not None:
            point_coords = np.array(points)
            point_labels = np.array(labels)
            
            mask, scores, _ = self.predictor.add_new_points(
                frame_idx=0,
                obj_id=1,
                points=point_coords,
                labels=point_labels
            )
        else:
            # 自动分割
            mask = self.predictor.predict(frame)
        
        return mask.astype(np.uint8)
    
    def _predict_with_model(self, frame: np.ndarray, points, labels) -> np.ndarray:
        """使用普通模型"""
        # 预处理
        frame_tensor = self._preprocess(frame)
        
        with torch.no_grad():
            if points is not None and labels is not None:
                # 使用点提示
                point_coords = torch.tensor(points).unsqueeze(0).to(self.device)
                point_labels = torch.tensor(labels).unsqueeze(0).to(self.device)
                
                masks, scores, _ = self.model.predict_torch(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )
                mask = masks[0, 0].cpu().numpy()
            else:
                # 自动分割
                mask = self.model.predict(frame_tensor)
        
        return (mask > 0.5).astype(np.uint8)
    
    def track_video(self, frames: List[np.ndarray], init_mask: np.ndarray) -> List[np.ndarray]:
        """
        视频跟踪分割
        
        Args:
            frames: RGB 帧列表
            init_mask: 首帧初始化 mask
        
        Returns:
            每帧的 mask 列表
        """
        if self.predictor is None:
            print("[SAM2] Video predictor not available")
            return [init_mask] * len(frames)
        
        try:
            masks = []
            
            # 初始化视频状态
            for i, frame in enumerate(frames):
                if i == 0:
                    # 首帧使用 init_mask
                    self.predictor.init_state(frame)
                    mask = init_mask
                else:
                    # 跟踪后续帧
                    mask = self.predictor.track(frame)
                
                masks.append(mask)
            
            return masks
            
        except Exception as e:
            print(f"[SAM2] Tracking failed: {e}")
            return [init_mask] * len(frames)
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """预处理帧"""
        frame = frame.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device)
        return frame_tensor
    
    def generate_sequence(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """序列处理 - 返回 masks"""
        # 首帧自动生成 mask
        first_mask = self.generate_mask(frames[0])
        
        # 跟踪视频
        return self.track_video(frames, first_mask)
