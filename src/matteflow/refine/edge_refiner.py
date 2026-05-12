"""边缘细化模块"""

import numpy as np
import cv2

from ..config import MattingConfig


class EdgeRefiner:
    """边缘细化器（毛发/羽毛/发丝）"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
    
    def refine(self, frames, alphas):
        """
        细化边缘 alpha
        
        Args:
            frames: RGB 帧列表
            alphas: Alpha 列表
        
        Returns:
            细化后的 alpha 列表
        """
        refined = []
        for frame, alpha in zip(frames, alphas):
            refined_alpha = self._refine_single(frame, alpha)
            refined.append(refined_alpha)
        return refined
    
    def _refine_single(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """单帧边缘细化"""
        # 1. 提取边缘区域（trimap: 确定前景/背景/未知）
        trimap = self._generate_trimap(alpha)
        
        # 2. 仅在未知区域进行细化
        unknown_mask = (trimap == 128)
        
        if not np.any(unknown_mask):
            return alpha
        
        # 3. 局部导向滤波细化
        refined = self._local_guided_filter(frame, alpha, unknown_mask)
        
        return refined
    
    def _generate_trimap(self, alpha: np.ndarray) -> np.ndarray:
        """生成 trimap"""
        # 硬阈值
        fg_mask = alpha > 0.9
        bg_mask = alpha < 0.1
        
        # 未知区域
        trimap = np.full(alpha.shape, 128, dtype=np.uint8)
        trimap[fg_mask] = 255
        trimap[bg_mask] = 0
        
        return trimap
    
    def _local_guided_filter(self, guide: np.ndarray, src: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """局部导向滤波"""
        # 简化版：对边缘区域进行小半径高斯平滑
        result = src.copy()
        
        # 仅在 mask 区域应用平滑
        src_u8 = (src * 255).astype(np.uint8)
        smoothed = cv2.GaussianBlur(src_u8, (5, 5), 0).astype(np.float32) / 255.0
        
        # 保留非边缘区域的原值
        result = np.where(mask, smoothed, result)
        
        return np.clip(result, 0, 1)
