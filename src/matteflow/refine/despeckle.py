"""去噪点模块 - 去除 alpha 遮罩中的噪点"""

import numpy as np
import cv2

from ..config import MattingConfig


class Despeckle:
    """去除 alpha 遮罩中的噪点"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
    
    def process(self, alphas):
        """
        去除噪点
        
        Args:
            alphas: Alpha 列表
        
        Returns:
            去噪后的 alpha 列表
        """
        if not self.config.despeckle_enable:
            return alphas
        
        cleaned = []
        for alpha in alphas:
            cleaned_alpha = self._despeckle_single(alpha)
            cleaned.append(cleaned_alpha)
        return cleaned
    
    def _despeckle_single(self, alpha: np.ndarray) -> np.ndarray:
        """单帧去噪点"""
        # 转换为 uint8
        alpha_u8 = (alpha * 255).astype(np.uint8)
        
        # 使用中值滤波去除噪点
        radius = self.config.despeckle_radius
        ksize = 2 * radius + 1
        cleaned = cv2.medianBlur(alpha_u8, ksize)
        
        # 阈值处理去除微小噪点
        threshold = int(self.config.despeckle_threshold * 255)
        if threshold > 0:
            cleaned = np.where(cleaned <= threshold, 0, cleaned).astype(np.uint8)
        
        return cleaned.astype(np.float32) / 255.0
