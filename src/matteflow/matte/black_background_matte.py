"""黑底特效 Matte 生成模块 - 优化版"""

import numpy as np
import cv2

from ..config import MattingConfig


class BlackBackgroundMatte:
    """黑底特效抠图引擎 - 优化版：保留暗部半透明物体"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
    
    def generate(self, frame: np.ndarray) -> np.ndarray:
        """
        生成黑底 Alpha Matte - 修复版：保留粒子/辉光/弱光细节
        
        Args:
            frame: RGB 图像 (H, W, 3), uint8
        
        Returns:
            alpha: Alpha 通道 (H, W), float32 [0, 1]
        """
        frame_f = frame.astype(np.float32)
        r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
        
        # 1. 基础亮度
        gray = np.mean(frame_f, axis=2)
        max_channel = np.max(frame_f, axis=2)
        min_channel = np.min(frame_f, axis=2)
        color_range = max_channel - min_channel
        
        # 2. 饱和度（有颜色的物体即使暗也要有 alpha）
        saturation = np.where(max_channel > 0, color_range / (max_channel + 1e-6), 0)
        
        # 3. 亮度 alpha - 非线性映射，保护暗部
        # 阈值: 低于此值开始变透明
        threshold = self.config.black_threshold * 255
        
        # 非线性映射: 暗部保留更多 alpha
        # 0 -> 0, threshold -> 0.3 (保留弱粒子), 60 -> 1.0
        brightness_alpha = np.zeros_like(gray)
        dark_region = (gray > threshold * 0.3) & (gray <= threshold)
        mid_region = (gray > threshold) & (gray <= 60)
        bright_region = gray > 60
        
        # 极暗区：粒子/辉光保留（半透明）
        brightness_alpha[dark_region] = 0.2 + (gray[dark_region] - threshold * 0.3) / (threshold * 0.7) * 0.3
        # 中等亮度：平滑过渡
        brightness_alpha[mid_region] = 0.5 + (gray[mid_region] - threshold) / (60 - threshold) * 0.5
        # 亮区：不透明
        brightness_alpha[bright_region] = 1.0
        
        # 4. 颜色 alpha - 只要有颜色就保留（即使很暗）
        # color_range > 3 说明不是纯灰
        color_alpha = np.clip((color_range - 2) / 15.0, 0, 1)
        
        # 5. 饱和度 alpha - 彩色物体保护
        sat_alpha = np.clip((saturation - 0.03) / 0.2, 0, 1)
        
        # 6. 综合 alpha - 粒子/辉光保护模式
        # 亮度权重降低，颜色和饱和度权重提高
        alpha = np.maximum(brightness_alpha * 0.4, 
                          np.maximum(color_alpha * 0.6, sat_alpha * 0.6))
        
        # 7. 辉光增强 - 暗部有颜色的区域保留更多
        glow_mask = (gray < 30) & (color_range > 3)
        alpha = np.where(glow_mask, np.maximum(alpha, 0.35), alpha)
        
        # 8. 粒子保护 - 单点亮点（粒子）
        # 局部亮度高于周围 → 可能是粒子
        gray_u8 = gray.astype(np.uint8)
        local_max = cv2.dilate(gray_u8, np.ones((5, 5), np.uint8))
        particle_mask = (gray > 5) & (gray.astype(np.uint8) == local_max) & (gray < 40)
        alpha = np.where(particle_mask, np.maximum(alpha, 0.4), alpha)
        
        # 9. 纯黑强制透明（但阈值放宽）
        black_mask = gray < threshold * 0.3
        alpha = np.where(black_mask, 0.0, alpha)
        
        # 10. 平滑（小半径避免 artifact）
        alpha_u8 = (alpha * 255).astype(np.uint8)
        alpha_smooth = cv2.GaussianBlur(alpha_u8, (3, 3), 0).astype(np.float32) / 255.0
        
        return np.clip(alpha_smooth, 0, 1).astype(np.float32)
