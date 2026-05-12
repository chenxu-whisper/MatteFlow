"""颜色去污染模块 - 优化版"""

import numpy as np
import cv2

from ..config import MattingConfig, BackgroundMode


class ColorDecontaminate:
    """颜色去污染/边缘修复 - 优化版：更强去绿边"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
    
    def process(self, frames, alphas, bg_mode):
        processed = []
        for frame, alpha in zip(frames, alphas):
            if bg_mode == BackgroundMode.GREEN_SCREEN:
                result = self._remove_green_spill(frame, alpha)
            elif bg_mode == BackgroundMode.BLACK_BACKGROUND:
                result = self._remove_black_spill(frame, alpha)
            else:
                result = frame.copy()
            processed.append(result)
        return processed
    
    def _remove_green_spill(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """去除绿溢色 - 保守版：只处理明显绿边，保护白色/浅色区域"""
        result = frame.astype(np.float32)
        r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        
        strength = self.config.green_despill_strength
        
        # 1. 严格检测绿色溢色：必须 G 明显大于 R 和 B
        # 条件：G > R + threshold 且 G > B + threshold
        threshold = 15  # 严格阈值，避免误判
        green_tint = (g > r + threshold) & (g > b + threshold)
        
        # 2. 强白色保护 - 避免把白色/灰色去成粉色
        brightness = np.mean(result, axis=2)
        # 白色 = 高亮度 + 低饱和度（RGB接近）
        rgb_diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
        white_mask = (brightness > 180) & (rgb_diff < 30)  # 严格的白色条件
        
        # 3. 只处理边缘区域（alpha 在 0.05-0.95 之间）
        # 完全前景和完全背景不处理
        edge_mask = (alpha > 0.05) & (alpha < 0.95)
        
        # 4. 计算绿色过量（只在边缘且明显发绿的地方）
        green_excess = np.maximum(0, g - np.maximum(r, b))
        
        # 边缘去绿：alpha 越低（越接近背景）→ 去绿越强
        edge_factor = np.clip((0.95 - alpha) / 0.9, 0, 1)  # 0.05→1.0, 0.95→0.0
        despill = green_excess * strength * edge_factor * self.config.edge_despill_factor
        
        # 只处理明显发绿的边缘
        despill = despill * green_tint * edge_mask
        
        # 白色区域：几乎不去绿（保护白色毛发/耳朵）
        despill = np.where(white_mask, despill * 0.05, despill)
        
        # 5. 应用去绿
        g_corrected = g - despill
        
        # 6. 补偿到 R 和 B（保持亮度）
        compensation = despill * 0.5
        r_corrected = r + compensation
        b_corrected = b + compensation
        
        result[:, :, 0] = np.clip(r_corrected, 0, 255)
        result[:, :, 1] = np.clip(g_corrected, 0, 255)
        result[:, :, 2] = np.clip(b_corrected, 0, 255)
        
        return result.astype(np.uint8)
    
    def _remove_black_spill(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """去除黑边/发灰 - 修复版：保护粒子/辉光颜色"""
        result = frame.astype(np.float32)
        
        brightness = np.mean(result, axis=2)
        max_c = np.max(result, axis=2)
        min_c = np.min(result, axis=2)
        color_range = max_c - min_c
        
        # 1. 黑边区域（暗部边缘）
        black_edge = (alpha > 0.05) & (alpha < 0.95) & (brightness < 60)
        
        # 2. 提升暗部 - 保守提升，避免过曝粒子
        boost = self.config.black_despill_strength
        # 根据 alpha 调整提升量：半透明区域提升更多
        lift = np.maximum(0, 50 - brightness) * boost * (1.0 - alpha) * 0.3
        
        for c in range(3):
            result[:, :, c] = np.where(black_edge, result[:, :, c] + lift, result[:, :, c])
        
        # 3. 颜色增强（去灰）- 保护已有颜色的粒子
        hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # 只对灰度区域增强饱和度
        gray_mask = (alpha > 0.1) & (s < 50) & (v > 15) & (color_range < 15)
        s_boost = s * (1.0 + self.config.black_contrast_restore * 0.5)
        s = np.where(gray_mask, np.clip(s_boost, 0, 255), s)
        
        # 4. 暗部亮度提升 - 保护极暗粒子
        dark_mask = (alpha > 0.05) & (v < 60) & (v > 5)  # v > 5 保护纯黑
        v_boost = v * (1.0 + self.config.black_contrast_restore * 0.2)
        v = np.where(dark_mask, np.clip(v_boost, 0, 255), v)
        
        hsv[:, :, 1] = s
        hsv[:, :, 2] = v
        
        result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 5. 粒子颜色保护 - 暗部亮点保持原色
        particle_mask = (alpha > 0.1) & (brightness < 40) & (color_range > 10)
        # 这些区域不过度处理，保持原始颜色
        original = frame.astype(np.float32)
        blend = 0.3  # 保留 30% 原始颜色
        for c in range(3):
            result[:, :, c] = np.where(particle_mask, 
                                       result[:, :, c] * (1 - blend) + original[:, :, c] * blend,
                                       result[:, :, c])
        
        return np.clip(result, 0, 255).astype(np.uint8)
