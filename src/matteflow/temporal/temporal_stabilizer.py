"""时序稳定模块"""

import numpy as np
from typing import List

from ..config import MattingConfig, QualityMode


class TemporalStabilizer:
    """时序稳定器"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
    
    def stabilize(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """
        时序稳定 Alpha
        
        Args:
            alphas: Alpha 帧列表
        
        Returns:
            稳定后的 Alpha 列表
        """
        if len(alphas) <= 1:
            return alphas
        
        if self.config.quality_mode == QualityMode.FAST:
            return self._ema_smooth(alphas)
        elif self.config.quality_mode == QualityMode.STANDARD:
            return self._adaptive_smooth(alphas)
        else:  # HIGH
            return self._optical_flow_smooth(alphas)

    def _transparency_mask(self, alpha: np.ndarray) -> np.ndarray:
        low = float(getattr(self.config, "transparency_temporal_low", 0.03))
        high = float(getattr(self.config, "transparency_temporal_high", 0.75))
        return (alpha > low) & (alpha < high)
    
    def _ema_smooth(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """快速 EMA 平滑"""
        strength = self.config.temporal_strength
        alpha = 1.0 - strength * 0.3  # EMA 系数
        
        smoothed = [alphas[0].copy()]
        for i in range(1, len(alphas)):
            ema = alpha * smoothed[-1] + (1 - alpha) * alphas[i]
            smoothed.append(ema)
        
        return smoothed
    
    def _adaptive_smooth(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """标准自适应平滑 - 增强版"""
        strength = self.config.temporal_strength
        transparency_blend = float(getattr(self.config, "transparency_temporal_blend", 0.20))
        
        # 第一次：前向平滑
        forward = []
        for i, alpha in enumerate(alphas):
            if i == 0:
                forward.append(alpha.copy())
                continue
            
            prev = forward[-1]
            
            # 计算差异
            diff = np.abs(alpha - prev)
            
            # 自适应平滑权重：
            # - 差异大 = 更可能是闪烁，需要更多平滑
            # - 差异小 = 保持稳定，减少平滑
            adaptive_weight = np.clip(diff * 3.0, 0, 1) * strength
            
            # 核心区（alpha 接近 0 或 1）减少平滑，保持边缘清晰
            core_mask = (alpha < 0.02) | (alpha > 0.98)
            adaptive_weight = np.where(core_mask, adaptive_weight * 0.1, adaptive_weight)
            
            # 应用平滑
            result = (1 - adaptive_weight) * alpha + adaptive_weight * prev
            transparency_mask = self._transparency_mask(alpha)
            transparency_result = alpha * (1.0 - transparency_blend) + prev * transparency_blend
            result = np.where(transparency_mask, transparency_result, result)
            forward.append(np.clip(result, 0, 1))
        
        # 第二次：后向平滑（双向）
        backward = [forward[-1].copy()]
        for i in range(len(forward) - 2, -1, -1):
            curr = forward[i]
            next_frame = backward[-1]
            
            diff = np.abs(curr - next_frame)
            adaptive_weight = np.clip(diff * 3.0, 0, 1) * strength * 0.5
            
            core_mask = (curr < 0.02) | (curr > 0.98)
            adaptive_weight = np.where(core_mask, adaptive_weight * 0.1, adaptive_weight)
            
            result = (1 - adaptive_weight) * curr + adaptive_weight * next_frame
            transparency_mask = self._transparency_mask(curr)
            transparency_result = curr * (1.0 - transparency_blend) + next_frame * transparency_blend
            result = np.where(transparency_mask, transparency_result, result)
            backward.append(np.clip(result, 0, 1))
        
        # 合并双向结果
        backward.reverse()
        
        final = []
        for f, b in zip(forward, backward):
            # 取平均，但权重偏向更稳定的值
            final.append((f + b) * 0.5)
        
        return final
    
    def _optical_flow_smooth(self, alphas: List[np.ndarray]) -> List[np.ndarray]:
        """高质量光流辅助平滑（MVP 用双向平滑代替）"""
        return self._adaptive_smooth(alphas)
