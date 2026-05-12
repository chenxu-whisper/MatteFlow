"""Matte 融合模块"""

import numpy as np


class MatteFusion:
    """融合 Core/Detail/FX Matte"""
    
    def fuse(
        self,
        core_alpha: np.ndarray,
        detail_alpha: np.ndarray = None,
        fx_alpha: np.ndarray = None,
        weights: tuple = (0.6, 0.25, 0.15)
    ) -> np.ndarray:
        """
        融合三类 Matte
        
        Args:
            core_alpha: 主体核心 matte
            detail_alpha: 细节边缘 matte (可选)
            fx_alpha: 特效半透明 matte (可选)
            weights: (core, detail, fx) 权重
        
        Returns:
            融合后的 alpha
        """
        w_core, w_detail, w_fx = weights
        
        total_weight = w_core
        result = core_alpha * w_core
        
        if detail_alpha is not None:
            result += detail_alpha * w_detail
            total_weight += w_detail
        
        if fx_alpha is not None:
            result += fx_alpha * w_fx
            total_weight += w_fx
        
        return np.clip(result / total_weight, 0, 1) if total_weight > 0 else core_alpha
