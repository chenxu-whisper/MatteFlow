"""Matte 融合模块"""

import numpy as np


class MatteFusion:
    """融合 Core/Detail/FX Matte"""

    def fuse(
        self,
        core_alpha: np.ndarray,
        detail_alpha: np.ndarray = None,
        fx_alpha: np.ndarray = None,
        weights: tuple = (0.6, 0.25, 0.15),
        core_confidence: np.ndarray = None,
        detail_confidence: np.ndarray = None,
        fx_confidence: np.ndarray = None,
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
        confidence_maps = (core_confidence, detail_confidence, fx_confidence)
        if any(confidence is not None for confidence in confidence_maps):
            return self._fuse_with_confidence(
                core_alpha,
                detail_alpha,
                fx_alpha,
                weights,
                core_confidence,
                detail_confidence,
                fx_confidence,
            )

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

    def _fuse_with_confidence(
        self,
        core_alpha: np.ndarray,
        detail_alpha: np.ndarray = None,
        fx_alpha: np.ndarray = None,
        weights: tuple = (0.6, 0.25, 0.15),
        core_confidence: np.ndarray = None,
        detail_confidence: np.ndarray = None,
        fx_confidence: np.ndarray = None,
    ) -> np.ndarray:
        layers = [
            (core_alpha, core_confidence, weights[0]),
            (detail_alpha, detail_confidence, weights[1]),
            (fx_alpha, fx_confidence, weights[2]),
        ]
        numerator = None
        denominator = None

        for alpha, confidence, base_weight in layers:
            if alpha is None:
                continue
            alpha_f = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
            if confidence is None:
                confidence_f = np.ones_like(alpha_f, dtype=np.float32)
            else:
                confidence_f = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
                if confidence_f.shape != alpha_f.shape:
                    raise ValueError(
                        f"confidence shape {confidence_f.shape} does not match alpha shape {alpha_f.shape}"
                    )
            layer_weight = np.square(confidence_f) * float(base_weight)
            numerator = alpha_f * layer_weight if numerator is None else numerator + alpha_f * layer_weight
            denominator = layer_weight if denominator is None else denominator + layer_weight

        if numerator is None or denominator is None:
            return np.clip(np.asarray(core_alpha, dtype=np.float32), 0.0, 1.0)
        return np.clip(np.divide(numerator, np.maximum(denominator, 1e-6)), 0.0, 1.0)
