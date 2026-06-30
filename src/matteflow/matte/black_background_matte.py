"""黑底特效 Matte 生成模块 - 优化版"""

import cv2
import numpy as np

from ..config import MattingConfig
from .black_effect_enhancer import BlackEffectEnhancer


class BlackBackgroundMatte:
    """黑底抠图入口：结构 baseline + 统一黑底增强器。"""

    def __init__(self, config: MattingConfig):
        self.config = config
        self._effect_enhancer = BlackEffectEnhancer(config)
        self.last_effect_enhancement = {}
        self.effect_enhancement_history = []

    def generate(self, frame: np.ndarray) -> np.ndarray:
        """
        生成黑底 Alpha Matte。

        这里的 baseline 只负责明显主体/亮部结构；黑底烟雾、辉光、
        粒子和主体暗边统一交给 BlackEffectEnhancer 处理，避免旧的
        颜色/亮度规则在主流程里重复抬升背景噪声。

        Args:
            frame: RGB 图像 (H, W, 3), uint8

        Returns:
            alpha: Alpha 通道 (H, W), float32 [0, 1]
        """
        frame_f = frame.astype(np.float32, copy=False)
        gray = np.mean(frame_f, axis=2)

        solid_start = 60.0
        transition_start = 45.0
        alpha = np.zeros_like(gray, dtype=np.float32)
        transition = (gray > transition_start) & (gray < solid_start)
        alpha[transition] = (gray[transition] - transition_start) / (solid_start - transition_start) * 0.75
        alpha[gray >= solid_start] = 1.0

        alpha_u8 = (alpha * 255).astype(np.uint8)
        alpha_smooth = cv2.GaussianBlur(alpha_u8, (3, 3), 0).astype(np.float32) / 255.0

        base_alpha = np.clip(alpha_smooth, 0, 1).astype(np.float32)
        enhanced = self.enhance_effects(frame, base_alpha)
        return enhanced.alpha

    def enhance_effects(self, frame: np.ndarray, base_alpha: np.ndarray):
        """Run the unified enhancer and keep video-level diagnostics."""
        enhanced = self._effect_enhancer.enhance(frame, base_alpha)
        self._record_effect_enhancement(enhanced.diagnostics)
        return enhanced

    def reset_effect_enhancement_history(self) -> None:
        self.effect_enhancement_history = []
        self.last_effect_enhancement = {}

    def _record_effect_enhancement(self, diagnostics: dict) -> None:
        self.effect_enhancement_history.append(dict(diagnostics))
        self.last_effect_enhancement = self._aggregate_effect_enhancement_history()

    def _aggregate_effect_enhancement_history(self) -> dict:
        count_fields = (
            "smoke_pixels",
            "glow_pixels",
            "particle_pixels",
            "subject_edge_pixels",
            "black_residue_suppressed_pixels",
        )
        summary = {"frames": len(self.effect_enhancement_history)}
        for field in count_fields:
            summary[field] = int(
                sum(int(item.get(field, 0)) for item in self.effect_enhancement_history)
            )
        summary["mean_alpha_delta"] = float(
            np.mean(
                [
                    float(item.get("mean_alpha_delta", 0.0))
                    for item in self.effect_enhancement_history
                ]
            )
        )
        return summary
