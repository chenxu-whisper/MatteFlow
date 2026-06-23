"""Effect prop integrity repair for green-screen composites."""

from __future__ import annotations

import numpy as np

from ..analysis.region_ownership import RegionOwnershipAnalyzer
from ..config import BackgroundMode, MattingConfig
from ..matte.green_screen_matte import GreenScreenMatte


class EffectPropRepair:
    """Rebuild small luminous prop alpha from structural color and key evidence."""

    def __init__(self, config: MattingConfig):
        self.config = config
        self._green_matte = GreenScreenMatte(config)
        self._region_analyzer = RegionOwnershipAnalyzer()

    def process(self, frames, alphas, bg_mode: BackgroundMode, active_model: str | None = None, context=None):
        if bg_mode != BackgroundMode.GREEN_SCREEN:
            return alphas
        if active_model not in {"gvm", "traditional_green_fallback", None}:
            return alphas

        repaired = []
        context = context or {}
        for index, (frame, alpha) in enumerate(zip(frames, alphas)):
            base_alpha = self._green_matte.generate(frame)
            ownership = self._ownership_from_context(context, index)
            repaired.append(self._repair_single(frame, alpha, base_alpha, ownership=ownership))
        return repaired

    def _repair_single(self, frame: np.ndarray, alpha: np.ndarray, base_alpha: np.ndarray, ownership=None) -> np.ndarray:
        alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
        base_f = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        if ownership is None:
            ownership = self._region_analyzer.analyze(frame, alpha_f, base_f)
        prop_mask = ownership.luminous_prop
        if not np.any(prop_mask):
            return alpha_f

        repaired = alpha_f.copy()
        restore_alpha = np.maximum(base_f, 0.985)
        repaired[prop_mask] = np.maximum(repaired[prop_mask], restore_alpha[prop_mask])
        return np.clip(repaired, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _ownership_from_context(context: dict, index: int):
        ownerships = context.get("region_ownership") if context else None
        if ownerships is None or index >= len(ownerships):
            return None
        return ownerships[index]
