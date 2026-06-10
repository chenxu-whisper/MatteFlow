"""Contracts for future competitive green-screen layer composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ShapeContract = dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class LayerCandidate:
    """One named alpha layer that a later composer task can compete."""

    name: str
    alpha: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerOwnership:
    """Per-pixel ownership metadata for a competitive layer result."""

    owner: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class CompetitiveLayerResult:
    """Result contract for a future competitive layer composition step."""

    alpha: np.ndarray
    ownership: LayerOwnership
    candidates: tuple[LayerCandidate, ...]
    debug: dict[str, Any] = field(default_factory=dict)

    def debug_shape_contract(self) -> ShapeContract:
        """Return debug shape metadata without performing composition."""
        contract: ShapeContract = {
            "alpha": tuple(self.alpha.shape),
            "ownership.owner": tuple(self.ownership.owner.shape),
            "ownership.confidence": tuple(self.ownership.confidence.shape),
        }
        for candidate in self.candidates:
            contract[f"candidates.{candidate.name}.alpha"] = tuple(candidate.alpha.shape)
        return contract


class GreenScreenCompetitiveLayerComposer:
    """Validates the Task 1 competitive layer debug shape contract."""

    def validate_debug_shape_contract(
        self,
        result: CompetitiveLayerResult,
    ) -> CompetitiveLayerResult:
        expected_shape = self._require_2d_shape("alpha", result.alpha)
        contract = result.debug_shape_contract()
        for name, shape in contract.items():
            if shape != expected_shape:
                raise ValueError(
                    f"{name} shape {shape} does not match alpha shape {expected_shape}"
                )
        return result

    @staticmethod
    def _require_2d_shape(name: str, value: np.ndarray) -> tuple[int, int]:
        shape = tuple(np.asarray(value).shape)
        if len(shape) != 2:
            raise ValueError(f"{name} must be 2D, got shape {shape}")
        return shape

__all__ = [
    "CompetitiveLayerResult",
    "GreenScreenCompetitiveLayerComposer",
    "LayerCandidate",
    "LayerOwnership",
]
