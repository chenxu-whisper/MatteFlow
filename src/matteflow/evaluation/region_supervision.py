"""Weak region supervision for matting quality regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..analysis.region_ownership import RegionOwnership

REGION_FIELDS = (
    "subject",
    "hair_edge",
    "luminous_prop",
    "transparent_effect",
    "background_residue",
    "uncertain_edge",
)


@dataclass(frozen=True)
class RegionExpectation:
    """Weak per-sample region expectations loaded from manifests."""

    required_regions: tuple[str, ...] = ()
    min_region_ratios: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RegionExpectation":
        if not payload:
            return cls()
        required = tuple(str(region) for region in payload.get("required_regions", ()))
        ratios = {
            str(region): float(value)
            for region, value in dict(payload.get("min_region_ratios", {})).items()
        }
        for region in (*required, *ratios.keys()):
            if region not in REGION_FIELDS:
                raise ValueError(f"Unsupported region: {region}")
        return cls(required_regions=required, min_region_ratios=ratios)


@dataclass(frozen=True)
class RegionSupervisionReport:
    """Aggregated region coverage and weak-supervision failures."""

    frame_count: int
    total_pixels: int
    region_pixels: Mapping[str, int]
    region_ratios: Mapping[str, float]
    failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": int(self.frame_count),
            "total_pixels": int(self.total_pixels),
            "region_pixels": dict(self.region_pixels),
            "region_ratios": dict(self.region_ratios),
            "failures": list(self.failures),
        }


class RegionSupervisionEvaluator:
    """Evaluate weak region expectations against ownership masks."""

    def evaluate(
        self,
        *,
        ownerships: Sequence[RegionOwnership],
        expectation: RegionExpectation | None = None,
    ) -> RegionSupervisionReport:
        expectation = expectation or RegionExpectation()
        total_pixels = sum(int(ownership.subject.size) for ownership in ownerships)
        region_pixels = {
            region: sum(
                int(np.count_nonzero(getattr(ownership, region))) for ownership in ownerships
            )
            for region in REGION_FIELDS
        }
        denominator = max(total_pixels, 1)
        region_ratios = {
            region: round(count / denominator, 6) for region, count in region_pixels.items()
        }

        failures: list[str] = []
        for region in expectation.required_regions:
            if region_pixels[region] <= 0:
                failures.append(f"required_region {region} missing")
        for region, minimum in expectation.min_region_ratios.items():
            if region_ratios[region] < minimum:
                failures.append(
                    f"region_ratio {region} {region_ratios[region]:.6f} "
                    f"below minimum {minimum:.6f}"
                )

        return RegionSupervisionReport(
            frame_count=len(ownerships),
            total_pixels=total_pixels,
            region_pixels=region_pixels,
            region_ratios=region_ratios,
            failures=tuple(failures),
        )
