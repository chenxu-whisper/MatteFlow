import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig  # noqa: E402
from matteflow.matte.candidates.birefnet import BiRefNetCandidateGenerator  # noqa: E402
from matteflow.matte.candidates.types import CandidateSkipReason  # noqa: E402


class FakeBiRefNetEngine:
    model = object()

    def generate_sequence(self, frames, progress_callback=None):
        return [np.full(frame.shape[:2], 0.8, dtype=np.float32) for frame in frames]


class FailingBiRefNetEngine:
    model = object()

    def generate_sequence(self, frames, progress_callback=None):
        raise RuntimeError("inference failed")


def test_birefnet_candidate_skips_without_engine_or_auto_load():
    generator = BiRefNetCandidateGenerator(MattingConfig())

    result = generator.generate(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        frame_shapes=[(2, 2)],
    )

    assert result.skipped is True
    assert result.skip_reason == CandidateSkipReason.MODEL_UNAVAILABLE
    assert "auto-load" in result.message


def test_birefnet_candidate_generates_with_fake_engine():
    generator = BiRefNetCandidateGenerator(MattingConfig(), engine=FakeBiRefNetEngine())

    result = generator.generate(
        [np.zeros((3, 4, 3), dtype=np.uint8)],
        frame_shapes=[(3, 4)],
    )

    assert result.candidate is not None
    assert result.candidate.name == "birefnet"
    assert result.candidate.alphas[0].shape == (3, 4)
    assert float(result.candidate.alphas[0].mean()) == pytest.approx(0.8)


def test_birefnet_candidate_auto_loads_engine_when_enabled():
    config = MattingConfig()
    config.quality_birefnet_auto_load = True
    calls = []

    def factory(factory_config):
        calls.append(factory_config)
        return FakeBiRefNetEngine()

    generator = BiRefNetCandidateGenerator(config, engine_factory=factory)

    result = generator.generate(
        [np.zeros((2, 3, 3), dtype=np.uint8)],
        frame_shapes=[(2, 3)],
    )

    assert calls == [config]
    assert result.candidate is not None
    assert result.candidate.diagnostics["model"] == "birefnet"


def test_birefnet_candidate_skips_when_auto_load_fails():
    config = MattingConfig()
    config.quality_birefnet_auto_load = True

    def factory(factory_config):
        raise RuntimeError("missing weights")

    generator = BiRefNetCandidateGenerator(config, engine_factory=factory)

    result = generator.generate(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        frame_shapes=[(2, 2)],
    )

    assert result.skipped is True
    assert result.skip_reason == CandidateSkipReason.MODEL_UNAVAILABLE
    assert "missing weights" in result.message


def test_birefnet_candidate_skips_when_inference_fails():
    generator = BiRefNetCandidateGenerator(MattingConfig(), engine=FailingBiRefNetEngine())

    result = generator.generate(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        frame_shapes=[(2, 2)],
    )

    assert result.skipped is True
    assert result.skip_reason == CandidateSkipReason.GENERATION_FAILED
    assert "inference failed" in result.message
