import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, MattingConfig  # noqa: E402
from matteflow.matte.candidates.birefnet import BiRefNetCandidateGenerator  # noqa: E402
from matteflow.matte.candidates.matanyone2 import MatAnyone2CandidateGenerator  # noqa: E402
from matteflow.matte.candidates.sam2_guided import SAM2GuidedCandidateGenerator  # noqa: E402
from matteflow.matte.candidates.traditional import TraditionalCandidateGenerator  # noqa: E402
from matteflow.matte.candidates.types import MatteCandidateSequence  # noqa: E402


def test_config_disables_quality_selection_by_default():
    config = MattingConfig()

    assert config.quality_selection_enable is False
    assert config.quality_candidate_models == ("matanyone2", "sam2", "birefnet", "traditional")
    assert config.quality_selection_mode == "region"


def test_quality_birefnet_auto_load_defaults_to_false():
    config = MattingConfig()

    assert config.quality_birefnet_auto_load is False


def test_candidate_sequence_normalizes_alpha_dtype_and_range():
    candidate = MatteCandidateSequence.from_raw(
        name="fake",
        alphas=[np.array([[-1.0, 0.5, 2.0]], dtype=np.float64)],
        confidences=[None],
        source="fake",
        runtime_ms=1.25,
        diagnostics={"available": True},
        frame_shapes=[(1, 3)],
    )

    assert candidate.alphas[0].dtype == np.float32
    assert candidate.alphas[0].tolist() == [[0.0, 0.5, 1.0]]
    assert candidate.runtime_ms == 1.25
    assert candidate.diagnostics == {"available": True}


def test_candidate_sequence_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="alpha shape"):
        MatteCandidateSequence.from_raw(
            name="bad",
            alphas=[np.zeros((2, 2), dtype=np.float32)],
            confidences=[None],
            source="fake",
            runtime_ms=0.0,
            diagnostics={},
            frame_shapes=[(1, 2)],
        )


def test_candidate_sequence_rejects_explicit_empty_confidences():
    with pytest.raises(ValueError, match="confidences length"):
        MatteCandidateSequence.from_raw(
            name="bad",
            alphas=[np.zeros((1, 1), dtype=np.float32)],
            confidences=[],
            source="fake",
            runtime_ms=0.0,
            diagnostics={},
            frame_shapes=[(1, 1)],
        )


def test_traditional_candidate_generator_uses_green_screen_path():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[:, :, 1] = 255
    frame[0, 0] = [255, 0, 0]

    result = TraditionalCandidateGenerator(
        MattingConfig(background_mode=BackgroundMode.GREEN_SCREEN)
    ).generate([frame], frame_shapes=[(2, 2)])

    assert result.candidate is not None
    assert result.candidate.name == "traditional"
    assert result.candidate.alphas[0].shape == (2, 2)
    assert result.candidate.diagnostics["background_mode"] == "green_screen"


def test_sam2_guided_candidate_skips_without_guidance():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    result = SAM2GuidedCandidateGenerator(MattingConfig()).generate(
        [frame],
        frame_shapes=[(2, 2)],
    )

    assert result.candidate is None
    assert result.skipped is True
    assert result.skip_reason.value == "guidance_missing"


def test_matanyone2_candidate_skips_when_engine_is_unavailable():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    result = MatAnyone2CandidateGenerator(MattingConfig()).generate(
        [frame],
        frame_shapes=[(2, 2)],
    )

    assert result.candidate is None
    assert result.skipped is True
    assert result.skip_reason.value == "model_unavailable"


def test_birefnet_candidate_skips_when_engine_is_unavailable():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    result = BiRefNetCandidateGenerator(MattingConfig()).generate(
        [frame],
        frame_shapes=[(2, 2)],
    )

    assert result.candidate is None
    assert result.skipped is True
    assert result.skip_reason.value == "model_unavailable"


class _FakeSequenceEngine:
    model = object()

    def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
        return [np.full(frame.shape[:2], 0.25, dtype=np.float32) for frame in frames]


def test_matanyone2_candidate_wraps_available_sequence_engine():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    result = MatAnyone2CandidateGenerator(
        MattingConfig(),
        engine=_FakeSequenceEngine(),
    ).generate([frame], frame_shapes=[(2, 2)])

    assert result.candidate is not None
    assert result.candidate.name == "matanyone2"
    assert result.candidate.alphas[0].tolist() == [[0.25, 0.25], [0.25, 0.25]]
