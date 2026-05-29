import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, MattingConfig, QualityMode
from matteflow.errors import JobCancelledError, ProgressCallbackError
from matteflow.matte.matanyone2_matte import MatAnyone2Matte
from matteflow import pipeline as pipeline_module
from matteflow.pipeline import MattingPipeline


class RecordingEncoder:
    def __init__(self, on_image_write=None):
        self.image_writes = []
        self.grayscale_writes = []
        self._on_image_write = on_image_write

    def encode_image(self, image, output_path):
        self.image_writes.append(Path(output_path))
        if self._on_image_write is not None:
            self._on_image_write(Path(output_path))

    def encode_grayscale(self, image, output_path):
        del image
        self.grayscale_writes.append(Path(output_path))


def test_pipeline_stops_immediately_when_progress_callback_raises(monkeypatch, tmp_path):
    pipeline = MattingPipeline(MattingConfig())

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (_ for _ in ()).throw(
            AssertionError("decode should not run after progress callback failure")
        ),
    )

    def raising_progress_callback(current, total, stage):
        del current, total, stage
        raise RuntimeError("progress callback exploded")

    with pytest.raises(ProgressCallbackError, match="progress callback exploded") as exc_info:
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=raising_progress_callback,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_pipeline_preserves_existing_progress_callback_error(monkeypatch, tmp_path):
    pipeline = MattingPipeline(MattingConfig())
    original_error = ProgressCallbackError("typed progress callback failure")

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (_ for _ in ()).throw(
            AssertionError("decode should not run after progress callback failure")
        ),
    )

    def raising_progress_callback(current, total, stage):
        del current, total, stage
        raise original_error

    with pytest.raises(ProgressCallbackError, match="typed progress callback failure") as exc_info:
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=raising_progress_callback,
        )

    assert exc_info.value is original_error
    assert exc_info.value.__cause__ is None


def test_pipeline_preserves_matting_stage_name_when_on_progress_callback_raises(
    monkeypatch, tmp_path
):
    pipeline = MattingPipeline(MattingConfig())
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(pipeline.analyzer, "analyze", lambda decoded_frames: BackgroundMode.GREEN_SCREEN)

    def fake_generate_sequence(decoded_frames, bg_mode, progress_callback=None, cancel_check=None):
        del decoded_frames, bg_mode, cancel_check
        assert progress_callback is not None
        progress_callback(1, 1)
        raise AssertionError("matte generation should stop after on_progress failure")

    monkeypatch.setattr(pipeline.hybrid_matte, "generate_sequence", fake_generate_sequence)

    def raising_progress_callback(current, total, stage):
        del current, total
        if stage == "matting":
            raise RuntimeError("matting progress sink exploded")

    with pytest.raises(
        ProgressCallbackError,
        match="Progress callback failed during matting: matting progress sink exploded",
    ) as exc_info:
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=raising_progress_callback,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_pipeline_passes_cancel_check_into_hybrid_matte(monkeypatch, tmp_path):
    pipeline = MattingPipeline(MattingConfig())
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(pipeline.analyzer, "analyze", lambda decoded_frames: BackgroundMode.GREEN_SCREEN)

    def fake_generate_sequence(decoded_frames, bg_mode, progress_callback=None, cancel_check=None):
        captured["frames"] = decoded_frames
        captured["bg_mode"] = bg_mode
        captured["cancel_check"] = cancel_check
        raise JobCancelledError("cancelled inside matte engine")

    monkeypatch.setattr(pipeline.hybrid_matte, "generate_sequence", fake_generate_sequence)

    with pytest.raises(JobCancelledError, match="cancelled inside matte engine"):
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            cancel_check=lambda: False,
        )

    assert captured["frames"] == frames
    assert captured["bg_mode"] == BackgroundMode.GREEN_SCREEN
    assert callable(captured["cancel_check"])
    assert captured["cancel_check"]() is False


def test_matanyone2_generate_sequence_passes_cancel_check_to_runtime():
    matte = MatAnyone2Matte.__new__(MatAnyone2Matte)
    matte.model = object()
    matte.config = MattingConfig()
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    captured = {}

    def fake_run_sequence_inference(decoded_frames, first_frame_mask, cancel_check=None):
        captured["frames"] = decoded_frames
        captured["first_frame_mask"] = first_frame_mask
        captured["cancel_check"] = cancel_check
        return [np.ones((2, 2), dtype=np.float32)]

    matte._run_sequence_inference = fake_run_sequence_inference

    result = matte.generate_sequence(frames, cancel_check=lambda: True)

    assert len(result) == 1
    assert captured["frames"] == frames
    assert captured["first_frame_mask"] is None
    assert callable(captured["cancel_check"])
    assert captured["cancel_check"]() is True


def test_pipeline_rejects_alpha_frame_count_mismatch(monkeypatch, tmp_path):
    pipeline = MattingPipeline(MattingConfig())
    frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8),
    ]

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(pipeline.analyzer, "analyze", lambda decoded_frames: BackgroundMode.GREEN_SCREEN)
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda decoded_frames, bg_mode, progress_callback, cancel_check=None: [
            np.ones((2, 2), dtype=np.float32)
        ],
    )

    with pytest.raises(RuntimeError, match="Alpha count mismatch after matte"):
        pipeline.process(tmp_path / "input.png", tmp_path / "out")


def test_pipeline_stops_before_decoding_when_cancel_requested_by_decoding_progress(monkeypatch, tmp_path):
    pipeline = MattingPipeline(MattingConfig())
    cancel_requested = {"value": False}

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (_ for _ in ()).throw(
            AssertionError("decode should not run after decoding progress cancellation")
        ),
    )

    def progress_callback(current, total, stage):
        del current, total
        if stage == "decoding":
            cancel_requested["value"] = True

    with pytest.raises(JobCancelledError, match="Processing cancelled by user"):
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=progress_callback,
            cancel_check=lambda: cancel_requested["value"],
        )


def test_pipeline_stops_before_analyzing_when_cancel_requested_by_analyzing_progress(monkeypatch, tmp_path):
    pipeline = MattingPipeline(MattingConfig())
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    cancel_requested = {"value": False}

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(
        pipeline.analyzer,
        "analyze",
        lambda decoded_frames: (_ for _ in ()).throw(
            AssertionError("analyze should not run after analyzing progress cancellation")
        ),
    )

    def progress_callback(current, total, stage):
        del current, total
        if stage == "analyzing":
            cancel_requested["value"] = True

    with pytest.raises(JobCancelledError, match="Processing cancelled by user"):
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=progress_callback,
            cancel_check=lambda: cancel_requested["value"],
        )


def test_pipeline_stops_before_matting_when_cancel_requested_by_matting_progress(monkeypatch, tmp_path):
    pipeline = MattingPipeline(MattingConfig())
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    cancel_requested = {"value": False}

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(pipeline.analyzer, "analyze", lambda decoded_frames: BackgroundMode.GREEN_SCREEN)
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda decoded_frames, bg_mode, progress_callback, cancel_check=None: (_ for _ in ()).throw(
            AssertionError("matte generation should not run after matting progress cancellation")
        ),
    )

    def progress_callback(current, total, stage):
        del current, total
        if stage == "matting":
            cancel_requested["value"] = True

    with pytest.raises(JobCancelledError, match="Processing cancelled by user"):
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=progress_callback,
            cancel_check=lambda: cancel_requested["value"],
        )


@pytest.mark.parametrize(
    ("stage_name", "patch_target", "patch_method_name", "replacement"),
    [
        (
            "refining",
            lambda pipeline: pipeline.refiner,
            "refine",
            lambda decoded_frames, alphas: (_ for _ in ()).throw(
                AssertionError("refine should not run after refining progress cancellation")
            ),
        ),
        (
            "despeckling",
            lambda pipeline: pipeline.despeckle,
            "process",
            lambda alphas, frames=None, context=None: (_ for _ in ()).throw(
                AssertionError("despeckle should not run after despeckling progress cancellation")
            ),
        ),
        (
            "stabilizing",
            lambda pipeline: pipeline.stabilizer,
            "stabilize",
            lambda alphas: (_ for _ in ()).throw(
                AssertionError("stabilize should not run after stabilizing progress cancellation")
            ),
        ),
        (
            "decontaminating",
            lambda pipeline: pipeline.decontaminate,
            "process",
            lambda decoded_frames, alphas, bg_mode: (_ for _ in ()).throw(
                AssertionError("decontaminate should not run after decontaminating progress cancellation")
            ),
        ),
    ],
)
def test_pipeline_stops_before_expensive_stage_when_cancel_requested_by_stage_progress(
    monkeypatch, tmp_path, stage_name, patch_target, patch_method_name, replacement
):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.HIGH,
        output_matte=False,
        output_processed=True,
    )
    pipeline = MattingPipeline(config)
    frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8),
    ]
    cancel_requested = {"value": False}

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(pipeline.analyzer, "analyze", lambda decoded_frames: BackgroundMode.GREEN_SCREEN)
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda decoded_frames, bg_mode, progress_callback, cancel_check=None: [
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ],
    )
    monkeypatch.setattr(patch_target(pipeline), patch_method_name, replacement)

    def progress_callback(current, total, stage):
        del current, total
        if stage == stage_name:
            cancel_requested["value"] = True

    with pytest.raises(JobCancelledError, match="Processing cancelled by user"):
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=progress_callback,
            cancel_check=lambda: cancel_requested["value"],
        )


def test_pipeline_stops_before_encoding_when_cancel_requested_by_encoding_progress(monkeypatch, tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.FAST,
        output_matte=False,
        output_processed=True,
    )
    pipeline = MattingPipeline(config)
    frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8),
    ]
    cancel_requested = {"value": False}
    pipeline.encoder = RecordingEncoder()

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda decoded_frames, bg_mode, progress_callback, cancel_check=None: [
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ],
    )
    monkeypatch.setattr(
        pipeline.decontaminate,
        "process",
        lambda decoded_frames, alphas, bg_mode: decoded_frames,
    )

    def progress_callback(current, total, stage):
        del current, total
        if stage == "encoding":
            cancel_requested["value"] = True

    with pytest.raises(JobCancelledError, match="Processing cancelled by user"):
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=progress_callback,
            cancel_check=lambda: cancel_requested["value"],
        )

    assert pipeline.encoder.image_writes == []
    assert pipeline.encoder.grayscale_writes == []


def test_pipeline_stops_during_encoding_after_cancel_request(monkeypatch, tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.FAST,
        output_matte=False,
        output_processed=True,
    )
    pipeline = MattingPipeline(config)
    frames = [
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2, 3), dtype=np.uint8),
    ]
    cancel_requested = {"value": False}

    def request_cancel_after_first_write(_output_path: Path) -> None:
        cancel_requested["value"] = True

    pipeline.encoder = RecordingEncoder(on_image_write=request_cancel_after_first_write)

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda decoded_frames, bg_mode, progress_callback, cancel_check=None: [
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        ],
    )
    monkeypatch.setattr(
        pipeline.decontaminate,
        "process",
        lambda decoded_frames, alphas, bg_mode: decoded_frames,
    )

    with pytest.raises(JobCancelledError, match="Processing cancelled by user"):
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=lambda current, total, stage: None,
            cancel_check=lambda: cancel_requested["value"],
        )

    assert len(pipeline.encoder.image_writes) == 1


def test_pipeline_stops_after_encoding_when_cancel_requested_by_last_write(monkeypatch, tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.FAST,
        output_matte=False,
        output_processed=True,
    )
    pipeline = MattingPipeline(config)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)]
    cancel_requested = {"value": False}

    def request_cancel_after_last_write(_output_path: Path) -> None:
        cancel_requested["value"] = True

    pipeline.encoder = RecordingEncoder(on_image_write=request_cancel_after_last_write)

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: (frames, {"width": 2, "height": 2, "fps": 1.0}),
    )
    monkeypatch.setattr(
        pipeline,
        "_generate_matte",
        lambda decoded_frames, bg_mode, progress_callback, cancel_check=None: [
            np.ones((2, 2), dtype=np.float32),
        ],
    )
    monkeypatch.setattr(
        pipeline.decontaminate,
        "process",
        lambda decoded_frames, alphas, bg_mode: decoded_frames,
    )

    with pytest.raises(JobCancelledError, match="Processing cancelled by user"):
        pipeline.process(
            tmp_path / "input.png",
            tmp_path / "out",
            progress_callback=lambda current, total, stage: None,
            cancel_check=lambda: cancel_requested["value"],
        )


def test_encode_output_does_not_build_extra_rgba_variant_for_processed_only(monkeypatch, tmp_path):
    config = MattingConfig(
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.FAST,
        output_fg=False,
        output_matte=False,
        output_comp=False,
        output_processed=True,
        output_premultiplied=False,
    )
    pipeline = MattingPipeline(config)
    pipeline.encoder = RecordingEncoder()
    frames = [np.ones((2, 2, 3), dtype=np.uint8) * 255]
    alphas = [np.full((2, 2), 0.5, dtype=np.float32)]
    dstack_calls = {"count": 0}
    real_dstack = pipeline_module.np.dstack

    def counting_dstack(parts):
        dstack_calls["count"] += 1
        return real_dstack(parts)

    monkeypatch.setattr(pipeline_module.np, "dstack", counting_dstack)

    pipeline._encode_output(frames, alphas, tmp_path / "out", {"fps": 1.0})

    assert dstack_calls["count"] == 1

