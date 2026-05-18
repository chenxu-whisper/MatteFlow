import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, QualityMode
from matteflow.errors import ProcessingError
from matteflow.service import MatteFlowService, ProcessJobParams, ProcessResult


def test_process_job_params_freezes_mutable_config_overrides(tmp_path):
    samples = [(1, 2, 3)]

    params = ProcessJobParams(
        input_path=tmp_path / "input.png",
        output_dir=tmp_path / "out",
        config_overrides={"key_samples": samples},
    )
    samples.append((4, 5, 6))

    assert params.config_overrides["key_samples"] == [(1, 2, 3)]
    with pytest.raises(TypeError):
        params.config_overrides["key_strength"] = 2.0


def test_service_passes_snapshot_to_pipeline(tmp_path):
    captured = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def process(self, input_path, output_dir, progress_callback=None):
            captured["input_path"] = input_path
            captured["output_dir"] = output_dir
            captured["progress_callback"] = progress_callback
            return {
                "background_mode": "green_screen",
                "frame_count": 3,
                "processing_time": 1.25,
                "timings": {"decode": 0.1},
            }

    service = MatteFlowService(pipeline_factory=FakePipeline)
    params = ProcessJobParams(
        input_path=tmp_path / "input.png",
        output_dir=tmp_path / "out",
        background_mode=BackgroundMode.GREEN_SCREEN,
        quality_mode=QualityMode.HIGH,
        use_ai=False,
        ai_model="gvm",
        config_overrides={"key_strength": 1.7},
    )

    result = service.process(params, progress_callback=lambda *_: None)

    assert captured["config"].background_mode == BackgroundMode.GREEN_SCREEN
    assert captured["config"].quality_mode == QualityMode.HIGH
    assert captured["config"].use_ai is False
    assert captured["config"].ai_model == "gvm"
    assert captured["config"].key_strength == 1.7
    assert captured["input_path"] == tmp_path / "input.png"
    assert captured["output_dir"] == tmp_path / "out"
    assert captured["progress_callback"] is not None
    assert isinstance(result, ProcessResult)
    assert result.success is True
    assert result.frame_count == 3
    assert result.background_mode == "green_screen"


def test_service_wraps_pipeline_errors(tmp_path):
    class FailingPipeline:
        def __init__(self, config):
            pass

        def process(self, input_path, output_dir, progress_callback=None):
            raise RuntimeError("CUDA out of memory")

    service = MatteFlowService(pipeline_factory=FailingPipeline)
    params = ProcessJobParams(input_path=tmp_path / "input.png", output_dir=tmp_path / "out")

    with pytest.raises(ProcessingError) as exc_info:
        service.process(params)

    assert "GPU memory" in str(exc_info.value)
    assert "CUDA out of memory" in str(exc_info.value)
