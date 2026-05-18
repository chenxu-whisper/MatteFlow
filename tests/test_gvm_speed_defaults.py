import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig
from matteflow.pipeline import MattingPipeline


def test_gvm_speed_defaults_are_explicit():
    config = MattingConfig()

    assert config.gvm_max_internal_size == 768
    assert config.generate_zip_by_default is False
    assert config.preview_quality_mode == "fast"
    assert config.output_fg is False
    assert config.output_matte is True
    assert config.output_comp is False
    assert config.output_processed is True


def test_pipeline_process_returns_stage_timings(tmp_path):
    pipeline = MattingPipeline.__new__(MattingPipeline)
    pipeline.config = MattingConfig()
    pipeline._notify = lambda callback, current, total, stage: None
    pipeline._decode_input = lambda input_path: (
        [np.zeros((4, 4, 3), dtype=np.uint8)],
        {"width": 4, "height": 4},
    )
    pipeline.analyzer = type("Analyzer", (), {"analyze": lambda self, frames: pipeline.config.background_mode})()
    pipeline.refiner = type("Refiner", (), {"refine": lambda self, frames, alphas: alphas})()
    pipeline.despeckle = type(
        "Despeckle",
        (),
        {"process": lambda self, alphas, frames=None, context=None: alphas},
    )()
    pipeline.stabilizer = type("Stabilizer", (), {"stabilize": lambda self, alphas: alphas})()
    pipeline.decontaminate = type("Decontaminate", (), {"process": lambda self, frames, alphas, bg_mode: frames})()
    pipeline._generate_matte = lambda frames, bg_mode, progress_callback: [
        np.ones((4, 4), dtype=np.float32)
    ]
    pipeline._encode_output = lambda frames, alphas, output_dir, meta: None

    result = pipeline.process(tmp_path / "input.mp4", tmp_path / "out")

    assert set(result["timings"]) == {
        "decode",
        "analyze",
        "matte",
        "refine",
        "despeckle",
        "stabilize",
        "decontaminate",
        "encode",
        "total",
    }
