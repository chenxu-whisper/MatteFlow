import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig
from matteflow.pipeline import MattingPipeline
from matteflow.matte import chroma_key_postprocess, corridorkey_matte, gvm_matte, hybrid_matte, matanyone2_matte


def test_pipeline_process_logs_stage_summary(tmp_path, caplog):
    pipeline = MattingPipeline.__new__(MattingPipeline)
    pipeline.config = MattingConfig()
    pipeline._notify = lambda callback, current, total, stage: None
    pipeline._decode_input = lambda input_path: (
        [np.zeros((4, 4, 3), dtype=np.uint8)],
        {"width": 4, "height": 4},
    )
    pipeline.analyzer = type("Analyzer", (), {"analyze": lambda self, frames: pipeline.config.background_mode})()
    pipeline.refiner = type("Refiner", (), {"refine": lambda self, frames, alphas: alphas})()
    pipeline.despeckle = type("Despeckle", (), {"process": lambda self, alphas: alphas})()
    pipeline.stabilizer = type("Stabilizer", (), {"stabilize": lambda self, alphas: alphas})()
    pipeline.decontaminate = type("Decontaminate", (), {"process": lambda self, frames, alphas, bg_mode: frames})()
    pipeline._generate_matte = lambda frames, bg_mode, progress_callback: [
        np.ones((4, 4), dtype=np.float32)
    ]
    pipeline._encode_output = lambda frames, alphas, output_dir, meta: None

    caplog.set_level(logging.INFO, logger="matteflow.pipeline")
    pipeline.process(tmp_path / "input.mp4", tmp_path / "out")

    assert "Loaded 1 frames" in caplog.text
    assert "Stage timings summary" in caplog.text
    assert "Process completed" in caplog.text


def test_hybrid_green_screen_logs_selected_ai_engine(caplog, monkeypatch):
    hybrid = hybrid_matte.HybridMatte.__new__(hybrid_matte.HybridMatte)
    hybrid.config = MattingConfig()
    hybrid.config.use_ai = True
    hybrid.config.ai_enhance = False
    hybrid.config.ai_model = "gvm"
    hybrid.gvm = type(
        "FakeGVM",
        (),
        {"model": object(), "generate_sequence": lambda self, frames: [np.ones((2, 2), dtype=np.float32)]},
    )()
    hybrid.matanyone2 = None
    hybrid.corridorkey = None
    hybrid.rembg = None
    hybrid.rmbg = None
    hybrid.birefnet = None
    hybrid.rvm = None
    hybrid.sam2 = None
    hybrid.green_matte = type(
        "FakeGreenMatte",
        (),
        {"generate": lambda self, frame: np.zeros((2, 2), dtype=np.float32)},
    )()

    monkeypatch.setattr(
        chroma_key_postprocess,
        "apply_chroma_key_postprocess",
        lambda alphas, config: alphas,
    )

    caplog.set_level(logging.INFO, logger="matteflow.matte.hybrid_matte")
    frame = np.array(
        [
            [[248, 152, 205], [152, 138, 128]],
            [[42, 132, 62], [240, 240, 245]],
        ],
        dtype=np.uint8,
    )

    result = hybrid._green_screen_matte([frame], None)

    assert np.allclose(result[0], 1.0)
    assert "Using GVM for green screen" in caplog.text


def test_hybrid_green_screen_preserves_base_transparency_effects(monkeypatch):
    hybrid = hybrid_matte.HybridMatte.__new__(hybrid_matte.HybridMatte)
    hybrid.config = MattingConfig()
    hybrid.config.use_ai = True
    hybrid.config.ai_enhance = False
    hybrid.config.ai_model = "gvm"
    hybrid.config.transparency_preserve = 0.7

    ai_alpha = np.array(
        [[0.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    base_alpha = np.array(
        [[0.8, 0.45], [0.14, 0.9]],
        dtype=np.float32,
    )

    hybrid.gvm = type(
        "FakeGVM",
        (),
        {"model": object(), "generate_sequence": lambda self, frames: [ai_alpha.copy()]},
    )()
    hybrid.matanyone2 = None
    hybrid.corridorkey = None
    hybrid.rembg = None
    hybrid.rmbg = None
    hybrid.birefnet = None
    hybrid.rvm = None
    hybrid.sam2 = None
    hybrid.green_matte = type(
        "FakeGreenMatte",
        (),
        {"generate": lambda self, frame: base_alpha.copy()},
    )()

    monkeypatch.setattr(
        chroma_key_postprocess,
        "apply_chroma_key_postprocess",
        lambda alphas, config: alphas,
    )

    frame = np.array(
        [
            [[248, 152, 205], [152, 138, 128]],
            [[42, 132, 62], [240, 240, 245]],
        ],
        dtype=np.uint8,
    )

    result = hybrid._green_screen_matte([frame], None)

    assert result[0][0, 0] > 0.3
    assert result[0][0, 1] < 0.04
    assert result[0][1, 0] == 0.0
    assert result[0][1, 1] >= 1.0


def test_hybrid_green_screen_suppresses_screen_colored_effect_blob():
    hybrid = hybrid_matte.HybridMatte.__new__(hybrid_matte.HybridMatte)
    hybrid.config = MattingConfig()
    hybrid.config.transparency_preserve = 0.7

    base_alpha = np.array([[0.9, 0.9, 0.9, 0.0]], dtype=np.float32)
    ai_alpha = np.zeros((1, 4), dtype=np.float32)
    frame = np.array(
        [[[248, 152, 205], [152, 138, 128], [42, 132, 62], [30, 128, 58]]],
        dtype=np.uint8,
    )

    result = hybrid._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert result[0, 0] > 0.4
    assert result[0, 1] < 0.08
    assert result[0, 2] < 0.02


def test_hybrid_green_screen_keeps_solid_foreground_when_ai_misses_it():
    hybrid = hybrid_matte.HybridMatte.__new__(hybrid_matte.HybridMatte)
    hybrid.config = MattingConfig()
    hybrid.config.transparency_preserve = 0.7

    base_alpha = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    ai_alpha = np.zeros((1, 4), dtype=np.float32)
    frame = np.array(
        [[[212, 218, 238], [246, 130, 190], [170, 145, 150], [30, 128, 58]]],
        dtype=np.uint8,
    )

    result = hybrid._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert result[0, 0] > 0.9
    assert result[0, 1] > 0.9
    assert result[0, 2] < 0.12
    assert result[0, 3] == 0.0


def test_hybrid_green_screen_keeps_soft_blue_ear_when_ai_misses_it():
    hybrid = hybrid_matte.HybridMatte.__new__(hybrid_matte.HybridMatte)
    hybrid.config = MattingConfig()
    hybrid.config.transparency_preserve = 0.7

    base_alpha = np.array([[0.35, 0.35, 0.35, 0.20]], dtype=np.float32)
    ai_alpha = np.zeros((1, 4), dtype=np.float32)
    frame = np.array(
        [[[208, 214, 238], [185, 178, 205], [170, 145, 150], [34, 128, 58]]],
        dtype=np.uint8,
    )

    result = hybrid._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert result[0, 0] > 0.75
    assert result[0, 1] > 0.75
    assert result[0, 2] < 0.08
    assert result[0, 3] == 0.0


def test_gvm_generate_logs_when_model_is_unavailable(caplog):
    matte = gvm_matte.GVMMatte.__new__(gvm_matte.GVMMatte)
    matte.config = MattingConfig()
    matte.model = None

    caplog.set_level(logging.INFO, logger="matteflow.matte.gvm_matte")
    result = matte.generate([np.zeros((2, 2, 3), dtype=np.uint8)])

    assert result[0].shape == (2, 2)
    assert "Model not available" in caplog.text


def test_matanyone2_generate_logs_when_model_is_unavailable(caplog):
    matte = matanyone2_matte.MatAnyone2Matte.__new__(matanyone2_matte.MatAnyone2Matte)
    matte.config = MattingConfig()
    matte.model = None

    caplog.set_level(logging.INFO, logger="matteflow.matte.matanyone2_matte")
    result = matte.generate([np.zeros((2, 2, 3), dtype=np.uint8)])

    assert result[0].shape == (2, 2)
    assert "Model not available" in caplog.text


def test_matanyone2_run_sequence_preserves_rgb_input_order(tmp_path):
    captured = {}

    class FakeProcessor:
        def process_frames(self, input_frames, mask_frame, output_dir, frame_names, clip_name):
            captured["frame"] = input_frames[0].copy()
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            alpha = np.full((2, 2), 255, dtype=np.uint8)
            from PIL import Image

            Image.fromarray(alpha, mode="L").save(out_dir / f"{frame_names[0]}.png")

    matte = matanyone2_matte.MatAnyone2Matte.__new__(matanyone2_matte.MatAnyone2Matte)
    matte.config = MattingConfig()
    matte.model = FakeProcessor()

    rgb_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb_frame[:, :, 0] = 10
    rgb_frame[:, :, 1] = 20
    rgb_frame[:, :, 2] = 30

    result = matte._run_sequence_inference([rgb_frame], None)

    assert np.array_equal(captured["frame"], rgb_frame)
    assert result[0].shape == (2, 2)


def test_corridorkey_generate_logs_when_model_is_unavailable(caplog, monkeypatch):
    matte = corridorkey_matte.CorridorKeyMatte.__new__(corridorkey_matte.CorridorKeyMatte)
    matte.config = MattingConfig()
    matte.model = None

    class FakeGreenScreenMatte:
        def __init__(self, config):
            self.config = config

        def generate(self, frame):
            return np.zeros(frame.shape[:2], dtype=np.float32)

    monkeypatch.setattr(corridorkey_matte, "GreenScreenMatte", FakeGreenScreenMatte, raising=False)
    caplog.set_level(logging.INFO, logger="matteflow.matte.corridorkey_matte")
    result = matte.generate(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result.shape == (2, 2)
    assert "Model unavailable, falling back to GreenScreenMatte" in caplog.text
