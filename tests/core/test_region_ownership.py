import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.region_ownership import RegionOwnershipAnalyzer
from matteflow.analysis.region_ownership import RegionOwnership
from matteflow.config import MattingConfig
from matteflow.config import BackgroundMode
from matteflow.pipeline import MattingPipeline
from matteflow.refine.color_decontaminate import ColorDecontaminate
from matteflow.refine.despeckle import Despeckle
from matteflow.refine.edge_refiner import EdgeRefiner
from matteflow.refine.effect_prop_repair import EffectPropRepair


def test_region_ownership_classifies_luminous_prop_and_translucent_effect():
    frame = np.full((96, 128, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((96, 128), dtype=np.float32)
    base_alpha = np.zeros((96, 128), dtype=np.float32)

    cv2.circle(frame, (34, 48), 14, [235, 172, 70], thickness=-1)
    cv2.circle(frame, (34, 48), 6, [250, 238, 180], thickness=-1)
    cv2.circle(alpha, (34, 48), 14, 0.35, thickness=-1)
    cv2.circle(base_alpha, (34, 48), 14, 1.0, thickness=-1)

    frame[64:70, 72:112] = [210, 220, 245]
    alpha[64:70, 72:112] = 0.42
    base_alpha[64:70, 72:112] = 0.35

    ownership = RegionOwnershipAnalyzer().analyze(frame, alpha, base_alpha)

    prop_region = np.zeros_like(alpha, dtype=np.uint8)
    cv2.circle(prop_region, (34, 48), 10, 1, thickness=-1)
    prop_region = prop_region.astype(bool)
    effect_region = np.zeros_like(alpha, dtype=bool)
    effect_region[64:70, 72:112] = True

    assert float(ownership.luminous_prop[prop_region].mean()) >= 0.95
    assert float(ownership.transparent_effect[effect_region].mean()) >= 0.85
    assert not np.any(ownership.background_residue[prop_region])


def test_region_ownership_detects_background_residue_and_uncertain_edge():
    frame = np.full((80, 100, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((80, 100), dtype=np.float32)
    base_alpha = np.zeros((80, 100), dtype=np.float32)

    frame[8:34, 8:40] = [232, 232, 228]
    alpha[8:34, 8:40] = 0.92
    base_alpha[8:34, 8:40] = 1.0

    frame[45:65, 55:80] = [230, 150, 225]
    alpha[45:65, 55:80] = 1.0
    alpha[43:67, 53:82] = np.maximum(alpha[43:67, 53:82], 0.45)
    base_alpha[45:65, 55:80] = 1.0

    ownership = RegionOwnershipAnalyzer().analyze(frame, alpha, base_alpha)

    document_region = np.zeros_like(alpha, dtype=bool)
    document_region[10:32, 10:38] = True
    edge_region = np.zeros_like(alpha, dtype=bool)
    edge_region[43:67, 53:82] = True
    edge_region[45:65, 55:80] = False

    assert float(ownership.background_residue[document_region].mean()) >= 0.90
    assert float(ownership.uncertain_edge[edge_region].mean()) >= 0.70
    assert not np.any(ownership.background_residue[55:60, 62:72])


def test_effect_prop_repair_uses_region_ownership_luminous_prop_mask():
    frame = np.full((32, 32, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((32, 32), dtype=np.float32)
    base_alpha = np.zeros((32, 32), dtype=np.float32)
    base_alpha[12:20, 12:20] = 1.0
    prop_mask = np.zeros((32, 32), dtype=bool)
    prop_mask[12:20, 12:20] = True

    class FakeRegionAnalyzer:
        def analyze(self, frame_arg, alpha_arg, base_alpha_arg):
            empty = np.zeros_like(prop_mask)
            return RegionOwnership(
                subject=empty,
                hair_edge=empty,
                luminous_prop=prop_mask,
                transparent_effect=empty,
                background_residue=empty,
                uncertain_edge=empty,
            )

    repair = EffectPropRepair(MattingConfig())
    repair._region_analyzer = FakeRegionAnalyzer()

    repaired = repair._repair_single(frame, alpha, base_alpha)

    assert float(repaired[prop_mask].mean()) >= 0.98


def test_color_decontaminate_uses_region_ownership_for_prop_green_projection():
    frame = np.full((24, 24, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((24, 24), dtype=np.float32)
    prop_mask = np.zeros((24, 24), dtype=bool)
    prop_mask[8:16, 8:16] = True
    frame[prop_mask] = [88, 102, 76]
    alpha[prop_mask] = 1.0

    class FakeRegionAnalyzer:
        def analyze(self, frame_arg, alpha_arg, base_alpha_arg=None):
            empty = np.zeros_like(prop_mask)
            return RegionOwnership(
                subject=empty,
                hair_edge=empty,
                luminous_prop=prop_mask,
                transparent_effect=empty,
                background_residue=empty,
                uncertain_edge=empty,
            )

    decontaminate = ColorDecontaminate(MattingConfig())
    decontaminate._region_analyzer = FakeRegionAnalyzer()

    processed = decontaminate._remove_green_spill(frame, alpha).astype(np.float32)

    processed_f = processed.astype(np.float32)
    assert float((processed_f[:, :, 1] - processed_f[:, :, 0])[prop_mask].mean()) <= 6.0


def test_edge_refiner_restores_luminous_props_from_region_ownership():
    frame = np.full((32, 32, 3), [0, 220, 40], dtype=np.uint8)
    original = np.zeros((32, 32), dtype=np.float32)
    refined = np.zeros((32, 32), dtype=np.float32)
    prop_mask = np.zeros((32, 32), dtype=bool)
    prop_mask[10:22, 10:22] = True
    original[prop_mask] = 1.0
    refined[prop_mask] = 0.25

    class FakeRegionAnalyzer:
        def analyze(self, frame_arg, alpha_arg, base_alpha_arg=None):
            empty = np.zeros_like(prop_mask)
            return RegionOwnership(
                subject=empty,
                hair_edge=empty,
                luminous_prop=prop_mask,
                transparent_effect=empty,
                background_residue=empty,
                uncertain_edge=empty,
            )

    refiner = EdgeRefiner(MattingConfig())
    refiner._region_analyzer = FakeRegionAnalyzer()

    restored = refiner._restore_warm_luminous_props(frame, original, refined)

    assert float(restored[prop_mask].mean()) >= 0.98


def test_despeckle_restores_luminous_props_from_region_ownership():
    frame = np.full((32, 32, 3), [0, 220, 40], dtype=np.uint8)
    original = np.zeros((32, 32), dtype=np.uint8)
    cleaned = np.zeros((32, 32), dtype=np.uint8)
    prop_mask = np.zeros((32, 32), dtype=bool)
    prop_mask[12:20, 12:20] = True
    original[prop_mask] = 255
    cleaned[prop_mask] = 32

    class FakeRegionAnalyzer:
        def analyze(self, frame_arg, alpha_arg, base_alpha_arg=None):
            empty = np.zeros_like(prop_mask)
            return RegionOwnership(
                subject=empty,
                hair_edge=empty,
                luminous_prop=prop_mask,
                transparent_effect=empty,
                background_residue=empty,
                uncertain_edge=empty,
            )

    despeckle = Despeckle(MattingConfig())
    despeckle._region_analyzer = FakeRegionAnalyzer()

    restored = despeckle._restore_warm_luminous_props(original, cleaned, frame=frame)

    assert int(restored[prop_mask].min()) == 255


def test_despeckle_restores_transparent_effects_from_region_ownership():
    frame = np.full((32, 32, 3), [0, 220, 40], dtype=np.uint8)
    original = np.zeros((32, 32), dtype=np.uint8)
    cleaned = np.zeros((32, 32), dtype=np.uint8)
    effect_mask = np.zeros((32, 32), dtype=bool)
    effect_mask[12, 12] = True
    effect_mask[20, 20] = True
    original[effect_mask] = 96

    class FakeRegionAnalyzer:
        def analyze(self, frame_arg, alpha_arg, base_alpha_arg=None):
            empty = np.zeros_like(effect_mask)
            return RegionOwnership(
                subject=empty,
                hair_edge=empty,
                luminous_prop=empty,
                transparent_effect=effect_mask,
                background_residue=empty,
                uncertain_edge=empty,
            )

    despeckle = Despeckle(MattingConfig())
    despeckle._region_analyzer = FakeRegionAnalyzer()

    restored = despeckle._restore_supported_soft_alpha(
        original,
        cleaned,
        ksize=5,
        frame=frame,
        context={},
    )

    assert int(restored[effect_mask].min()) == 96


def test_pipeline_builds_shared_region_ownership_context():
    frame = np.full((16, 16, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((16, 16), dtype=np.float32)
    expected_mask = np.zeros((16, 16), dtype=bool)
    expected_mask[4:8, 4:8] = True
    expected = RegionOwnership(
        subject=np.zeros_like(expected_mask),
        hair_edge=np.zeros_like(expected_mask),
        luminous_prop=expected_mask,
        transparent_effect=np.zeros_like(expected_mask),
        background_residue=np.zeros_like(expected_mask),
        uncertain_edge=np.zeros_like(expected_mask),
    )
    calls = []

    class FakeRegionAnalyzer:
        def analyze(self, frame_arg, alpha_arg, base_alpha_arg=None):
            calls.append((frame_arg, alpha_arg, base_alpha_arg))
            return expected

    pipeline = MattingPipeline(MattingConfig())
    pipeline.region_analyzer = FakeRegionAnalyzer()

    context = pipeline._build_region_context([frame], [alpha])

    assert context["region_ownership"] == [expected]
    assert len(calls) == 1
    assert calls[0][0] is frame
    assert calls[0][1] is alpha
    assert calls[0][2] is None


def test_refine_modules_reuse_pipeline_region_ownership_context():
    frame = np.full((24, 24, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((24, 24), dtype=np.float32)
    prop_mask = np.zeros((24, 24), dtype=bool)
    prop_mask[8:16, 8:16] = True
    frame[prop_mask] = [88, 102, 76]
    alpha[prop_mask] = 1.0
    ownership = RegionOwnership(
        subject=np.zeros_like(prop_mask),
        hair_edge=np.zeros_like(prop_mask),
        luminous_prop=prop_mask,
        transparent_effect=np.zeros_like(prop_mask),
        background_residue=np.zeros_like(prop_mask),
        uncertain_edge=np.zeros_like(prop_mask),
    )
    context = {"region_ownership": [ownership]}

    class RaisingRegionAnalyzer:
        def analyze(self, frame_arg, alpha_arg, base_alpha_arg=None):
            raise AssertionError("module should reuse pipeline region ownership")

    decontaminate = ColorDecontaminate(MattingConfig())
    decontaminate._region_analyzer = RaisingRegionAnalyzer()
    processed = decontaminate.process(
        [frame],
        [alpha],
        BackgroundMode.GREEN_SCREEN,
        context=context,
    )[0]

    refiner = EdgeRefiner(MattingConfig())
    refiner._region_analyzer = RaisingRegionAnalyzer()
    refined = refiner.refine([frame], [alpha], context=context)[0]

    processed_f = processed.astype(np.float32)
    assert float((processed_f[:, :, 1] - processed_f[:, :, 0])[prop_mask].mean()) <= 6.0
    assert float(refined[prop_mask].mean()) >= 0.98


def test_despeckle_reuses_pipeline_region_ownership_context():
    frame = np.full((32, 32, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((32, 32), dtype=np.float32)
    effect_mask = np.zeros((32, 32), dtype=bool)
    effect_mask[12, 12] = True
    alpha[effect_mask] = 96.0 / 255.0
    ownership = RegionOwnership(
        subject=np.zeros_like(effect_mask),
        hair_edge=np.zeros_like(effect_mask),
        luminous_prop=np.zeros_like(effect_mask),
        transparent_effect=effect_mask,
        background_residue=np.zeros_like(effect_mask),
        uncertain_edge=np.zeros_like(effect_mask),
    )

    class RaisingRegionAnalyzer:
        def analyze(self, frame_arg, alpha_arg, base_alpha_arg=None):
            raise AssertionError("despeckle should reuse pipeline region ownership")

    config = MattingConfig()
    config.despeckle_threshold = 0.5
    despeckle = Despeckle(config)
    despeckle._region_analyzer = RaisingRegionAnalyzer()

    cleaned = despeckle.process(
        [alpha],
        frames=[frame],
        context={"region_ownership": [ownership]},
    )[0]

    assert float(cleaned[effect_mask].min()) >= 96.0 / 255.0


def test_pipeline_passes_shared_region_context_to_repair_stages(monkeypatch, tmp_path):
    config = MattingConfig()
    config.output_fg = False
    config.output_comp = False
    config.output_processed = False
    config.output_matte = False
    pipeline = MattingPipeline(config)
    frame = np.full((8, 8, 3), [0, 220, 40], dtype=np.uint8)
    alpha = np.zeros((8, 8), dtype=np.float32)
    ownership = RegionOwnership(
        subject=np.zeros((8, 8), dtype=bool),
        hair_edge=np.zeros((8, 8), dtype=bool),
        luminous_prop=np.zeros((8, 8), dtype=bool),
        transparent_effect=np.zeros((8, 8), dtype=bool),
        background_residue=np.zeros((8, 8), dtype=bool),
        uncertain_edge=np.zeros((8, 8), dtype=bool),
    )
    captured = {}

    monkeypatch.setattr(
        pipeline,
        "_decode_input",
        lambda input_path: ([frame], {"width": 8, "height": 8, "fps": 1.0}),
    )
    monkeypatch.setattr(pipeline.analyzer, "analyze", lambda frames_arg: BackgroundMode.GREEN_SCREEN)
    monkeypatch.setattr(pipeline, "_generate_matte", lambda frames_arg, bg_mode, progress_callback: [alpha])
    monkeypatch.setattr(
        pipeline,
        "_build_region_context",
        lambda frames_arg, alphas_arg, base_alphas=None: {"region_ownership": [ownership]},
    )

    def fake_refine(frames_arg, alphas_arg, context=None):
        captured["refine"] = context
        return alphas_arg

    def fake_despeckle(alphas_arg, frames=None, context=None):
        captured["despeckle"] = context
        return alphas_arg

    def fake_effect_repair(frames_arg, alphas_arg, bg_mode, active_model=None, context=None):
        captured["effect_prop_repair"] = context
        return alphas_arg

    def fake_decontaminate(frames_arg, alphas_arg, bg_mode, context=None):
        captured["decontaminate"] = context
        return frames_arg

    monkeypatch.setattr(pipeline.refiner, "refine", fake_refine)
    monkeypatch.setattr(pipeline.despeckle, "process", fake_despeckle)
    monkeypatch.setattr(pipeline.effect_prop_repair, "process", fake_effect_repair)
    monkeypatch.setattr(pipeline.decontaminate, "process", fake_decontaminate)

    pipeline.process(tmp_path / "input.png", tmp_path / "out")

    assert captured["refine"]["region_ownership"] == [ownership]
    assert captured["despeckle"]["region_ownership"] == [ownership]
    assert captured["effect_prop_repair"]["region_ownership"] == [ownership]
    assert captured["decontaminate"]["region_ownership"] == [ownership]
