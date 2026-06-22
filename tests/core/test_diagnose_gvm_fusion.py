import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "diagnose_gvm_fusion.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("diagnose_gvm_fusion", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_default_input_paths_includes_green_frames_and_videos():
    module = _load_script_module()

    inputs = module._resolve_default_input_paths(PROJECT_ROOT)

    assert PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg" in inputs
    assert PROJECT_ROOT / "assets" / "frame" / "test_frame_4.jpg" in inputs
    assert PROJECT_ROOT / "assets" / "video" / "test_green_1.mp4" in inputs
    assert PROJECT_ROOT / "assets" / "video" / "test_green_4.mp4" in inputs
    assert all(path.exists() for path in inputs)
    assert all("black" not in path.name for path in inputs)


def test_select_sample_frame_indices_covers_start_middle_end_without_duplicates():
    module = _load_script_module()

    assert module._select_sample_frame_indices(1) == [0]
    assert module._select_sample_frame_indices(2) == [0, 1]
    assert module._select_sample_frame_indices(5) == [0, 2, 4]
    assert module._select_sample_frame_indices(10) == [0, 5, 9]


def test_build_summary_entry_includes_model_selection_metadata():
    module = _load_script_module()

    entry = module._build_summary_entry(
        input_name="test_green_4.mp4",
        sample_name="test_green_4_f00045",
        selected_model="corridorkey",
        selected_source="fallback_corridorkey",
        fallback_model="corridorkey",
        fallback_applied=True,
        base_alpha_mean=0.35,
        ai_alpha_mean=0.01,
        old_alpha_mean=0.16,
        new_alpha_mean=0.72,
        mean_abs_diff=0.56,
        entity_pixels=123,
        effect_pixels=45,
        transition_pixels=67,
        entity_old_mean=0.12,
        entity_new_mean=0.92,
        entity_gvm_mean=0.01,
        effect_old_mean=0.98,
        effect_new_mean=0.40,
        effect_base_mean=0.88,
        subject_conf_mean=0.30,
        subject_conf_entity_mean=0.83,
        subject_conf_effect_mean=0.22,
        old_effect_mean=0.06,
        new_effect_mean=0.02,
        old_solid_mean=0.10,
        new_solid_mean=0.71,
        selected_mean=0.91,
        selected_vs_gvm_mean_delta=0.90,
    )

    assert entry["selected_model"] == "corridorkey"
    assert entry["selected_source"] == "fallback_corridorkey"
    assert entry["fallback_model"] == "corridorkey"
    assert entry["fallback_applied"] is True
    assert entry["selected_mean"] == 0.91
    assert entry["selected_vs_gvm_mean_delta"] == 0.90


def test_collect_sample_diagnostics_reports_gvm_sequence_fallback():
    module = _load_script_module()

    class _ConstantMatte:
        def generate(self, frame):
            return np.full(frame.shape[:2], 0.55, dtype=np.float32)

    class _SequenceEngine:
        def __init__(self, value: float):
            self.model = object()
            self.value = value

        def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
            del progress_callback, cancel_check
            return [np.full(frame.shape[:2], self.value, dtype=np.float32) for frame in frames]

    matte = module.HybridMatte(module.MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7))
    matte.green_matte = _ConstantMatte()
    matte.gvm = _SequenceEngine(0.0)
    matte.corridorkey = _SequenceEngine(0.92)
    frame = np.full((1, 1, 3), [190, 120, 200], dtype=np.uint8)
    frames = [frame.copy(), frame.copy(), frame.copy()]

    entry, _artifacts = module._collect_sample_diagnostics(
        matte=matte,
        input_name="test_green_4.mp4",
        sample_name="test_green_4_f00045",
        frame=frame,
        sequence_frames=frames,
    )

    assert entry["selected_model"] == "corridorkey"
    assert entry["selected_source"] == "fallback_corridorkey"
    assert entry["fallback_model"] == "corridorkey"
    assert entry["fallback_applied"] is True
    assert entry["selected_mean"] > entry["gvm_mean"]
    assert entry["selected_vs_gvm_mean_delta"] > 0.0


def test_collect_input_diagnostics_uses_sample_sequence_for_video_fallback():
    module = _load_script_module()

    class _ConstantMatte:
        def generate(self, frame):
            return np.full(frame.shape[:2], 0.35, dtype=np.float32)

    class _SequenceEngine:
        def __init__(self, value: float):
            self.model = object()
            self.value = value

        def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
            del progress_callback, cancel_check
            return [np.full(frame.shape[:2], self.value, dtype=np.float32) for frame in frames]

    matte = module.HybridMatte(module.MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7))
    matte.green_matte = _ConstantMatte()
    matte.gvm = _SequenceEngine(0.0)
    matte.corridorkey = _SequenceEngine(0.92)
    frame = np.full((1, 1, 3), [180, 180, 180], dtype=np.uint8)
    samples = [("sample_a", frame.copy()), ("sample_b", frame.copy()), ("sample_c", frame.copy())]

    entries = module._collect_input_diagnostics(
        matte=matte,
        input_name="test_green_4.mp4",
        samples=samples,
    )

    assert len(entries) == 3
    summary_entries = [entry for entry, _artifacts in entries]
    assert all(entry["selected_model"] == "corridorkey" for entry in summary_entries)
    assert all(entry["selected_source"] == "fallback_corridorkey" for entry in summary_entries)
    assert all(entry["fallback_applied"] is True for entry in summary_entries)


def test_collect_sample_diagnostics_marks_sequence_gvm_source_when_sequence_context_beats_single_frame():
    module = _load_script_module()

    class _ConstantMatte:
        def generate(self, frame):
            return np.full(frame.shape[:2], 0.35, dtype=np.float32)

    class _SequenceAwareEngine:
        def __init__(self):
            self.model = object()

        def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
            del progress_callback, cancel_check
            value = 0.05 if len(frames) == 1 else 0.20
            return [np.full(frame.shape[:2], value, dtype=np.float32) for frame in frames]

    matte = module.HybridMatte(module.MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7))
    matte.green_matte = _ConstantMatte()
    matte.gvm = _SequenceAwareEngine()
    frame = np.full((1, 1, 3), [180, 180, 180], dtype=np.uint8)
    frames = [frame.copy(), frame.copy(), frame.copy()]

    entry, _artifacts = module._collect_sample_diagnostics(
        matte=matte,
        input_name="test_green_1.mp4",
        sample_name="test_green_1_f00031",
        frame=frame,
        sequence_frames=frames,
    )

    assert entry["selected_model"] == "gvm"
    assert entry["selected_source"] == "sequence_gvm"
    assert entry["selected_mean"] > entry["gvm_mean"]


def test_collect_input_diagnostics_uses_per_sample_sequence_alpha_metrics():
    module = _load_script_module()

    class _ConstantMatte:
        def generate(self, frame):
            return np.full(frame.shape[:2], 0.35, dtype=np.float32)

    class _PerFrameSequenceEngine:
        def __init__(self):
            self.model = object()

        def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
            del progress_callback, cancel_check
            if len(frames) == 1:
                return [np.full(frame.shape[:2], 0.05, dtype=np.float32) for frame in frames]
            values = [0.20, 0.40, 0.60]
            return [
                np.full(frame.shape[:2], values[index], dtype=np.float32)
                for index, frame in enumerate(frames)
            ]

    matte = module.HybridMatte(module.MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7))
    matte.green_matte = _ConstantMatte()
    matte.gvm = _PerFrameSequenceEngine()
    frame = np.full((1, 1, 3), [180, 180, 180], dtype=np.uint8)
    samples = [("sample_a", frame.copy()), ("sample_b", frame.copy()), ("sample_c", frame.copy())]

    entries = module._collect_input_diagnostics(
        matte=matte,
        input_name="test_green_1.mp4",
        samples=samples,
    )

    summary_entries = [entry for entry, _artifacts in entries]
    assert [entry["selected_source"] for entry in summary_entries] == ["sequence_gvm"] * 3
    assert np.allclose([entry["selected_mean"] for entry in summary_entries], [0.20, 0.40, 0.60])
    assert np.allclose([entry["selected_vs_gvm_mean_delta"] for entry in summary_entries], [0.15, 0.35, 0.55])


def test_collect_sample_diagnostics_includes_region_level_selected_gains():
    module = _load_script_module()

    class _ConstantMatte:
        def __init__(self, alpha):
            self.alpha = np.asarray(alpha, dtype=np.float32)

        def generate(self, frame):
            del frame
            return self.alpha.copy()

    class _SequenceEngine:
        def __init__(self, alpha):
            self.model = object()
            self.alpha = np.asarray(alpha, dtype=np.float32)

        def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
            del progress_callback, cancel_check
            return [self.alpha.copy() for _ in frames]

    matte = module.HybridMatte(module.MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7))
    matte.green_matte = _ConstantMatte([[0.80, 0.70, 0.30]])
    matte.gvm = _SequenceEngine([[0.20, 0.10, 0.40]])
    matte._maybe_fallback_degenerate_gvm_sequence = lambda *_args, **_kwargs: (
        "corridorkey",
        [np.array([[0.90, 0.60, 0.50]], dtype=np.float32)],
    )
    matte._green_screen_subject_confidence = lambda *_args, **_kwargs: np.array([[0.90, 0.20, 0.40]], dtype=np.float32)
    matte._smoothstep = lambda array, _low, _high: np.asarray(array, dtype=np.float32)
    matte._green_screen_effect_color_weight = lambda _frame: np.array([[0.10, 0.80, 0.10]], dtype=np.float32)
    matte._green_screen_ai_subject_layer = lambda ai_alpha, *_args, **_kwargs: np.asarray(ai_alpha, dtype=np.float32)
    matte._green_screen_solid_layer = lambda *_args, **_kwargs: np.zeros((1, 3), dtype=np.float32)
    matte._green_screen_effect_layer = lambda *_args, **_kwargs: np.zeros((1, 3), dtype=np.float32)
    matte._merge_green_screen_effects = lambda *_args, **_kwargs: [
        np.array([[0.95, 0.50, 0.30]], dtype=np.float32)
    ]
    module._legacy_merge = lambda *_args, **_kwargs: (
        np.array([[0.85, 0.40, 0.20]], dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
        np.zeros((1, 3), dtype=np.float32),
    )

    frame = np.full((1, 3, 3), 180, dtype=np.uint8)

    entry, _artifacts = module._collect_sample_diagnostics(
        matte=matte,
        input_name="test_green_4.mp4",
        sample_name="test_green_4_f00045",
        frame=frame,
    )

    assert np.isclose(entry["entity_selected_mean"], 0.90)
    assert np.isclose(entry["entity_selected_vs_gvm_mean_delta"], 0.70)
    assert np.isclose(entry["effect_selected_mean"], 0.60)
    assert np.isclose(entry["effect_selected_vs_gvm_mean_delta"], 0.50)
    assert np.isclose(entry["transition_selected_mean"], 0.50)
    assert np.isclose(entry["transition_selected_vs_gvm_mean_delta"], 0.10)


def test_collect_sample_diagnostics_includes_fallback_quality_gate_runtime_metrics():
    module = _load_script_module()

    class _ConstantMatte:
        def generate(self, frame):
            return np.full(frame.shape[:2], 0.55, dtype=np.float32)

    class _SequenceEngine:
        def __init__(self, value: float):
            self.model = object()
            self.value = value

        def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
            del progress_callback, cancel_check
            return [np.full(frame.shape[:2], self.value, dtype=np.float32) for frame in frames]

    matte = module.HybridMatte(module.MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7))
    matte.green_matte = _ConstantMatte()
    matte.gvm = _SequenceEngine(0.0)
    matte.corridorkey = _SequenceEngine(0.92)
    matte._compute_region_weighted_fallback_quality = lambda *_args, **_kwargs: {
        "entity_delta": 0.40,
        "effect_delta": -0.05,
        "transition_delta": 0.03,
        "global_mean_delta": 0.01,
        "weighted_score": 0.25,
    }
    frame = np.full((1, 1, 3), [190, 120, 200], dtype=np.uint8)
    frames = [frame.copy(), frame.copy(), frame.copy()]

    entry, _artifacts = module._collect_sample_diagnostics(
        matte=matte,
        input_name="test_green_4.mp4",
        sample_name="test_green_4_f00045",
        frame=frame,
        sequence_frames=frames,
    )

    assert np.isclose(entry["fallback_weighted_score"], 0.25)
    assert np.isclose(entry["fallback_entity_delta"], 0.40)
    assert np.isclose(entry["fallback_effect_delta"], -0.05)
    assert np.isclose(entry["fallback_transition_delta"], 0.03)
    assert np.isclose(entry["fallback_global_mean_delta"], 0.01)
    assert entry["fallback_effect_damage_blocked"] is True
    assert entry["fallback_score_blocked"] is False


def test_collect_sample_diagnostics_includes_crop_debug_metadata_for_large_diff():
    module = _load_script_module()

    class _ConstantMatte:
        def generate(self, frame):
            del frame
            return np.zeros((3, 3), dtype=np.float32)

    class _SequenceEngine:
        def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
            del progress_callback, cancel_check
            return [np.zeros(frame.shape[:2], dtype=np.float32) for frame in frames]

    matte = module.HybridMatte(module.MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7))
    matte.green_matte = _ConstantMatte()
    matte.gvm = _SequenceEngine()
    matte._green_screen_subject_confidence = lambda *_args, **_kwargs: np.zeros((3, 3), dtype=np.float32)
    matte._smoothstep = lambda array, _low, _high: np.asarray(array, dtype=np.float32)
    matte._green_screen_effect_color_weight = lambda _frame: np.zeros((3, 3), dtype=np.float32)
    matte._green_screen_ai_subject_layer = lambda *_args, **_kwargs: np.zeros((3, 3), dtype=np.float32)
    matte._green_screen_solid_layer = lambda *_args, **_kwargs: np.zeros((3, 3), dtype=np.float32)
    matte._green_screen_effect_layer = lambda *_args, **_kwargs: np.zeros((3, 3), dtype=np.float32)
    matte._merge_green_screen_effects = lambda *_args, **_kwargs: [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.6, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    ]
    module._legacy_merge = lambda *_args, **_kwargs: (
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
    )

    frame = np.full((3, 3, 3), 180, dtype=np.uint8)

    entry, artifacts = module._collect_sample_diagnostics(
        matte=matte,
        input_name="test_green_crop.mp4",
        sample_name="test_green_crop_f00000",
        frame=frame,
    )

    assert entry["debug_crop_exported"] is True
    assert entry["debug_crop_top"] == 0
    assert entry["debug_crop_left"] == 0
    assert entry["debug_crop_height"] == 3
    assert entry["debug_crop_width"] == 3
    assert np.isclose(entry["debug_crop_peak_abs_diff"], 0.6)
    assert artifacts["debug_crop"] is not None
    assert artifacts["debug_crop"]["frame_rgb"].shape == (3, 3, 3)
    assert artifacts["debug_crop"]["new_alpha"].shape == (3, 3)


def test_collect_sample_diagnostics_skips_crop_debug_for_small_diff():
    module = _load_script_module()

    class _ConstantMatte:
        def generate(self, frame):
            del frame
            return np.zeros((3, 3), dtype=np.float32)

    class _SequenceEngine:
        def generate_sequence(self, frames, progress_callback=None, cancel_check=None):
            del progress_callback, cancel_check
            return [np.zeros(frame.shape[:2], dtype=np.float32) for frame in frames]

    matte = module.HybridMatte(module.MattingConfig(use_ai=True, ai_model="gvm", transparency_preserve=0.7))
    matte.green_matte = _ConstantMatte()
    matte.gvm = _SequenceEngine()
    matte._green_screen_subject_confidence = lambda *_args, **_kwargs: np.zeros((3, 3), dtype=np.float32)
    matte._smoothstep = lambda array, _low, _high: np.asarray(array, dtype=np.float32)
    matte._green_screen_effect_color_weight = lambda _frame: np.zeros((3, 3), dtype=np.float32)
    matte._green_screen_ai_subject_layer = lambda *_args, **_kwargs: np.zeros((3, 3), dtype=np.float32)
    matte._green_screen_solid_layer = lambda *_args, **_kwargs: np.zeros((3, 3), dtype=np.float32)
    matte._green_screen_effect_layer = lambda *_args, **_kwargs: np.zeros((3, 3), dtype=np.float32)
    matte._merge_green_screen_effects = lambda *_args, **_kwargs: [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.04, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    ]
    module._legacy_merge = lambda *_args, **_kwargs: (
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
    )

    frame = np.full((3, 3, 3), 180, dtype=np.uint8)

    entry, artifacts = module._collect_sample_diagnostics(
        matte=matte,
        input_name="test_green_crop.mp4",
        sample_name="test_green_crop_f00001",
        frame=frame,
    )

    assert entry["debug_crop_exported"] is False
    assert entry["debug_crop_top"] is None
    assert entry["debug_crop_peak_abs_diff"] is None
    assert artifacts["debug_crop"] is None


def test_set_debug_crop_prefix_adds_relative_prefix_for_exported_crop():
    module = _load_script_module()

    entry = {
        "sample": "test_green_crop_f00000",
        "debug_crop_exported": True,
    }

    module._set_debug_crop_prefix(entry)

    assert entry["debug_crop_prefix"] == "crops/test_green_crop_f00000_crop"


def test_set_debug_crop_prefix_appends_effect_risk_suffix_when_present():
    module = _load_script_module()

    entry = {
        "sample": "test_green_crop_f00000",
        "debug_crop_exported": True,
        "debug_effect_risk": "bright_cool_effect_risk",
    }

    module._set_debug_crop_prefix(entry)

    assert (
        entry["debug_crop_prefix"]
        == "crops/test_green_crop_f00000_crop_bright_cool_effect_risk"
    )


def test_set_debug_crop_prefix_leaves_missing_prefix_when_crop_not_exported():
    module = _load_script_module()

    entry = {
        "sample": "test_green_crop_f00001",
        "debug_crop_exported": False,
    }

    module._set_debug_crop_prefix(entry)

    assert entry["debug_crop_prefix"] is None


def test_build_summary_payload_groups_samples_by_input_and_aggregates_counts():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_4.mp4",
            "sample": "test_green_4_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": True,
            "fallback_score_blocked": True,
            "selected_vs_gvm_mean_delta": 0.04,
            "fallback_weighted_score": -0.50,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.80,
            "debug_crop_prefix": "crops/test_green_4_f00000_crop",
        },
        {
            "input": "test_green_4.mp4",
            "sample": "test_green_4_f00045",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": True,
            "fallback_score_blocked": True,
            "selected_vs_gvm_mean_delta": 0.05,
            "fallback_weighted_score": -0.52,
        },
        {
            "input": "test_frame_3.jpg",
            "sample": "test_frame_3",
            "selected_model": "gvm",
            "selected_source": "single_frame_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.0,
            "fallback_weighted_score": None,
        },
    ]

    payload = module._build_summary_payload(samples)

    assert set(payload.keys()) == {"samples", "inputs"}
    assert payload["samples"] == samples
    assert len(payload["inputs"]) == 2

    video_entry = next(entry for entry in payload["inputs"] if entry["input"] == "test_green_4.mp4")
    assert video_entry["sample_count"] == 2
    assert video_entry["selected_source_counts"] == {"sequence_gvm": 2}
    assert video_entry["selected_model_counts"] == {"gvm": 2}
    assert video_entry["fallback_quality_evaluated_count"] == 2
    assert video_entry["fallback_quality_gate_passed_count"] == 0
    assert video_entry["fallback_effect_damage_blocked_count"] == 2
    assert video_entry["fallback_score_blocked_count"] == 2
    assert np.isclose(video_entry["mean_selected_vs_gvm_mean_delta"], 0.045)
    assert np.isclose(video_entry["mean_fallback_weighted_score"], -0.51)

    image_entry = next(entry for entry in payload["inputs"] if entry["input"] == "test_frame_3.jpg")
    assert image_entry["sample_count"] == 1
    assert image_entry["fallback_quality_evaluated_count"] == 0
    assert image_entry["mean_fallback_weighted_score"] is None


def test_build_summary_payload_marks_effect_damage_as_dominant_reason():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_4.mp4",
            "sample": "test_green_4_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": True,
            "fallback_score_blocked": True,
            "fallback_effect_delta": -0.062,
            "selected_vs_gvm_mean_delta": 0.04,
            "fallback_weighted_score": -0.50,
        },
        {
            "input": "test_green_4.mp4",
            "sample": "test_green_4_f00045",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": True,
            "fallback_score_blocked": False,
            "fallback_effect_delta": -0.041,
            "selected_vs_gvm_mean_delta": 0.05,
            "fallback_weighted_score": -0.20,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.40,
            "debug_crop_prefix": "crops/test_green_4_f00045_crop_bright_cool_effect_risk",
            "debug_effect_risk": "bright_cool_effect_risk",
        },
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]
    sample_entry = next(
        entry for entry in payload["samples"] if entry["sample"] == "test_green_4_f00045"
    )

    assert (
        input_entry["dominant_decision_reason"]
        == "fallback_blocked_by_effect_damage_bright_cool_effect_risk"
    )
    assert input_entry["dominant_effect_risk"] == "bright_cool_effect_risk"
    assert (
        input_entry["recommended_action"]
        == "inspect bright cool effect-risk crops first"
    )
    assert (
        sample_entry["debug_focus_reason"]
        == "fallback_effect_delta=-0.041 suggests fallback would damage transparent effect regions"
    )


def test_build_summary_payload_marks_sequence_gvm_retained_when_no_fallback_evaluation_occurs():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_1.mp4",
            "sample": "test_green_1_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.01,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.80,
            "debug_crop_prefix": "crops/test_green_1_f00000_crop",
            "entity_selected_vs_gvm_mean_delta": 0.12,
        },
        {
            "input": "test_green_1.mp4",
            "sample": "test_green_1_f00061",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.02,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.30,
            "debug_crop_prefix": "crops/test_green_1_f00061_crop",
            "entity_selected_vs_gvm_mean_delta": 0.08,
        },
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]

    assert (
        input_entry["dominant_decision_reason"]
        == "sequence_gvm_retained_without_fallback_evaluation"
    )
    assert input_entry["recommended_action"] == "review subject recovery gain hotspots in priority debug crops"


def test_build_summary_payload_uses_actionable_fallback_for_generic_retained_case():
    module = _load_script_module()

    samples = [
        {
            "input": "test_frame_generic.jpg",
            "sample": "test_frame_generic",
            "selected_model": "gvm",
            "selected_source": "single_frame_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": -0.02,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.90,
            "debug_crop_prefix": "crops/test_frame_generic_crop_cool_gray_transition_effect_risk",
            "effect_selected_vs_gvm_mean_delta": -0.08,
            "debug_effect_risk": "cool_gray_transition_effect_risk",
        }
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]

    assert (
        input_entry["dominant_decision_reason"]
        == "single_frame_or_sequence_gvm_retained_cool_gray_transition_effect_risk"
    )
    assert input_entry["dominant_effect_risk"] == "cool_gray_transition_effect_risk"
    assert input_entry["recommended_action"] == "inspect cool gray transition effect-risk crops first"


def test_build_summary_payload_refines_sequence_retained_reason_with_bright_cool_effect_risk():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_sequence_risk.mp4",
            "sample": "test_green_sequence_risk_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.01,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.5,
            "debug_crop_prefix": "crops/test_green_sequence_risk_f00000_crop_bright_cool_effect_risk",
            "debug_effect_risk": "bright_cool_effect_risk",
        }
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]

    assert (
        input_entry["dominant_decision_reason"]
        == "sequence_gvm_retained_without_fallback_evaluation_bright_cool_effect_risk"
    )
    assert input_entry["dominant_effect_risk"] == "bright_cool_effect_risk"
    assert input_entry["recommended_action"] == "inspect bright cool effect-risk crops first"


def test_build_summary_payload_uses_actionable_text_for_low_weighted_score_block():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_score_blocked.mp4",
            "sample": "test_green_score_blocked_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": False,
            "fallback_score_blocked": True,
            "fallback_entity_delta": -0.230,
            "selected_vs_gvm_mean_delta": 0.01,
            "fallback_weighted_score": -0.01,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.90,
            "debug_crop_prefix": "crops/test_green_score_blocked_f00000_crop",
        },
        {
            "input": "test_green_score_blocked.mp4",
            "sample": "test_green_score_blocked_f00020",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": False,
            "fallback_score_blocked": True,
            "fallback_entity_delta": -0.120,
            "selected_vs_gvm_mean_delta": 0.02,
            "fallback_weighted_score": -0.02,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.30,
            "debug_crop_prefix": "crops/test_green_score_blocked_f00020_crop",
        },
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]
    sample_entry = next(
        entry for entry in payload["samples"] if entry["sample"] == "test_green_score_blocked_f00000"
    )

    assert input_entry["dominant_decision_reason"] == "fallback_blocked_by_low_weighted_score"
    assert (
        input_entry["recommended_action"]
        == "inspect subject recovery gaps in priority debug crops"
    )
    assert (
        sample_entry["debug_focus_reason"]
        == "fallback_entity_delta=-0.230 suggests fallback misses subject backbone recovery"
    )


def test_build_summary_payload_exposes_dominant_selected_source():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_mix.mp4",
            "sample": "test_green_mix_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.01,
            "fallback_weighted_score": None,
        },
        {
            "input": "test_green_mix.mp4",
            "sample": "test_green_mix_f00020",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.02,
            "fallback_weighted_score": None,
        },
        {
            "input": "test_green_mix.mp4",
            "sample": "test_green_mix_f00040",
            "selected_model": "corridorkey",
            "selected_source": "fallback_corridorkey",
            "fallback_applied": True,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": True,
            "fallback_effect_damage_blocked": False,
            "fallback_score_blocked": False,
            "selected_vs_gvm_mean_delta": 0.03,
            "fallback_weighted_score": 0.08,
        },
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]

    assert input_entry["selected_source_counts"] == {
        "sequence_gvm": 2,
        "fallback_corridorkey": 1,
    }
    assert input_entry["dominant_selected_source"] == "sequence_gvm"


def test_build_summary_payload_exposes_dominant_selected_model():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_mix.mp4",
            "sample": "test_green_mix_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.01,
            "fallback_weighted_score": None,
        },
        {
            "input": "test_green_mix.mp4",
            "sample": "test_green_mix_f00020",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.02,
            "fallback_weighted_score": None,
        },
        {
            "input": "test_green_mix.mp4",
            "sample": "test_green_mix_f00040",
            "selected_model": "corridorkey",
            "selected_source": "fallback_corridorkey",
            "fallback_applied": True,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": True,
            "fallback_effect_damage_blocked": False,
            "fallback_score_blocked": False,
            "selected_vs_gvm_mean_delta": 0.03,
            "fallback_weighted_score": 0.08,
        },
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]

    assert input_entry["selected_model_counts"] == {
        "gvm": 2,
        "corridorkey": 1,
    }
    assert input_entry["dominant_selected_model"] == "gvm"


def test_build_summary_payload_includes_priority_debug_crops_per_input():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_4.mp4",
            "sample": "test_green_4_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": False,
            "fallback_score_blocked": True,
            "selected_vs_gvm_mean_delta": 0.04,
            "fallback_weighted_score": -0.50,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.30,
            "debug_crop_prefix": "crops/test_green_4_f00000_crop",
        },
        {
            "input": "test_green_4.mp4",
            "sample": "test_green_4_f00045",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": True,
            "fallback_score_blocked": True,
            "selected_vs_gvm_mean_delta": 0.05,
            "fallback_weighted_score": -0.52,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.80,
            "debug_crop_prefix": "crops/test_green_4_f00045_crop",
        },
        {
            "input": "test_green_4.mp4",
            "sample": "test_green_4_f00089",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": True,
            "fallback_quality_gate_passed": False,
            "fallback_effect_damage_blocked": True,
            "fallback_score_blocked": True,
            "selected_vs_gvm_mean_delta": 0.03,
            "fallback_weighted_score": -0.48,
            "debug_crop_exported": False,
            "debug_crop_peak_abs_diff": None,
            "debug_crop_prefix": None,
        },
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]

    assert input_entry["priority_debug_crops"] == [
        {
            "sample": "test_green_4_f00045",
            "debug_crop_prefix": "crops/test_green_4_f00045_crop",
            "debug_crop_peak_abs_diff": 0.80,
            "top_debug_focus": "effect_damage_hotspot",
            "debug_effect_risk": None,
        },
        {
            "sample": "test_green_4_f00000",
            "debug_crop_prefix": "crops/test_green_4_f00000_crop",
            "debug_crop_peak_abs_diff": 0.30,
            "top_debug_focus": "subject_recovery_gap",
            "debug_effect_risk": None,
        },
    ]


def test_build_summary_payload_maps_general_priority_debug_crop_focus():
    module = _load_script_module()

    samples = [
        {
            "input": "test_frame_3.jpg",
            "sample": "test_frame_3",
            "selected_model": "gvm",
            "selected_source": "single_frame_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.0,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 1.0,
            "debug_crop_prefix": "crops/test_frame_3_crop",
        }
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]

    assert input_entry["priority_debug_crops"] == [
        {
            "sample": "test_frame_3",
            "debug_crop_prefix": "crops/test_frame_3_crop",
            "debug_crop_peak_abs_diff": 1.0,
            "top_debug_focus": "general_merge_delta_hotspot",
            "debug_effect_risk": None,
        }
    ]


def test_build_summary_payload_includes_debug_effect_risk_on_priority_crops():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_2.mp4",
            "sample": "test_green_2_f00060",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": -0.02,
            "effect_selected_vs_gvm_mean_delta": -0.08,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.9,
            "debug_crop_prefix": "crops/test_green_2_f00060_crop_bright_cool_effect_risk",
            "debug_effect_risk": "bright_cool_effect_risk",
        },
        {
            "input": "test_green_2.mp4",
            "sample": "test_green_2_f00120",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": -0.01,
            "effect_selected_vs_gvm_mean_delta": -0.05,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.7,
            "debug_crop_prefix": "crops/test_green_2_f00120_crop_cool_gray_transition_effect_risk",
            "debug_effect_risk": "cool_gray_transition_effect_risk",
        },
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]

    assert input_entry["dominant_effect_risk"] == "bright_cool_effect_risk"
    assert input_entry["priority_debug_crops"] == [
        {
            "sample": "test_green_2_f00060",
            "debug_crop_prefix": "crops/test_green_2_f00060_crop_bright_cool_effect_risk",
            "debug_crop_peak_abs_diff": 0.9,
            "top_debug_focus": "transparent_effect_instability_hotspot",
            "debug_effect_risk": "bright_cool_effect_risk",
        },
        {
            "sample": "test_green_2_f00120",
            "debug_crop_prefix": "crops/test_green_2_f00120_crop_cool_gray_transition_effect_risk",
            "debug_crop_peak_abs_diff": 0.7,
            "top_debug_focus": "transparent_effect_instability_hotspot",
            "debug_effect_risk": "cool_gray_transition_effect_risk",
        },
    ]


def test_infer_debug_effect_risk_marks_bright_cool_effect_crop():
    module = _load_script_module()

    sample = {
        "effect_selected_vs_gvm_mean_delta": -0.08,
    }
    crop_debug = {
        "frame_rgb": np.array(
            [
                [[90, 120, 150], [95, 125, 155], [100, 130, 160]],
                [[100, 130, 160], [173, 195, 227], [110, 140, 170]],
                [[110, 140, 170], [115, 145, 175], [120, 150, 180]],
            ],
            dtype=np.uint8,
        ),
    }

    assert (
        module._infer_debug_effect_risk(sample, crop_debug)
        == "bright_cool_effect_risk"
    )


def test_infer_debug_effect_risk_marks_cool_gray_transition_effect_crop():
    module = _load_script_module()

    sample = {
        "effect_selected_vs_gvm_mean_delta": -0.05,
    }
    crop_debug = {
        "frame_rgb": np.array(
            [
                [[140, 145, 180], [145, 150, 185], [150, 155, 190]],
                [[150, 155, 190], [184, 183, 225], [160, 165, 200]],
                [[160, 165, 200], [165, 170, 205], [170, 175, 210]],
            ],
            dtype=np.uint8,
        ),
    }

    assert (
        module._infer_debug_effect_risk(sample, crop_debug)
        == "cool_gray_transition_effect_risk"
    )


def test_resolve_known_effect_risk_crop_spec_returns_real_negative_samples():
    module = _load_script_module()

    bright_spec = module._resolve_known_effect_risk_crop_spec(
        "test_green_2.mp4",
        "test_green_2_f00060",
    )
    gray_spec = module._resolve_known_effect_risk_crop_spec(
        "test_green_2.mp4",
        "test_green_2_f00120",
    )

    assert bright_spec == {
        "top": 212,
        "left": 1128,
        "height": 5,
        "width": 5,
        "debug_effect_risk": "bright_cool_effect_risk",
    }
    assert gray_spec == {
        "top": 781,
        "left": 1085,
        "height": 5,
        "width": 5,
        "debug_effect_risk": "cool_gray_transition_effect_risk",
    }

def test_build_summary_payload_maps_subject_recovery_gain_priority_debug_crop_focus():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_subject_gain.mp4",
            "sample": "test_green_subject_gain_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": 0.04,
            "entity_selected_vs_gvm_mean_delta": 0.11,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.7,
            "debug_crop_prefix": "crops/test_green_subject_gain_f00000_crop",
        }
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]
    sample_entry = payload["samples"][0]

    assert input_entry["priority_debug_crops"] == [
        {
            "sample": "test_green_subject_gain_f00000",
            "debug_crop_prefix": "crops/test_green_subject_gain_f00000_crop",
            "debug_crop_peak_abs_diff": 0.7,
            "top_debug_focus": "subject_recovery_gain_hotspot",
            "debug_effect_risk": None,
        }
    ]
    assert (
        sample_entry["debug_focus_reason"]
        == "entity_selected_vs_gvm_mean_delta=0.110 suggests stronger subject recovery than baseline gvm"
    )


def test_build_summary_payload_maps_transparent_effect_instability_priority_debug_crop_focus():
    module = _load_script_module()

    samples = [
        {
            "input": "test_green_effect_instability.mp4",
            "sample": "test_green_effect_instability_f00000",
            "selected_model": "gvm",
            "selected_source": "sequence_gvm",
            "fallback_applied": False,
            "fallback_quality_evaluated": False,
            "fallback_quality_gate_passed": None,
            "fallback_effect_damage_blocked": None,
            "fallback_score_blocked": None,
            "selected_vs_gvm_mean_delta": -0.02,
            "effect_selected_vs_gvm_mean_delta": -0.08,
            "fallback_weighted_score": None,
            "debug_crop_exported": True,
            "debug_crop_peak_abs_diff": 0.65,
            "debug_crop_prefix": "crops/test_green_effect_instability_f00000_crop",
        }
    ]

    payload = module._build_summary_payload(samples)
    input_entry = payload["inputs"][0]
    sample_entry = payload["samples"][0]

    assert input_entry["priority_debug_crops"] == [
        {
            "sample": "test_green_effect_instability_f00000",
            "debug_crop_prefix": "crops/test_green_effect_instability_f00000_crop",
            "debug_crop_peak_abs_diff": 0.65,
            "top_debug_focus": "transparent_effect_instability_hotspot",
            "debug_effect_risk": None,
        }
    ]
    assert (
        sample_entry["debug_focus_reason"]
        == "effect_selected_vs_gvm_mean_delta=-0.080 suggests transparent effect instability"
    )
