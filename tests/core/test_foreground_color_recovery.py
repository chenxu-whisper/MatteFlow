import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.analysis.region_ownership import RegionOwnership  # noqa: E402
from matteflow.config import BackgroundMode, MattingConfig  # noqa: E402
from matteflow.refine.color_decontaminate import ColorDecontaminate  # noqa: E402
from matteflow.refine.foreground_color_recovery import ForegroundColorRecovery  # noqa: E402


def _blank_ownership(shape, **overrides):
    values = {
        "subject": np.zeros(shape, dtype=bool),
        "hair_edge": np.zeros(shape, dtype=bool),
        "luminous_prop": np.zeros(shape, dtype=bool),
        "transparent_effect": np.zeros(shape, dtype=bool),
        "background_residue": np.zeros(shape, dtype=bool),
        "uncertain_edge": np.zeros(shape, dtype=bool),
    }
    values.update(overrides)
    return RegionOwnership(**values)


def test_foreground_color_recovery_unmixes_transparent_green_screen_spill():
    screen = np.array([0.0, 210.0, 40.0], dtype=np.float32)
    foreground = np.array([230.0, 145.0, 80.0], dtype=np.float32)
    alpha_value = 0.50
    observed = np.round(alpha_value * foreground + (1.0 - alpha_value) * screen).astype(np.uint8)

    frame = np.tile(screen.astype(np.uint8), (12, 12, 1))
    alpha = np.zeros((12, 12), dtype=np.float32)
    effect_mask = np.zeros((12, 12), dtype=bool)
    effect_mask[4:8, 4:8] = True
    frame[effect_mask] = observed
    alpha[effect_mask] = alpha_value
    ownership = _blank_ownership(alpha.shape, transparent_effect=effect_mask)

    recovered = ForegroundColorRecovery(MattingConfig()).recover(frame, alpha, ownership=ownership)
    recovered_region = recovered.astype(np.float32)[effect_mask]

    assert float(np.abs(recovered_region[:, 0] - foreground[0]).mean()) <= 8.0
    assert float(np.abs(recovered_region[:, 1] - foreground[1]).mean()) <= 8.0
    assert float(np.abs(recovered_region[:, 2] - foreground[2]).mean()) <= 8.0


def test_foreground_color_recovery_limits_changes_to_owned_regions():
    frame = np.full((10, 10, 3), [0, 210, 40], dtype=np.uint8)
    alpha = np.zeros((10, 10), dtype=np.float32)
    owned_mask = np.zeros((10, 10), dtype=bool)
    owned_mask[2:5, 2:5] = True
    unowned_mask = np.zeros((10, 10), dtype=bool)
    unowned_mask[6:9, 6:9] = True
    frame[owned_mask | unowned_mask] = [95, 150, 60]
    alpha[owned_mask | unowned_mask] = 0.55
    ownership = _blank_ownership(alpha.shape, uncertain_edge=owned_mask)

    recovered = ForegroundColorRecovery(MattingConfig()).recover(frame, alpha, ownership=ownership)

    assert float(recovered.astype(np.float32)[owned_mask, 0].mean()) > float(frame[owned_mask, 0].mean()) + 40.0
    assert np.array_equal(recovered[unowned_mask], frame[unowned_mask])


def test_foreground_color_recovery_sequence_reuses_stable_screen_color():
    screen = np.array([0.0, 210.0, 40.0], dtype=np.float32)
    foreground = np.array([220.0, 130.0, 70.0], dtype=np.float32)
    alpha_value = 0.50
    observed = np.round(alpha_value * foreground + (1.0 - alpha_value) * screen).astype(np.uint8)

    first = np.tile(screen.astype(np.uint8), (12, 12, 1))
    second = np.tile(observed, (12, 12, 1))
    alpha_first = np.zeros((12, 12), dtype=np.float32)
    alpha_second = np.full((12, 12), alpha_value, dtype=np.float32)
    effect_mask = np.zeros((12, 12), dtype=bool)
    effect_mask[4:8, 4:8] = True
    first[effect_mask] = observed
    alpha_first[effect_mask] = alpha_value
    ownerships = [
        _blank_ownership(alpha_first.shape, transparent_effect=effect_mask),
        _blank_ownership(alpha_second.shape, transparent_effect=np.ones_like(effect_mask)),
    ]

    recovery = ForegroundColorRecovery(MattingConfig())
    recovered_single = recovery.recover(second, alpha_second, ownership=ownerships[1])
    recovered_sequence = recovery.recover_sequence(
        [first, second],
        [alpha_first, alpha_second],
        ownerships=ownerships,
    )

    single_error = np.abs(recovered_single.astype(np.float32) - foreground).mean()
    sequence_error = np.abs(recovered_sequence[1].astype(np.float32) - foreground).mean()
    assert sequence_error + 20.0 < single_error
    assert recovery.last_sequence_diagnostics["screen_rgb"] == [0.0, 210.0, 40.0]


def test_foreground_color_recovery_quality_gate_rejects_destructive_low_alpha_unmix():
    frame = np.full((12, 12, 3), [0, 210, 40], dtype=np.uint8)
    alpha = np.zeros((12, 12), dtype=np.float32)
    effect_mask = np.zeros((12, 12), dtype=bool)
    effect_mask[4:8, 4:8] = True
    frame[effect_mask] = [8, 170, 35]
    alpha[effect_mask] = 0.12
    ownership = _blank_ownership(alpha.shape, transparent_effect=effect_mask)

    recovery = ForegroundColorRecovery(MattingConfig())
    recovered = recovery.recover(frame, alpha, ownership=ownership)

    assert np.array_equal(recovered[effect_mask], frame[effect_mask])
    assert recovery.last_diagnostics["rejected_pixels"] >= int(effect_mask.sum())


def test_color_decontaminate_runs_foreground_color_recovery_with_shared_ownership():
    frame = np.full((12, 12, 3), [0, 210, 40], dtype=np.uint8)
    alpha = np.zeros((12, 12), dtype=np.float32)
    prop_mask = np.zeros((12, 12), dtype=bool)
    prop_mask[4:8, 4:8] = True
    frame[prop_mask] = [95, 150, 60]
    alpha[prop_mask] = 0.55
    ownership = _blank_ownership(alpha.shape, transparent_effect=prop_mask)

    result = ColorDecontaminate(MattingConfig()).process(
        [frame],
        [alpha],
        BackgroundMode.GREEN_SCREEN,
        context={"region_ownership": [ownership]},
    )[0]

    assert float(result.astype(np.float32)[prop_mask, 0].mean()) > 155.0


def test_color_decontaminate_records_foreground_recovery_diagnostics_in_context():
    frame = np.full((12, 12, 3), [0, 210, 40], dtype=np.uint8)
    alpha = np.zeros((12, 12), dtype=np.float32)
    effect_mask = np.zeros((12, 12), dtype=bool)
    effect_mask[4:8, 4:8] = True
    frame[effect_mask] = [35, 190, 55]
    alpha[effect_mask] = 0.55
    ownership = _blank_ownership(alpha.shape, transparent_effect=effect_mask)
    context = {"region_ownership": [ownership]}

    ColorDecontaminate(MattingConfig()).process(
        [frame],
        [alpha],
        BackgroundMode.GREEN_SCREEN,
        context=context,
    )

    diagnostics = context["foreground_recovery"]
    assert diagnostics["frames"] == 1
    assert diagnostics["attempted_pixels"] >= 0
    assert diagnostics["accepted_pixels"] >= 0
    assert diagnostics["rejected_pixels"] >= 0
    assert diagnostics["screen_rgb"] == [0.0, 210.0, 40.0]


def test_color_decontaminate_skips_foreground_color_recovery_for_gvm_context():
    frame = np.full((12, 12, 3), [0, 210, 40], dtype=np.uint8)
    alpha = np.zeros((12, 12), dtype=np.float32)
    effect_mask = np.zeros((12, 12), dtype=bool)
    effect_mask[4:8, 4:8] = True
    frame[effect_mask] = [95, 150, 60]
    alpha[effect_mask] = 0.55
    ownership = _blank_ownership(alpha.shape, transparent_effect=effect_mask)
    decontaminate = ColorDecontaminate(MattingConfig())

    class RaisingRecovery:
        def recover(self, frame_arg, alpha_arg, ownership=None):
            raise AssertionError("foreground recovery should be skipped for GVM context")

    decontaminate._foreground_recovery = RaisingRecovery()

    result = decontaminate.process(
        [frame],
        [alpha],
        BackgroundMode.GREEN_SCREEN,
        context={"region_ownership": [ownership], "active_ai_model": "gvm"},
    )[0]

    assert result.shape == frame.shape


def test_color_decontaminate_removes_high_alpha_teal_green_cast():
    frame = np.full((12, 12, 3), [0, 210, 40], dtype=np.uint8)
    alpha = np.zeros((12, 12), dtype=np.float32)
    cast_mask = np.zeros((12, 12), dtype=bool)
    cast_mask[4:8, 4:8] = True
    frame[cast_mask] = [62, 152, 138]
    alpha[cast_mask] = 0.956

    result = ColorDecontaminate(MattingConfig()).process(
        [frame],
        [alpha],
        BackgroundMode.GREEN_SCREEN,
        context={"active_ai_model": "gvm"},
    )[0].astype(np.float32)

    assert float((result[:, :, 1] - result[:, :, 0])[cast_mask].mean()) <= 8.0
