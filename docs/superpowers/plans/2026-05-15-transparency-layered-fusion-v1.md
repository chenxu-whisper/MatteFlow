# Transparency Layered Fusion V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a V1 layered transparency pipeline that improves glow, particle, and soft-edge matting quality for both green-screen and black-background inputs without regressing solid subject quality.

**Architecture:** Keep the existing pipeline structure, but split alpha logic into `solid_alpha` and `effect_alpha` inside `hybrid_matte`, combine them with soft fusion, and then refine RGB using alpha-aware decontamination. Reuse the current pipeline stages and config surface so GUI behavior stays stable while internals become transparency-aware.

**Tech Stack:** Python, NumPy, OpenCV, pytest, existing MatteFlow pipeline classes

---

## File Structure

- Modify: `e:\ByteDance\Projects\Code\MatteFlow\src\matteflow\matte\hybrid_matte.py`
  - Add reusable soft-curve helpers
  - Add layered alpha builders for green-screen and black-background paths
  - Replace hard `np.maximum()`-style effect merge with soft fusion
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\src\matteflow\refine\color_decontaminate.py`
  - Add alpha-aware RGB repair for transparency-heavy areas
  - Keep solid-subject protection behavior intact
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\src\matteflow\temporal\temporal_stabilizer.py`
  - Add lightweight transparency-range stabilization so only semi-transparent regions are smoothed
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\src\matteflow\config.py`
  - Add safe internal-only fields if implementation needs them
- Create: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_transparency_layered_fusion.py`
  - Cover soft-curve behavior, layered fusion monotonicity, glow retention, and black-background effect retention
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_color_decontaminate.py`
  - Add transparency-aware RGB repair regression tests
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_logging_instrumentation.py`
  - Assert new transparency-layer logging is emitted

### Task 1: Lock Behavior With Failing Transparency Tests

**Files:**
- Create: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_transparency_layered_fusion.py`
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_color_decontaminate.py`
- Test: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_transparency_layered_fusion.py`

- [ ] **Step 1: Write the failing test for soft fusion helper**

```python
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig
from matteflow.matte.hybrid_matte import HybridMatte


def test_soft_fusion_adds_effect_only_into_remaining_alpha_space():
    matte = HybridMatte(MattingConfig(use_ai=False))

    solid = np.array([[0.90, 0.20]], dtype=np.float32)
    effect = np.array([[0.80, 0.60]], dtype=np.float32)

    fused = matte._soft_fuse_layers(solid, effect)

    assert np.isclose(fused[0, 0], 0.98, atol=1e-4)
    assert np.isclose(fused[0, 1], 0.68, atol=1e-4)
    assert np.all(fused >= solid)
    assert np.all(fused <= 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_transparency_layered_fusion.py::test_soft_fusion_adds_effect_only_into_remaining_alpha_space -q`
Expected: FAIL with `AttributeError` because `_soft_fuse_layers` does not exist yet

- [ ] **Step 3: Write the failing test for green-screen glow retention**

```python
def test_green_screen_effect_layer_preserves_soft_glow_without_forcing_full_opacity():
    matte = HybridMatte(MattingConfig(use_ai=False, transparency_preserve=0.8))

    frame = np.full((6, 6, 3), [20, 185, 55], dtype=np.uint8)
    frame[2:4, 2:4] = [245, 180, 220]

    base_alpha = np.full((6, 6), 0.10, dtype=np.float32)
    base_alpha[2:4, 2:4] = 0.55
    ai_alpha = np.zeros((6, 6), dtype=np.float32)

    fused = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert fused[2, 2] > 0.20
    assert fused[2, 2] < 0.95
    assert fused[0, 0] < 0.05
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_transparency_layered_fusion.py::test_green_screen_effect_layer_preserves_soft_glow_without_forcing_full_opacity -q`
Expected: FAIL because current merge logic produces values that do not satisfy the new layered-fusion expectations

- [ ] **Step 5: Write the failing test for black-background effect retention**

```python
def test_black_background_effect_layer_keeps_bright_particles():
    matte = HybridMatte(MattingConfig(use_ai=False))

    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    frame[2:4, 2:4] = [210, 170, 90]

    base_alpha = np.zeros((6, 6), dtype=np.float32)
    base_alpha[2:4, 2:4] = 0.18
    ai_alpha = np.zeros((6, 6), dtype=np.float32)

    fused = matte._merge_black_background_effects([base_alpha], [ai_alpha], [frame])[0]

    assert fused[2, 2] > 0.15
    assert fused[2, 2] < 0.80
    assert fused[0, 0] == 0.0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_transparency_layered_fusion.py::test_black_background_effect_layer_keeps_bright_particles -q`
Expected: FAIL with `AttributeError` because `_merge_black_background_effects` does not exist yet

- [ ] **Step 7: Write the failing test for transparency-aware RGB repair**

```python
import numpy as np

from matteflow.config import BackgroundMode, MattingConfig
from matteflow.refine.color_decontaminate import ColorDecontaminate


def test_green_transparency_rgb_repair_lifts_dark_glow_without_overwriting_solid_subject():
    frame = np.full((3, 3, 3), [20, 150, 35], dtype=np.uint8)
    frame[1, 1] = [55, 70, 65]
    alpha = np.full((3, 3), 0.12, dtype=np.float32)
    alpha[0, 0] = 0.98

    repaired = ColorDecontaminate(MattingConfig()).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    assert repaired[1, 1, 0] > frame[1, 1, 0]
    assert repaired[1, 1, 2] > frame[1, 1, 2]
    assert np.abs(int(repaired[0, 0, 1]) - int(frame[0, 0, 1])) < 10
```

- [ ] **Step 8: Run test to verify it fails**

Run: `uv run pytest tests/test_color_decontaminate.py::test_green_transparency_rgb_repair_lifts_dark_glow_without_overwriting_solid_subject -q`
Expected: FAIL because current repair path does not yet distinguish transparency-focused repair from solid-subject preservation strongly enough

- [ ] **Step 9: Commit**

```bash
git add tests/test_transparency_layered_fusion.py tests/test_color_decontaminate.py
git commit -m "test: add transparency layered fusion regressions"
```

### Task 2: Implement Layered Fusion In `hybrid_matte.py`

**Files:**
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\src\matteflow\matte\hybrid_matte.py`
- Test: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_transparency_layered_fusion.py`

- [ ] **Step 1: Add the minimal helper methods required by the new tests**

```python
def _smoothstep(self, x: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((x - low) / max(high - low, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _soft_fuse_layers(self, solid_alpha: np.ndarray, effect_alpha: np.ndarray) -> np.ndarray:
    solid = np.clip(solid_alpha.astype(np.float32, copy=False), 0.0, 1.0)
    effect = np.clip(effect_alpha.astype(np.float32, copy=False), 0.0, 1.0)
    return np.clip(solid + effect * (1.0 - solid), 0.0, 1.0)
```

- [ ] **Step 2: Replace green-screen effect merge with explicit solid/effect/fused layers**

```python
def _merge_green_screen_effects(self, base_alphas, ai_alphas, frames=None):
    preserve = float(np.clip(getattr(self.config, "transparency_preserve", 0.7), 0.0, 1.0))
    if preserve <= 0.0:
        return ai_alphas

    merged = []
    frame_iter = frames if frames is not None else [None] * len(base_alphas)
    for base_alpha, ai_alpha, frame in zip(base_alphas, ai_alphas, frame_iter):
        base_alpha = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        ai_alpha = np.clip(ai_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        solid_alpha = np.maximum(ai_alpha, self._green_screen_solid_layer(base_alpha, frame))
        effect_alpha = self._green_screen_effect_layer(base_alpha, frame) * preserve
        merged.append(self._soft_fuse_layers(solid_alpha, effect_alpha))
    return merged
```

- [ ] **Step 3: Add the internal green-screen layer builders**

```python
def _green_screen_effect_layer(self, base_alpha: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
    background_floor = float(np.percentile(base_alpha, 10))
    normalized = np.clip((base_alpha - (background_floor + 0.02)) / max(1.0 - (background_floor + 0.02), 1e-6), 0.0, 1.0)
    effect_alpha = self._smoothstep(normalized, 0.10, 0.75)
    if frame is not None:
        effect_alpha *= self._green_screen_effect_color_weight(frame)
    return np.clip(effect_alpha, 0.0, 1.0)


def _green_screen_solid_layer(self, base_alpha: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
    if frame is None:
        return np.where(base_alpha >= 0.90, base_alpha, 0.0)
    solid_color_mask = self._green_screen_solid_foreground_mask(frame)
    soft_subject_mask = self._green_screen_soft_subject_mask(frame)
    solid_mask = ((base_alpha >= 0.92) & solid_color_mask) | ((base_alpha >= 0.28) & soft_subject_mask)
    return np.where(solid_mask, base_alpha, 0.0)
```

- [ ] **Step 4: Add the black-background layered merge path and effect heuristics**

```python
def _merge_black_background_effects(self, base_alphas, ai_alphas, frames=None):
    merged = []
    frame_iter = frames if frames is not None else [None] * len(base_alphas)
    for base_alpha, ai_alpha, frame in zip(base_alphas, ai_alphas, frame_iter):
        base_alpha = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        ai_alpha = np.clip(ai_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        solid_alpha = np.maximum(ai_alpha, np.where(base_alpha > 0.75, base_alpha, 0.0))
        effect_alpha = self._black_background_effect_layer(base_alpha, frame)
        merged.append(self._soft_fuse_layers(solid_alpha, effect_alpha))
    return merged
```

- [ ] **Step 5: Wire black-background merge into `_black_background_matte()`**

```python
if self.rvm is not None and self.rvm.model is not None:
    ai_alphas = self.rvm.generate_sequence(frames, progress_callback)
    return self._merge_black_background_effects(base_alphas, ai_alphas, frames)
return base_alphas
```

- [ ] **Step 6: Run the transparency fusion tests**

Run: `uv run pytest tests/test_transparency_layered_fusion.py -q`
Expected: PASS for the new helper, green-screen layered fusion, and black-background layered fusion tests

- [ ] **Step 7: Commit**

```bash
git add src/matteflow/matte/hybrid_matte.py tests/test_transparency_layered_fusion.py
git commit -m "feat: add layered transparency fusion"
```

### Task 3: Implement Alpha-Aware RGB Repair In `color_decontaminate.py`

**Files:**
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\src\matteflow\refine\color_decontaminate.py`
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_color_decontaminate.py`
- Test: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_color_decontaminate.py`

- [ ] **Step 1: Extract a reusable transparency mask helper**

```python
def _transparency_band(self, alpha: np.ndarray, low: float = 0.02, high: float = 0.75) -> np.ndarray:
    return (alpha > low) & (alpha < high)
```

- [ ] **Step 2: Add green-screen transparency-aware repair after despill**

```python
transparency_mask = self._transparency_band(alpha, 0.02, 0.75)
pink_glow = transparency_mask & (r_corrected > g_corrected + 8) & (b_corrected > g_corrected)
white_glow = transparency_mask & (corrected_brightness > 150) & (np.abs(r_corrected - b_corrected) < 20)
dark_glow = transparency_mask & (corrected_brightness < 95) & (~white_mask)

lift = np.clip((118 - corrected_brightness) * 0.8, 0, 55)
r_corrected = np.where(dark_glow, r_corrected + lift * 1.10, r_corrected)
g_corrected = np.where(dark_glow, g_corrected + lift * 0.10, g_corrected)
b_corrected = np.where(dark_glow, b_corrected + lift * 1.00, b_corrected)

r_corrected = np.where(pink_glow, r_corrected * 1.02, r_corrected)
b_corrected = np.where(pink_glow, b_corrected * 1.02, b_corrected)
```

- [ ] **Step 3: Add black-background transparency-aware color protection**

```python
transparency_mask = self._transparency_band(alpha, 0.02, 0.75)
particle_mask = transparency_mask & (brightness < 70) & (color_range > 10)
gray_haze_mask = transparency_mask & (color_range < 18) & (brightness > 10)

s = np.where(gray_haze_mask, np.clip(s * (1.0 + self.config.black_contrast_restore * 0.7), 0, 255), s)
v = np.where(transparency_mask, np.clip(v * 1.05, 0, 255), v)

for c in range(3):
    result[:, :, c] = np.where(
        particle_mask,
        result[:, :, c] * 0.75 + original[:, :, c] * 0.25,
        result[:, :, c],
    )
```

- [ ] **Step 4: Run the focused color repair tests**

Run: `uv run pytest tests/test_color_decontaminate.py -q`
Expected: PASS including the new transparency-aware repair regression

- [ ] **Step 5: Commit**

```bash
git add src/matteflow/refine/color_decontaminate.py tests/test_color_decontaminate.py
git commit -m "feat: add alpha-aware transparency rgb repair"
```

### Task 4: Add Transparency-Band Temporal Stabilization And Final Verification

**Files:**
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\src\matteflow\temporal\temporal_stabilizer.py`
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\src\matteflow\config.py`
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_logging_instrumentation.py`
- Modify: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_transparency_layered_fusion.py`
- Test: `e:\ByteDance\Projects\Code\MatteFlow\tests\test_logging_instrumentation.py`

- [ ] **Step 1: Add safe internal config defaults if needed**

```python
transparency_temporal_low: float = 0.03
transparency_temporal_high: float = 0.75
transparency_temporal_blend: float = 0.20
```

- [ ] **Step 2: Update temporal stabilizer to smooth only semi-transparent pixels**

```python
mask = (current_alpha > self.config.transparency_temporal_low) & (
    current_alpha < self.config.transparency_temporal_high
)
blended = current_alpha * (1.0 - self.config.transparency_temporal_blend) + prev_alpha * self.config.transparency_temporal_blend
current_alpha = np.where(mask, blended, current_alpha)
```

- [ ] **Step 3: Add logging for layered transparency statistics**

```python
logger.info(
    "Transparency fusion stats: solid_mean=%.4f effect_mean=%.4f fused_mean=%.4f",
    float(solid_alpha.mean()),
    float(effect_alpha.mean()),
    float(fused_alpha.mean()),
)
```

- [ ] **Step 4: Add or update tests for transparency-band stabilization and new logging**

```python
def test_transparency_temporal_stabilizer_only_blends_mid_alpha_range():
    ...


def test_logging_reports_transparency_fusion_stats(caplog):
    ...
```

- [ ] **Step 5: Run the full regression suite**

Run: `uv run pytest tests/test_transparency_layered_fusion.py tests/test_color_decontaminate.py tests/test_logging_instrumentation.py tests/test_auto_params.py tests/test_web_gui_defaults.py tests/test_web_gui_preview.py tests/test_green_screen_matte.py -q`
Expected: PASS with zero failures

- [ ] **Step 6: Run diagnostics on modified files**

Run diagnostic checks for:
- `src/matteflow/matte/hybrid_matte.py`
- `src/matteflow/refine/color_decontaminate.py`
- `src/matteflow/temporal/temporal_stabilizer.py`
- `src/matteflow/config.py`
- `tests/test_transparency_layered_fusion.py`
- `tests/test_color_decontaminate.py`
- `tests/test_logging_instrumentation.py`

Expected: no new diagnostics

- [ ] **Step 7: Restart GUI and verify it serves successfully**

Run: `powershell -ExecutionPolicy Bypass -File .\scripts\start_gui.ps1`
Expected: GUI starts on `http://localhost:7860`

Run: `(Invoke-WebRequest -Uri http://localhost:7860/ -UseBasicParsing).StatusCode`
Expected: `200`

- [ ] **Step 8: Commit**

```bash
git add src/matteflow/temporal/temporal_stabilizer.py src/matteflow/config.py tests/test_logging_instrumentation.py tests/test_transparency_layered_fusion.py
git commit -m "feat: stabilize transparency layers over time"
```

## Self-Review

- Spec coverage:
  - Layered alpha split: covered in Task 2
  - Soft fusion: covered in Task 2
  - Alpha-aware RGB repair: covered in Task 3
  - Transparency-only temporal stabilization: covered in Task 4
  - Logging and verification: covered in Task 4
- Placeholder scan:
  - No `TODO`, `TBD`, or implied “fill this in later” language remains
- Type consistency:
  - `_soft_fuse_layers`, `_green_screen_effect_layer`, `_green_screen_solid_layer`, and `_merge_black_background_effects` are named consistently across tasks
