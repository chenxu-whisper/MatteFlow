# Heart ROI Effect Alpha Green Haze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ROI-level regression coverage for the hand-heart glow region and suppress green outer haze during `effect_alpha` generation without regressing solid subject preservation.

**Architecture:** Extend `HybridMatte` in two focused steps. First, add failing tests that cover both synthetic green-haze suppression and a real-image ROI regression using `test_frame_2.png`. Then add a color-derived suppression mask inside `_green_screen_effect_layer()` so green outer haze is attenuated before soft fusion, while pink/white glow structure still survives through the existing color weighting and preserve blend.

**Tech Stack:** Python, `numpy`, `pytest`, `PIL.Image`, existing `HybridMatte` alpha fusion pipeline

---

### Task 1: Add failing ROI and effect-alpha tests

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_transparency_layered_fusion.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_transparency_layered_fusion.py`

- [ ] **Step 1: Write the failing synthetic test for effect-alpha suppression**

```python
def test_green_screen_effect_layer_suppresses_green_outer_haze_but_keeps_pink_glow():
    matte = HybridMatte(MattingConfig(use_ai=False))

    frame = np.array(
        [
            [[245, 180, 220], [220, 225, 200]],
        ],
        dtype=np.uint8,
    )
    base_alpha = np.array([[0.45, 0.45]], dtype=np.float32)

    effect_alpha = matte._green_screen_effect_layer(base_alpha, frame)

    assert effect_alpha[0, 0] > 0.15
    assert effect_alpha[0, 1] < effect_alpha[0, 0] * 0.45
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/test_transparency_layered_fusion.py::test_green_screen_effect_layer_suppresses_green_outer_haze_but_keeps_pink_glow -v`
Expected: FAIL because both pixels currently receive similar effect alpha after only color weighting.

- [ ] **Step 3: Write the failing real-image ROI regression**

```python
def test_green_screen_effect_layer_heart_roi_suppresses_outer_green_haze_on_test_frame():
    matte = HybridMatte(MattingConfig(use_ai=False))

    frame = np.array(Image.open(PROJECT_ROOT / "assets" / "frame" / "test_frame_2.png").convert("RGB"))
    roi = frame[380:460, 340:400]
    base_alpha = np.full(roi.shape[:2], 0.42, dtype=np.float32)

    effect_alpha = matte._green_screen_effect_layer(base_alpha, roi)

    pink_glow_alpha = float(effect_alpha[74, 20])
    outer_haze_alpha = float(effect_alpha[58, 43])

    assert pink_glow_alpha > 0.12
    assert outer_haze_alpha < pink_glow_alpha * 0.55
```

- [ ] **Step 4: Run the targeted ROI regression to verify it fails**

Run: `pytest tests/test_transparency_layered_fusion.py::test_green_screen_effect_layer_heart_roi_suppresses_outer_green_haze_on_test_frame -v`
Expected: FAIL because the outer haze still retains too much effect alpha relative to the pink glow sample.

- [ ] **Step 5: Commit the test-only red state**

```bash
git add tests/test_transparency_layered_fusion.py
git commit -m "test: lock heart roi green haze regression"
```

### Task 2: Add effect-alpha green haze suppression

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\matte\hybrid_matte.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_transparency_layered_fusion.py`

- [ ] **Step 1: Add a dedicated suppression helper used only by `effect_alpha`**

```python
def _green_screen_effect_haze_suppression(self, frame: np.ndarray) -> np.ndarray:
    frame_f = frame.astype(np.float32, copy=False)
    r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
    brightness = (r + g + b) / 3.0
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    outer_green_haze = np.minimum.reduce(
        [
            np.clip((g - b - 10.0) / 18.0, 0.0, 1.0),
            np.clip((g - r + 2.0) / 14.0, 0.0, 1.0),
            np.clip((brightness - 160.0) / 45.0, 0.0, 1.0),
            np.clip((90.0 - chroma) / 90.0, 0.0, 1.0),
        ]
    )
    return 1.0 - 0.75 * outer_green_haze
```

- [ ] **Step 2: Apply the suppression inside `_green_screen_effect_layer()` after the soft curve**

```python
def _green_screen_effect_layer(self, base_alpha: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
    background_floor = float(np.percentile(base_alpha, 10))
    effect_floor = min(background_floor + 0.02, 0.95)
    normalized = np.clip((base_alpha - effect_floor) / max(1.0 - effect_floor, 1e-6), 0.0, 1.0)
    effect_alpha = self._smoothstep(normalized, 0.08, 0.75)
    if frame is not None:
        effect_alpha = effect_alpha * self._green_screen_effect_haze_suppression(frame)
        effect_alpha = effect_alpha * self._green_screen_effect_color_weight(frame)
    return np.clip(effect_alpha, 0.0, 1.0)
```

- [ ] **Step 3: Run both targeted tests to verify they pass**

Run: `pytest tests/test_transparency_layered_fusion.py::test_green_screen_effect_layer_suppresses_green_outer_haze_but_keeps_pink_glow tests/test_transparency_layered_fusion.py::test_green_screen_effect_layer_heart_roi_suppresses_outer_green_haze_on_test_frame -v`
Expected: PASS

- [ ] **Step 4: Commit the minimal implementation**

```bash
git add src/matteflow/matte/hybrid_matte.py tests/test_transparency_layered_fusion.py
git commit -m "fix: suppress green haze in effect alpha"
```

### Task 3: Run regression, self-check, and sample validation

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\matte\hybrid_matte.py` if thresholds need tiny follow-up adjustment
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_transparency_layered_fusion.py`

- [ ] **Step 1: Run focused regression coverage**

Run: `pytest tests/test_transparency_layered_fusion.py tests/test_color_decontaminate.py tests/test_logging_instrumentation.py -q`
Expected: all targeted transparency and color regressions PASS

- [ ] **Step 2: Run a small ROI inspection script against `test_frame_2.png`**

```python
from pathlib import Path
import numpy as np
from PIL import Image
from matteflow.config import MattingConfig
from matteflow.matte.hybrid_matte import HybridMatte

root = Path(__file__).resolve().parent
frame = np.array(Image.open(root / "assets" / "frame" / "test_frame_2.png").convert("RGB"))
roi = frame[380:460, 340:400]
alpha = np.full(roi.shape[:2], 0.42, dtype=np.float32)
matte = HybridMatte(MattingConfig(use_ai=False))
effect_alpha = matte._green_screen_effect_layer(alpha, roi)
print("pink_glow", float(effect_alpha[74, 20]))
print("outer_haze", float(effect_alpha[58, 43]))
```

- [ ] **Step 3: Check diagnostics for edited files**

Run diagnostics for:
- `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\matte\hybrid_matte.py`
- `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_transparency_layered_fusion.py`

Expected: no new errors

- [ ] **Step 4: Commit verification-only threshold tweaks if needed**

```bash
git add src/matteflow/matte/hybrid_matte.py tests/test_transparency_layered_fusion.py
git commit -m "test: verify heart roi effect alpha regression"
```
