# GVM-Aware Despeckle Swirl Protection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tighten `despeckle` so it only preserves valid low-alpha swirl content when the pipeline is actually using `GVM`, including `auto` mode selecting `GVM`.

**Architecture:** Extend `MattingPipeline` to pass `frames` plus a small `context` payload into `Despeckle.process()`. Inside `Despeckle`, keep the existing median-filter cleanup, then restore crushed soft alpha only when both alpha support and a GVM swirl-color mask agree. Lock the behavior with focused tests that prove swirl pixels survive while non-swirl soft islands do not get the same protection.

**Tech Stack:** Python, `numpy`, `cv2`, `pytest`, MatteFlow `MattingPipeline`, `Despeckle`, `GVMMatte`

---

### Task 1: Add failing tests for GVM-aware swirl-only protection

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_despeckle_soft_alpha.py`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_logging_instrumentation.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_despeckle_soft_alpha.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_despeckle_gvm_context_keeps_supported_swirl_soft_alpha():
    config = MattingConfig()
    config.despeckle_enable = True
    config.despeckle_radius = 5
    config.despeckle_threshold = 0.0

    alpha = np.zeros((15, 15), dtype=np.float32)
    alpha[4:11, 4:11] = 0.6
    alpha[6:9, 6:9] = 0.2
    frame = np.full((15, 15, 3), [20, 200, 20], dtype=np.uint8)
    frame[4:11, 4:11] = [170, 135, 240]

    cleaned = Despeckle(config).process(
        [alpha],
        frames=[frame],
        context={"active_ai_model": "gvm"},
    )[0]

    assert cleaned[7, 7] >= 0.18


def test_despeckle_gvm_context_does_not_keep_non_swirl_soft_alpha():
    config = MattingConfig()
    config.despeckle_enable = True
    config.despeckle_radius = 5
    config.despeckle_threshold = 0.0

    alpha = np.zeros((15, 15), dtype=np.float32)
    alpha[4:11, 4:11] = 0.6
    alpha[6:9, 6:9] = 0.2
    frame = np.full((15, 15, 3), [20, 200, 20], dtype=np.uint8)
    frame[4:11, 4:11] = [150, 150, 150]

    cleaned = Despeckle(config).process(
        [alpha],
        frames=[frame],
        context={"active_ai_model": "gvm"},
    )[0]

    assert cleaned[7, 7] <= 0.03
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `python -m pytest tests/test_despeckle_soft_alpha.py -q`
Expected: FAIL because `Despeckle.process()` does not yet accept `frames/context` and cannot distinguish swirl from non-swirl protection.

### Task 2: Implement GVM-aware swirl-only restoration

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\refine\despeckle.py`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\pipeline.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_despeckle_soft_alpha.py`

- [ ] **Step 1: Extend `Despeckle.process()` to accept optional `frames` and `context`**

```python
def process(self, alphas, frames=None, context=None):
    ...
    for index, alpha in enumerate(alphas):
        frame = frames[index] if frames is not None else None
        cleaned_alpha = self._despeckle_single(alpha, frame=frame, context=context or {})
```

- [ ] **Step 2: Add GVM swirl-mask helper and use it only for active GVM runs**

```python
def _is_gvm_active(self, context: dict) -> bool:
    return (context or {}).get("active_ai_model") == "gvm"


def _swirl_color_mask(self, frame: np.ndarray) -> np.ndarray:
    frame_f = frame.astype(np.float32, copy=False)
    r = frame_f[:, :, 0]
    g = frame_f[:, :, 1]
    b = frame_f[:, :, 2]
    brightness = (r + g + b) / 3.0
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    return (
        (b > g + 8.0)
        | ((r > g + 8.0) & (b > g + 4.0))
        | ((brightness > 150.0) & (chroma < 70.0) & (g < 205.0))
    )
```

- [ ] **Step 3: Tighten restoration to require both alpha support and swirl-color support**

```python
if self._is_gvm_active(context) and frame is not None:
    swirl_mask = self._swirl_color_mask(frame)
    restored = np.where(crushed_soft & support_mask & swirl_mask, original, cleaned)
else:
    restored = np.where(crushed_soft & support_mask, original, cleaned)
```

- [ ] **Step 4: Pass the active AI model into `despeckle` from the pipeline**

```python
despeckle_context = {
    "active_ai_model": self._resolve_active_ai_model(bg_mode),
}
alphas = self.despeckle.process(alphas, frames=frames, context=despeckle_context)
```

- [ ] **Step 5: Run focused tests and verify they pass**

Run: `python -m pytest tests/test_despeckle_soft_alpha.py tests/test_logging_instrumentation.py -q`
Expected: PASS

### Task 3: Re-run real GVM verification on `test_frame_1.png`

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\pipeline.py`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\refine\despeckle.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_gvm_alpha_resize.py`

- [ ] **Step 1: Run the related regression set**

Run: `python -m pytest tests/test_despeckle_soft_alpha.py tests/test_gvm_alpha_resize.py tests/test_logging_instrumentation.py -q`
Expected: PASS

- [ ] **Step 2: Check diagnostics for edited files**

Check:
- `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\refine\despeckle.py`
- `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\pipeline.py`
- `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_despeckle_soft_alpha.py`

Expected: no new diagnostics.

- [ ] **Step 3: Re-run `test_frame_1.png` with real GVM and compare stage logs**

Run a one-off script with:

```python
config = MattingConfig()
config.background_mode = BackgroundMode.GREEN_SCREEN
config.quality_mode = QualityMode.STANDARD
config.ai_model = "gvm"
config.use_ai = True
config.ai_enhance = False
MattingPipeline(config).process(input_path, output_dir)
```

Expected: `Alpha stage delta: stage=despeckle ... suppressed_to_near_zero=...` is lower than the previous `720`, while `refine` remains near zero suppression.
