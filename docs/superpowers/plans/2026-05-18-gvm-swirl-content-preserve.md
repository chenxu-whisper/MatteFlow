# GVM Swirl Content Preserve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `GVM` matte output on `test_frame_1.png` by preserving semi-transparent swirl content inside the ring while avoiding solid expansion into the green background.

**Architecture:** Add a small post-process step inside `GVMMatte` that only repairs low-to-mid alpha holes when the pixel color matches swirl-like blue/purple or bright mixed-white content and when nearby alpha already provides foreground support. Lock the behavior with focused unit tests that prove the repair fills internal holes but does not lift pure green-screen background.

**Tech Stack:** Python, `numpy`, `cv2`, `pytest`, existing `GVMMatte` pipeline

---

### Task 1: Add the failing tests for GVM swirl preservation

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_gvm_alpha_resize.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_gvm_alpha_resize.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_gvm_preserve_internal_swirl_content_repairs_low_alpha_holes():
    matte = GVMMatte.__new__(GVMMatte)
    frame = np.full((9, 9, 3), [20, 200, 20], dtype=np.uint8)
    frame[2:7, 2:7] = [170, 135, 240]
    alpha = np.zeros((9, 9), dtype=np.float32)
    alpha[2:7, 2:7] = 0.52
    alpha[4, 4] = 0.08

    repaired = matte._preserve_internal_swirl_content(frame, alpha)

    assert repaired[4, 4] > alpha[4, 4]
    assert repaired[4, 4] <= 0.55


def test_gvm_preserve_internal_swirl_content_keeps_pure_green_background_flat():
    matte = GVMMatte.__new__(GVMMatte)
    frame = np.full((9, 9, 3), [20, 200, 20], dtype=np.uint8)
    alpha = np.zeros((9, 9), dtype=np.float32)

    repaired = matte._preserve_internal_swirl_content(frame, alpha)

    assert np.allclose(repaired, alpha)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gvm_alpha_resize.py::test_gvm_preserve_internal_swirl_content_repairs_low_alpha_holes tests/test_gvm_alpha_resize.py::test_gvm_preserve_internal_swirl_content_keeps_pure_green_background_flat -v`
Expected: FAIL with `AttributeError` because `_preserve_internal_swirl_content` does not exist yet.

### Task 2: Implement the minimal GVM content-preserve post-process

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\matte\gvm_matte.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_gvm_alpha_resize.py`

- [ ] **Step 1: Add the helper method**

```python
def _preserve_internal_swirl_content(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
    frame_f = frame.astype(np.float32, copy=False)

    r = frame_f[:, :, 0]
    g = frame_f[:, :, 1]
    b = frame_f[:, :, 2]
    brightness = (r + g + b) / 3.0
    chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    swirl_color = (
        (b > g + 8.0)
        | ((r > g + 8.0) & (b > g + 4.0))
        | ((brightness > 150.0) & (chroma < 70.0) & (g < 205.0))
    )
    weak_alpha = (alpha_f > 0.03) & (alpha_f < 0.42)

    alpha_u8 = np.clip(alpha_f * 255.0, 0.0, 255.0).astype(np.uint8)
    local_support = cv2.dilate(alpha_u8, np.ones((9, 9), np.uint8), iterations=1).astype(np.float32) / 255.0
    support_mask = local_support > 0.45

    recovered = np.clip(local_support * 0.58, 0.0, 0.55)
    repaired = np.where(swirl_color & weak_alpha & support_mask, np.maximum(alpha_f, recovered), alpha_f)
    return np.clip(repaired, 0.0, 1.0)
```

- [ ] **Step 2: Wire the helper into `_run_sequence_inference()`**

```python
alpha_f = alpha.astype(np.float32) / 255.0
alpha_f = self._preserve_internal_swirl_content(frame, alpha_f)
alphas.append(alpha_f)
```

- [ ] **Step 3: Run the targeted tests and verify they pass**

Run: `pytest tests/test_gvm_alpha_resize.py::test_gvm_preserve_internal_swirl_content_repairs_low_alpha_holes tests/test_gvm_alpha_resize.py::test_gvm_preserve_internal_swirl_content_keeps_pure_green_background_flat -v`
Expected: PASS

### Task 3: Re-run focused verification and produce a new GVM output

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\matte\gvm_matte.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_gvm_alpha_resize.py`

- [ ] **Step 1: Run the focused GVM test file**

Run: `pytest tests/test_gvm_alpha_resize.py -q`
Expected: PASS

- [ ] **Step 2: Check diagnostics for the edited files**

Run diagnostics for:
- `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\matte\gvm_matte.py`
- `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_gvm_alpha_resize.py`

Expected: no new errors introduced.

- [ ] **Step 3: Re-run `test_frame_1.png` with `GVM` and write a new output folder**

Run a one-off script that builds:

```python
config = MattingConfig(
    background_mode=BackgroundMode.GREEN_SCREEN,
    quality_mode=QualityMode.STANDARD,
    use_ai=True,
    ai_model="gvm",
)
MattingPipeline(config).process(input_path, output_dir)
```

Expected: a new `Processed/processed_000000.png` and `Matte/matte_000000.png` under a fresh `temp/output/...` folder for visual comparison.
