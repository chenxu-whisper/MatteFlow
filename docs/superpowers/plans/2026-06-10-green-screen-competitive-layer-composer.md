# Green Screen Competitive Layer Composer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GVM green-screen competitive composer that assigns pixels to subject, effect, or background ownership before composing final alpha.

**Architecture:** Add a focused `green_screen_layer_composer.py` module with dataclasses for candidates, ownership, and results. `HybridMatte._merge_green_screen_effects()` remains the public integration point and calls the composer only for GVM green-screen fusion, with the old soft-fuse path kept as fallback.

**Tech Stack:** Python, NumPy, OpenCV, pytest, existing `HybridMatte`, existing `MattingConfig`, existing real asset `assets/frame/test_frame_3.jpg`.

---

## File Structure

- Create: `src/matteflow/matte/green_screen_layer_composer.py`
  - Owns `LayerCandidate`, `LayerOwnership`, `CompetitiveLayerResult`, and `GreenScreenCompetitiveLayerComposer`.
  - Contains all candidate/evidence/ownership/composition logic.

- Modify: `src/matteflow/matte/hybrid_matte.py`
  - Imports the composer.
  - Builds existing subject/effect layer inputs.
  - Calls the composer for `last_active_ai_model == "gvm"`.
  - Stores `last_green_screen_layer_debug` for tests and diagnostics.
  - Keeps old `_soft_fuse_layers()` behavior for non-GVM and fallback.

- Create: `tests/core/test_green_screen_layer_composer.py`
  - Unit tests for ownership competition and debug layer shape.
  - Uses small synthetic arrays so root-cause behavior is isolated.

- Modify: `tests/core/test_green_screen_fusion.py`
  - Adds real `test_frame_3.jpg` integration coverage.
  - Verifies subject/effect/background ownership behavior through `HybridMatte`.

- Optional Modify: `.superpowers/diagnostics/export_semantic_layers.py`
  - If kept in workspace, update it to include composer ownership layers.
  - Do not make this required for production code or tests.

## Task 1: Add Composer Dataclasses And Debug Shape Contract

**Files:**
- Create: `src/matteflow/matte/green_screen_layer_composer.py`
- Create: `tests/core/test_green_screen_layer_composer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_green_screen_layer_composer.py`:

```python
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.matte.green_screen_layer_composer import GreenScreenCompetitiveLayerComposer


def test_competitive_composer_returns_expected_debug_layers():
    composer = GreenScreenCompetitiveLayerComposer()
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    base_alpha = np.zeros((3, 4), dtype=np.float32)
    ai_alpha = np.zeros((3, 4), dtype=np.float32)
    semantic_alpha = np.zeros((3, 4), dtype=np.float32)
    subject_alpha = np.zeros((3, 4), dtype=np.float32)
    effect_alpha = np.zeros((3, 4), dtype=np.float32)
    subject_gate = np.zeros((3, 4), dtype=np.float32)

    result = composer.compose(
        frame=frame,
        base_alpha=base_alpha,
        ai_alpha=ai_alpha,
        semantic_subject_alpha=semantic_alpha,
        subject_alpha=subject_alpha,
        effect_alpha=effect_alpha,
        subject_gate=subject_gate,
    )

    assert result.final_alpha.shape == (3, 4)
    assert result.subject_alpha_out.shape == (3, 4)
    assert result.effect_alpha_out.shape == (3, 4)
    assert result.ownership.subject.shape == (3, 4)
    assert result.ownership.effect.shape == (3, 4)
    assert result.ownership.background.shape == (3, 4)
    assert set(result.debug_layers) >= {
        "subject_candidate_alpha",
        "effect_candidate_alpha",
        "background_confidence",
        "subject_ownership",
        "effect_ownership",
        "background_ownership",
        "subject_alpha_out",
        "effect_alpha_out",
        "final_alpha",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/core/test_green_screen_layer_composer.py::test_competitive_composer_returns_expected_debug_layers -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'matteflow.matte.green_screen_layer_composer'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/matteflow/matte/green_screen_layer_composer.py`:

```python
"""Competitive subject/effect/background composer for GVM green-screen mattes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LayerCandidate:
    alpha: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class LayerOwnership:
    subject: np.ndarray
    effect: np.ndarray
    background: np.ndarray


@dataclass(frozen=True)
class CompetitiveLayerResult:
    final_alpha: np.ndarray
    subject_alpha_out: np.ndarray
    effect_alpha_out: np.ndarray
    ownership: LayerOwnership
    debug_layers: dict[str, np.ndarray]


class GreenScreenCompetitiveLayerComposer:
    """Assign pixels to subject, effect, or background before alpha composition."""

    def compose(
        self,
        *,
        frame: np.ndarray,
        base_alpha: np.ndarray,
        ai_alpha: np.ndarray,
        semantic_subject_alpha: np.ndarray | None,
        subject_alpha: np.ndarray,
        effect_alpha: np.ndarray,
        subject_gate: np.ndarray,
    ) -> CompetitiveLayerResult:
        base = self._clip(base_alpha)
        ai = self._clip(ai_alpha)
        semantic = self._clip(semantic_subject_alpha) if semantic_subject_alpha is not None else np.zeros_like(base)
        subject = self._clip(subject_alpha)
        effect = self._clip(effect_alpha)
        gate = self._clip(subject_gate)

        subject_confidence = self._clip(np.maximum.reduce([subject, ai, semantic, gate]))
        effect_confidence = self._clip(effect)
        background_confidence = self._clip((1.0 - base) * (1.0 - ai) * (1.0 - semantic))

        ownership = self._build_ownership(subject_confidence, effect_confidence, background_confidence)
        subject_alpha_out = self._clip(subject * ownership.subject)
        effect_alpha_out = self._clip(effect * ownership.effect)
        final_alpha = self._clip(subject_alpha_out + effect_alpha_out * (1.0 - subject_alpha_out))

        debug_layers = {
            "subject_candidate_alpha": subject,
            "effect_candidate_alpha": effect,
            "background_confidence": background_confidence,
            "subject_ownership": ownership.subject,
            "effect_ownership": ownership.effect,
            "background_ownership": ownership.background,
            "subject_alpha_out": subject_alpha_out,
            "effect_alpha_out": effect_alpha_out,
            "final_alpha": final_alpha,
        }
        return CompetitiveLayerResult(final_alpha, subject_alpha_out, effect_alpha_out, ownership, debug_layers)

    def _build_ownership(
        self,
        subject_confidence: np.ndarray,
        effect_confidence: np.ndarray,
        background_confidence: np.ndarray,
    ) -> LayerOwnership:
        subject = self._clip(subject_confidence * (1.0 - effect_confidence) * (1.0 - background_confidence))
        effect = self._clip(effect_confidence * (1.0 - 0.65 * subject) * (1.0 - background_confidence))
        background = self._clip(background_confidence * (1.0 - subject) * (1.0 - effect))
        return LayerOwnership(subject=subject, effect=effect, background=background)

    @staticmethod
    def _clip(alpha: np.ndarray) -> np.ndarray:
        return np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/core/test_green_screen_layer_composer.py::test_competitive_composer_returns_expected_debug_layers -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/matteflow/matte/green_screen_layer_composer.py tests/core/test_green_screen_layer_composer.py
git commit -m "feat: add green screen competitive composer contract"
```

## Task 2: Route Luminous Effects Away From Semantic Subject Bleed

**Files:**
- Modify: `src/matteflow/matte/green_screen_layer_composer.py`
- Modify: `tests/core/test_green_screen_layer_composer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_green_screen_layer_composer.py`:

```python
def test_competitive_composer_routes_luminous_effect_over_semantic_subject_bleed():
    composer = GreenScreenCompetitiveLayerComposer()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    base_alpha = np.full((4, 4), 0.50, dtype=np.float32)
    ai_alpha = np.full((4, 4), 0.90, dtype=np.float32)
    semantic_alpha = np.full((4, 4), 1.0, dtype=np.float32)
    subject_alpha = np.full((4, 4), 1.0, dtype=np.float32)
    effect_alpha = np.full((4, 4), 0.70, dtype=np.float32)
    subject_gate = np.full((4, 4), 1.0, dtype=np.float32)

    result = composer.compose(
        frame=frame,
        base_alpha=base_alpha,
        ai_alpha=ai_alpha,
        semantic_subject_alpha=semantic_alpha,
        subject_alpha=subject_alpha,
        effect_alpha=effect_alpha,
        subject_gate=subject_gate,
    )

    assert float(result.ownership.effect.mean()) >= 0.55
    assert float(result.ownership.subject.mean()) <= 0.45
    assert float(result.effect_alpha_out.mean()) >= 0.38
    assert float(result.final_alpha.mean()) < 0.95
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/core/test_green_screen_layer_composer.py::test_competitive_composer_routes_luminous_effect_over_semantic_subject_bleed -v
```

Expected: FAIL because the initial ownership formula still lets subject dominate when subject and semantic are both high.

- [ ] **Step 3: Implement effect-over-subject evidence**

Update `src/matteflow/matte/green_screen_layer_composer.py`:

```python
    def compose(
        self,
        *,
        frame: np.ndarray,
        base_alpha: np.ndarray,
        ai_alpha: np.ndarray,
        semantic_subject_alpha: np.ndarray | None,
        subject_alpha: np.ndarray,
        effect_alpha: np.ndarray,
        subject_gate: np.ndarray,
    ) -> CompetitiveLayerResult:
        base = self._clip(base_alpha)
        ai = self._clip(ai_alpha)
        semantic = self._clip(semantic_subject_alpha) if semantic_subject_alpha is not None else np.zeros_like(base)
        subject = self._clip(subject_alpha)
        effect = self._clip(effect_alpha)
        gate = self._clip(subject_gate)

        subject_confidence = self._clip(np.maximum.reduce([subject, ai, semantic, gate]))
        effect_confidence = self._clip(effect)
        background_confidence = self._clip((1.0 - base) * (1.0 - ai) * (1.0 - semantic))
        effect_over_subject = self._build_effect_over_subject_evidence(
            subject_confidence,
            effect_confidence,
            semantic,
            base,
        )

        ownership = self._build_ownership(
            subject_confidence,
            effect_confidence,
            background_confidence,
            effect_over_subject,
        )
        subject_alpha_out = self._clip(subject * ownership.subject)
        effect_alpha_out = self._clip(effect * ownership.effect)
        final_alpha = self._clip(subject_alpha_out + effect_alpha_out * (1.0 - subject_alpha_out))

        debug_layers = {
            "subject_candidate_alpha": subject,
            "effect_candidate_alpha": effect,
            "background_confidence": background_confidence,
            "effect_over_subject_evidence": effect_over_subject,
            "subject_ownership": ownership.subject,
            "effect_ownership": ownership.effect,
            "background_ownership": ownership.background,
            "subject_alpha_out": subject_alpha_out,
            "effect_alpha_out": effect_alpha_out,
            "final_alpha": final_alpha,
        }
        return CompetitiveLayerResult(final_alpha, subject_alpha_out, effect_alpha_out, ownership, debug_layers)

    def _build_effect_over_subject_evidence(
        self,
        subject_confidence: np.ndarray,
        effect_confidence: np.ndarray,
        semantic_subject_alpha: np.ndarray,
        base_alpha: np.ndarray,
    ) -> np.ndarray:
        semantic_bleed = semantic_subject_alpha * (1.0 - np.clip(base_alpha, 0.0, 1.0))
        effect_strength = self._smoothstep(effect_confidence, 0.28, 0.68)
        subject_is_semantic_dominated = self._clip(semantic_bleed * (1.0 - 0.35 * subject_confidence))
        return self._clip(np.maximum(effect_strength, effect_strength * subject_is_semantic_dominated))

    def _build_ownership(
        self,
        subject_confidence: np.ndarray,
        effect_confidence: np.ndarray,
        background_confidence: np.ndarray,
        effect_over_subject: np.ndarray,
    ) -> LayerOwnership:
        effect = self._clip(effect_confidence * (0.35 + 0.65 * effect_over_subject) * (1.0 - background_confidence))
        subject = self._clip(subject_confidence * (1.0 - effect_over_subject) * (1.0 - background_confidence))
        background = self._clip(background_confidence * (1.0 - subject) * (1.0 - effect))
        return LayerOwnership(subject=subject, effect=effect, background=background)

    @staticmethod
    def _smoothstep(x: np.ndarray, low: float, high: float) -> np.ndarray:
        t = np.clip((x - low) / max(high - low, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
```

- [ ] **Step 4: Run composer tests**

Run:

```powershell
python -m pytest tests/core/test_green_screen_layer_composer.py -v
```

Expected: all composer tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/matteflow/matte/green_screen_layer_composer.py tests/core/test_green_screen_layer_composer.py
git commit -m "feat: route luminous effects through ownership"
```

## Task 3: Add Background Ownership Suppression

**Files:**
- Modify: `src/matteflow/matte/green_screen_layer_composer.py`
- Modify: `tests/core/test_green_screen_layer_composer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_green_screen_layer_composer.py`:

```python
def test_competitive_composer_prefers_background_for_low_support_blue_cloud():
    composer = GreenScreenCompetitiveLayerComposer()
    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    base_alpha = np.full((5, 5), 0.03, dtype=np.float32)
    ai_alpha = np.full((5, 5), 0.01, dtype=np.float32)
    semantic_alpha = np.full((5, 5), 0.02, dtype=np.float32)
    subject_alpha = np.full((5, 5), 0.05, dtype=np.float32)
    effect_alpha = np.full((5, 5), 0.30, dtype=np.float32)
    subject_gate = np.zeros((5, 5), dtype=np.float32)

    result = composer.compose(
        frame=frame,
        base_alpha=base_alpha,
        ai_alpha=ai_alpha,
        semantic_subject_alpha=semantic_alpha,
        subject_alpha=subject_alpha,
        effect_alpha=effect_alpha,
        subject_gate=subject_gate,
    )

    assert float(result.ownership.background.mean()) >= 0.80
    assert float(result.ownership.effect.mean()) <= 0.10
    assert float(result.final_alpha.mean()) <= 0.08
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/core/test_green_screen_layer_composer.py::test_competitive_composer_prefers_background_for_low_support_blue_cloud -v
```

Expected: FAIL if effect ownership remains too high for weak low-support background regions.

- [ ] **Step 3: Strengthen background-first ownership**

Update `_build_ownership()` in `src/matteflow/matte/green_screen_layer_composer.py`:

```python
    def _build_ownership(
        self,
        subject_confidence: np.ndarray,
        effect_confidence: np.ndarray,
        background_confidence: np.ndarray,
        effect_over_subject: np.ndarray,
    ) -> LayerOwnership:
        background = self._clip(background_confidence)
        non_background = 1.0 - background
        effect = self._clip(effect_confidence * (0.35 + 0.65 * effect_over_subject) * non_background)
        subject = self._clip(subject_confidence * (1.0 - effect_over_subject) * non_background)
        return LayerOwnership(subject=subject, effect=effect, background=background)
```

- [ ] **Step 4: Run composer tests**

Run:

```powershell
python -m pytest tests/core/test_green_screen_layer_composer.py -v
```

Expected: all composer tests PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/matteflow/matte/green_screen_layer_composer.py tests/core/test_green_screen_layer_composer.py
git commit -m "feat: suppress low support background ownership"
```

## Task 4: Integrate Composer Into GVM Green-Screen Fusion

**Files:**
- Modify: `src/matteflow/matte/hybrid_matte.py`
- Modify: `tests/core/test_green_screen_fusion.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/core/test_green_screen_fusion.py`:

```python
def test_green_screen_gvm_merge_exports_competitive_ownership_debug_layers():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    frame_bgr = cv2.imread(str(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"))
    assert frame_bgr is not None
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_f = frame.astype(np.float32, copy=False)
    red = frame_f[:, :, 0]
    green = frame_f[:, :, 1]
    blue = frame_f[:, :, 2]
    purple_subject = (red > 120.0) & (blue > 130.0) & (green < 180.0)
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.where(purple_subject & (base_alpha >= 0.45), 1.0, 0.0039).astype(np.float32)
    semantic_subject_alpha = np.where(purple_subject, 1.0, 0.0).astype(np.float32)

    merged = matte._merge_green_screen_effects(
        [base_alpha],
        [ai_alpha],
        [frame],
        semantic_subject_alphas=[semantic_subject_alpha],
    )[0]

    debug = matte.last_green_screen_layer_debug
    assert debug is not None
    assert "subject_ownership" in debug
    assert "effect_ownership" in debug
    assert "background_ownership" in debug
    assert "final_alpha" in debug
    assert np.allclose(merged, debug["final_alpha"])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/core/test_green_screen_fusion.py::test_green_screen_gvm_merge_exports_competitive_ownership_debug_layers -v
```

Expected: FAIL with `AttributeError: 'HybridMatte' object has no attribute 'last_green_screen_layer_debug'` or missing debug keys.

- [ ] **Step 3: Add integration state and composer call**

Modify `src/matteflow/matte/hybrid_matte.py`.

In `__init__`, add:

```python
        self.last_green_screen_layer_debug = None
```

In `_merge_green_screen_effects()`, set debug to `None` before the loop:

```python
        self.last_green_screen_layer_debug = None
```

Replace the final old fusion section:

```python
            fused_alpha = self._soft_fuse_layers(solid_alpha, effect_alpha)
            self._log_transparency_fusion_stats("green_screen", solid_alpha, effect_alpha, fused_alpha)
            merged.append(fused_alpha)
```

with:

```python
            if self.last_active_ai_model == "gvm" and frame is not None:
                from .green_screen_layer_composer import GreenScreenCompetitiveLayerComposer

                result = GreenScreenCompetitiveLayerComposer().compose(
                    frame=frame,
                    base_alpha=base_alpha,
                    ai_alpha=ai_alpha,
                    semantic_subject_alpha=semantic_subject_alpha,
                    subject_alpha=solid_alpha,
                    effect_alpha=effect_alpha,
                    subject_gate=subject_gate,
                )
                fused_alpha = result.final_alpha
                self.last_green_screen_layer_debug = result.debug_layers
            else:
                fused_alpha = self._soft_fuse_layers(solid_alpha, effect_alpha)
            self._log_transparency_fusion_stats("green_screen", solid_alpha, effect_alpha, fused_alpha)
            merged.append(fused_alpha)
```

- [ ] **Step 4: Run integration test**

Run:

```powershell
python -m pytest tests/core/test_green_screen_fusion.py::test_green_screen_gvm_merge_exports_competitive_ownership_debug_layers -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/matteflow/matte/hybrid_matte.py tests/core/test_green_screen_fusion.py
git commit -m "feat: integrate competitive composer for gvm green screen"
```

## Task 5: Add Real `test_frame_3.jpg` Ownership Regression Tests

**Files:**
- Modify: `tests/core/test_green_screen_fusion.py`
- Modify: `src/matteflow/matte/green_screen_layer_composer.py`
- Modify: `src/matteflow/matte/hybrid_matte.py` only if the test proves integration data is missing

- [ ] **Step 1: Write the failing real-frame test**

Append to `tests/core/test_green_screen_fusion.py`:

```python
def test_green_screen_competitive_composer_balances_subject_effect_and_background_on_real_frame():
    matte = HybridMatte(MattingConfig(use_ai=False))
    matte.last_active_ai_model = "gvm"
    frame_bgr = cv2.imread(str(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"))
    assert frame_bgr is not None
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_f = frame.astype(np.float32, copy=False)
    red = frame_f[:, :, 0]
    green = frame_f[:, :, 1]
    blue = frame_f[:, :, 2]
    brightness = frame_f.mean(axis=2)
    chroma = frame_f.max(axis=2) - frame_f.min(axis=2)
    purple_subject = (red > 120.0) & (blue > 130.0) & (green < 180.0)
    base_alpha = matte.green_matte.generate(frame)
    ai_alpha = np.where(purple_subject & (base_alpha >= 0.45), 1.0, 0.0039).astype(np.float32)
    semantic_subject_alpha = np.where(purple_subject, 1.0, 0.0).astype(np.float32)
    low_base_purple = purple_subject & matte._green_screen_non_screen_mask(frame) & (base_alpha < 0.45)
    luminous_core = (brightness > 205.0) & (chroma < 70.0) & (base_alpha < 0.75) & (~purple_subject)
    core_reach = cv2.dilate(
        luminous_core.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
        iterations=1,
    ).astype(bool)
    far_blue_background = (
        (~core_reach)
        & (blue > 130.0)
        & (green > 100.0)
        & (red < 150.0)
        & (base_alpha < 0.20)
        & (~purple_subject)
    )

    merged = matte._merge_green_screen_effects(
        [base_alpha],
        [ai_alpha],
        [frame],
        semantic_subject_alphas=[semantic_subject_alpha],
    )[0]
    debug = matte.last_green_screen_layer_debug
    assert debug is not None

    assert int(low_base_purple.sum()) >= 100_000
    assert int(luminous_core.sum()) >= 50_000
    assert int(far_blue_background.sum()) >= 300_000
    assert float(merged[low_base_purple].mean()) >= 0.74
    assert float(debug["subject_ownership"][low_base_purple].mean()) >= 0.55
    assert float(debug["effect_ownership"][luminous_core].mean()) >= 0.35
    assert float(debug["background_ownership"][far_blue_background].mean()) >= 0.70
    assert float(merged[far_blue_background].mean()) <= 0.08
```

- [ ] **Step 2: Run test to verify it fails or exposes current ownership issue**

Run:

```powershell
python -m pytest tests/core/test_green_screen_fusion.py::test_green_screen_competitive_composer_balances_subject_effect_and_background_on_real_frame -v
```

Expected: FAIL on one of the ownership assertions if Task 4 only wired the initial synthetic composer.

- [ ] **Step 3: Add frame-aware effect and background evidence**

Update `src/matteflow/matte/green_screen_layer_composer.py` with frame-aware evidence. Add methods:

```python
    def _frame_effect_evidence(self, frame: np.ndarray, base_alpha: np.ndarray) -> np.ndarray:
        frame_f = frame.astype(np.float32, copy=False)
        red = frame_f[:, :, 0]
        green = frame_f[:, :, 1]
        blue = frame_f[:, :, 2]
        brightness = (red + green + blue) / 3.0
        chroma = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
        screen_green = (green > red + 30.0) & (green > blue + 20.0) & (green > 90.0)
        bright_core = (brightness > 205.0) & (chroma < 70.0) & (base_alpha < 0.80) & (~screen_green)
        blue_white = (brightness > 185.0) & (blue > 150.0) & (green > 140.0) & (red > 130.0) & (chroma < 95.0) & (~screen_green)
        return self._clip((bright_core | blue_white).astype(np.float32))

    def _frame_background_evidence(
        self,
        frame: np.ndarray,
        base_alpha: np.ndarray,
        ai_alpha: np.ndarray,
        semantic_alpha: np.ndarray,
        effect_evidence: np.ndarray,
    ) -> np.ndarray:
        frame_f = frame.astype(np.float32, copy=False)
        red = frame_f[:, :, 0]
        green = frame_f[:, :, 1]
        blue = frame_f[:, :, 2]
        low_support = (base_alpha < 0.20) & (ai_alpha < 0.20) & (semantic_alpha < 0.20)
        blue_cloud = (blue > 130.0) & (green > 100.0) & (red < 150.0)
        far_from_effect = effect_evidence < 0.20
        return self._clip((low_support & blue_cloud & far_from_effect).astype(np.float32))
```

Then in `compose()` replace:

```python
        effect_confidence = self._clip(effect)
        background_confidence = self._clip((1.0 - base) * (1.0 - ai) * (1.0 - semantic))
```

with:

```python
        frame_effect = self._frame_effect_evidence(frame, base)
        effect_confidence = self._clip(np.maximum(effect, frame_effect))
        frame_background = self._frame_background_evidence(frame, base, ai, semantic, effect_confidence)
        background_confidence = self._clip(np.maximum((1.0 - base) * (1.0 - ai) * (1.0 - semantic), frame_background))
```

And include debug keys:

```python
            "frame_effect_evidence": frame_effect,
            "frame_background_evidence": frame_background,
```

- [ ] **Step 4: Run real-frame test**

Run:

```powershell
python -m pytest tests/core/test_green_screen_fusion.py::test_green_screen_competitive_composer_balances_subject_effect_and_background_on_real_frame -v
```

Expected: PASS.

- [ ] **Step 5: Run composer and green-screen focused tests**

Run:

```powershell
python -m pytest tests/core/test_green_screen_layer_composer.py tests/core/test_green_screen_fusion.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/matteflow/matte/green_screen_layer_composer.py src/matteflow/matte/hybrid_matte.py tests/core/test_green_screen_layer_composer.py tests/core/test_green_screen_fusion.py
git commit -m "test: cover competitive composer on real gvm frame"
```

## Task 6: Export Diagnostic Ownership Layers For Manual Review

**Files:**
- Modify: `.superpowers/diagnostics/export_semantic_layers.py`
- Do not commit this file unless the repository already tracks it.

- [ ] **Step 1: Update diagnostic script to include composer debug layers**

After the call that computes `real_generate`, add:

```python
composer_debug = matte.last_green_screen_layer_debug or {}
for debug_name, debug_alpha in composer_debug.items():
    if isinstance(debug_alpha, np.ndarray) and debug_alpha.shape == base_alpha.shape:
        layers[f"composer_{debug_name}"] = debug_alpha
```

- [ ] **Step 2: Run diagnostic export**

Run:

```powershell
python .superpowers\diagnostics\export_semantic_layers.py
```

Expected: exits with code 0 and writes updated `contact_sheet.png` plus individual `composer_*` images.

- [ ] **Step 3: Inspect contact sheet manually**

Open:

```text
http://127.0.0.1:8791/contact_sheet.png
```

Expected visual result:

- `subject_ownership` is strongest on purple furry subject.
- `effect_ownership` is strongest on luminous bands and halo.
- `background_ownership` is strongest on distant blue/green background cloud.

- [ ] **Step 4: Do not commit diagnostic-only changes unless requested**

Run:

```powershell
git status --short
```

Expected: diagnostic script may appear modified or untracked. Leave it uncommitted unless the user asks for diagnostic tooling to be committed.

## Task 7: Full Verification And GUI Retest

**Files:**
- No source changes unless tests expose a specific regression.

- [ ] **Step 1: Run focused regression suite**

Run:

```powershell
python -m pytest tests/core/test_green_screen_layer_composer.py tests/core/test_birefnet_matte.py tests/core/test_green_screen_fusion.py -v
```

Expected: PASS.

- [ ] **Step 2: Check diagnostics**

Run:

```powershell
python -m pytest tests/core/test_diagnose_gvm_fusion.py -v
```

Expected: PASS if diagnostic tests are present in the current workspace.

- [ ] **Step 3: Check language diagnostics**

Run the editor diagnostics tool for all files.

Expected: no diagnostics introduced in:

- `src/matteflow/matte/green_screen_layer_composer.py`
- `src/matteflow/matte/hybrid_matte.py`
- `tests/core/test_green_screen_layer_composer.py`
- `tests/core/test_green_screen_fusion.py`

- [ ] **Step 4: Restart GUI on a free port**

If `7861` is still running from earlier, stop or reuse it. Otherwise run:

```powershell
python scripts/web_gui.py --port 7862
```

Expected log includes:

```text
Launching Gradio UI on port=7862 share=False
HTTP Request: HEAD http://localhost:7862/ "HTTP/1.1 200 OK"
```

- [ ] **Step 5: Ask user to retest**

Ask the user to open the active GUI URL and process `assets/frame/test_frame_3.jpg` in `GVM` mode.

Expected visual improvements:

- Purple subject is mostly complete.
- Lightning is preserved as an effect layer instead of becoming a solid subject blob.
- Distant blue/green background cloud remains mostly transparent.

- [ ] **Step 6: Commit final verification notes only if files changed**

If no files changed, do not commit.

If a verification helper was intentionally updated, commit only that helper:

```powershell
git add .superpowers/diagnostics/export_semantic_layers.py
git commit -m "chore: update competitive composer diagnostics"
```

## Self-Review

### Spec Coverage

- Candidate Builder: Task 1 and Task 5 create and populate candidate inputs.
- Evidence Maps: Task 2 and Task 5 add effect-over-subject, frame effect, and frame background evidence.
- Competitive Ownership: Tasks 2, 3, and 5 implement and test ownership maps.
- Composer: Task 1 creates composition output; Task 4 integrates it into `HybridMatte`.
- Debug Outputs: Task 1 defines debug keys; Task 4 exposes them on `HybridMatte`; Task 6 exports them for manual review.
- GVM-only integration: Task 4 calls composer only when `last_active_ai_model == "gvm"`.
- No GUI knobs: Task 7 retests existing `GVM` mode; no config or GUI controls are added.

### Placeholder Scan

- No unresolved placeholder entries remain in implementation steps.
- Each code-changing task includes concrete code blocks.
- Each test step includes exact command and expected result.

### Type Consistency

- `LayerOwnership.subject/effect/background` names match all tests.
- `CompetitiveLayerResult.final_alpha/subject_alpha_out/effect_alpha_out/ownership/debug_layers` names match all integration steps.
- `GreenScreenCompetitiveLayerComposer.compose(...)` signature is consistent across tasks.
