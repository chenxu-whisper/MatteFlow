# Matting Quality System Design

Date: 2026-06-24

## Summary

MatteFlow needs a quality decision layer that makes MatAnyone2, SAM2, and BiRefNet measurable, selectable, and regression-tested. The goal is not to add one more model priority rule. The goal is to generate multiple candidate mattes, evaluate their reliability by region, select or fuse the best candidate for each region, and record enough evidence to compare quality across future changes.

This design focuses on the quality problems observed in the current visual review:

- Model disagreement: base, GVM, and fused outputs can contradict each other, causing subject loss or false retention.
- Soft-edge and hair detail loss: alpha transitions can become hard, broken, or overly cleaned by later stages.
- Residue and contamination: green or gray edge residue can survive, while color decontamination can also become too aggressive.

The first implementation should use rule-based quality metrics and deterministic selection. The interfaces must leave room for a learned quality evaluator later.

## Goals

1. Make MatAnyone2, SAM2, and BiRefNet comparable through a shared candidate output contract.
2. Produce quality signals for each candidate at frame level and region level.
3. Select or fuse candidates by region instead of choosing one model for a whole frame.
4. Record model decisions, quality signals, and visual artifacts in the existing processing report flow.
5. Add a regression workflow that can compare quality changes on a fixed difficult sample set.
6. Keep existing green-screen, black-background, and fallback behavior available while the new quality system is rolled out.

## Non-Goals

1. Training or fine-tuning a new model in the first phase.
2. Replacing the entire `HybridMatte` implementation in one change.
3. Building a cloud service or distributed inference system.
4. Requiring manual labels for every sample before the first useful regression suite exists.
5. Making SAM2 mandatory for all runs. SAM2 should be used when target guidance is available or when the selector needs semantic constraints.

## Current Context

The current codebase already has important pieces:

- `src/matteflow/matte/hybrid_matte.py` coordinates traditional mattes, AI models, fallback logic, and green-screen-specific fusion.
- `src/matteflow/matte/fusion_quality_gate.py` has an initial region-aware fusion mechanism.
- `src/matteflow/analysis/region_ownership.py` classifies subject, hair edge, transparent effect, luminous prop, uncertain edge, and background residue regions.
- `src/matteflow/analysis/alpha_quality.py` computes lightweight alpha quality signals.
- `src/matteflow/analysis/p0_quality.py` classifies high-level risks.
- `src/matteflow/reporting/processing_report.py` already emits structured processing reports.
- `tests/core/test_quality_regression.py` provides a starting point for report-based regression checks.

The gap is that candidate generation, quality evaluation, selection, and regression are not unified around MatAnyone2, SAM2, and BiRefNet.

## Proposed Architecture

Add a quality decision layer with four new responsibilities:

1. Candidate generation
2. Candidate evaluation
3. Region-level selection
4. Regression evidence generation

The data flow is:

```text
Decoded frames
  |
  v
Background analysis / optional user guidance
  |
  v
Candidate generators
  |-- MatAnyone2 candidate
  |-- SAM2-guided candidate
  |-- BiRefNet candidate
  |-- existing traditional/base candidate
  |
  v
Quality evaluator
  |
  v
Region-level selector
  |
  v
Selected alpha sequence + decision diagnostics
  |
  v
Existing refine / despeckle / repair / temporal / decontaminate / encode stages
  |
  v
Processing report + regression artifacts
```

## Module Design

### 1. Candidate Contract

Introduce candidate contracts under `src/matteflow/matte/candidates/`. Use a package instead of a single large file because the real model adapters will grow independently.

Each candidate generator returns:

```python
@dataclass(frozen=True)
class MatteCandidate:
    name: str
    alpha: np.ndarray
    confidence: np.ndarray | None
    source: str
    runtime_ms: float
    diagnostics: dict[str, Any]
```

For sequences:

```python
@dataclass(frozen=True)
class MatteCandidateSequence:
    name: str
    alphas: list[np.ndarray]
    confidences: list[np.ndarray | None]
    source: str
    runtime_ms: float
    diagnostics: dict[str, Any]
```

Rules:

- `alpha` must be float32, clipped to `[0.0, 1.0]`, and match the frame height and width.
- `confidence` is optional in phase one. If absent, the evaluator derives confidence from quality metrics and region ownership.
- `source` should be stable, for example `matanyone2`, `sam2_guided`, `birefnet`, `traditional_green`, or `traditional_black`.
- `diagnostics` must be JSON-serializable.

### 2. Candidate Generators

Add wrappers that adapt existing model modules without changing their internal inference code first:

- `MatAnyone2CandidateGenerator`
- `SAM2GuidedCandidateGenerator`
- `BiRefNetCandidateGenerator`
- `TraditionalCandidateGenerator`

The wrappers should handle model availability, cancellation checks, progress callbacks, output normalization, and timing. If a model is unavailable, the generator returns a structured skipped result instead of throwing unless the user explicitly requested that model.

SAM2 should be treated as guidance-first:

- If a first-frame mask, box, or point prompt exists, SAM2 produces a target constraint candidate.
- If no guidance exists, SAM2 can be skipped by default in phase one to avoid turning semantic segmentation uncertainty into false alpha confidence.
- The selector may use SAM2 output as a region constraint even when SAM2 is not the final alpha source.

### 3. Quality Evaluator

Add `src/matteflow/evaluation/matte_quality.py`.

The evaluator consumes frames, candidate alphas, optional confidences, and region ownership. It returns per-candidate and per-region scores.

Initial signals:

- `subject_coverage`: whether known subject-like regions retain enough alpha.
- `soft_edge_continuity`: whether soft alpha bands are continuous instead of broken or stair-stepped.
- `hair_edge_preservation`: whether hair/feather edge regions avoid low-alpha collapse.
- `background_cleanliness`: whether background-residue regions stay near zero alpha.
- `transparent_effect_preservation`: whether transparent or luminous regions avoid being erased.
- `spill_risk`: whether green/gray contaminated edge regions remain likely visible.
- `temporal_stability`: frame-to-frame alpha change in comparable regions.
- `model_disagreement`: how strongly a candidate differs from other candidates in important regions.

The evaluator should produce:

```python
@dataclass(frozen=True)
class CandidateQuality:
    candidate_name: str
    frame_index: int
    overall_score: float
    region_scores: dict[str, float]
    signals: dict[str, float | int | str]
```

Score direction must be consistent: higher is better. Risk signals can still be recorded separately.

### 4. Region-Level Selector

Add `src/matteflow/matte/quality_selector.py`.

The selector receives candidates, quality scores, and region ownership. It returns the selected alpha plus diagnostics.

Selection principles:

- Subject core: prefer candidates with strong subject coverage and temporal stability.
- Hair and uncertain edge: prefer candidates with better soft-edge continuity and less low-alpha collapse.
- Transparent effects and luminous props: prefer candidates that preserve low-to-mid alpha instead of hard clipping.
- Background residue: prefer candidates with lower alpha and lower spill risk.
- SAM2-guided mask: use as a semantic constraint when available, not as a hard alpha replacement unless its quality score wins.
- Existing traditional green/black matte remains a candidate, especially for green-screen base structure and black-background transparent effects.

The selector must produce diagnostics:

```python
@dataclass(frozen=True)
class SelectionDecision:
    frame_index: int
    selected_by_region: dict[str, str]
    region_scores: dict[str, dict[str, float]]
    rejected_takeovers: dict[str, int]
    warnings: list[str]
```

This should extend or reuse concepts from `FusionQualityGate` instead of duplicating unrelated fusion logic.

### 5. Pipeline Integration

Add a config flag:

```python
quality_selection_enable: bool = False
quality_candidate_models: tuple[str, ...] = ("matanyone2", "sam2", "birefnet", "traditional")
quality_selection_mode: str = "region"
```

Initial rollout:

- Default remains current behavior.
- When `quality_selection_enable=True`, `HybridMatte` delegates matte generation to a new `QualityDrivenMatte` coordinator. `HybridMatte` remains the public integration point for the pipeline, while `QualityDrivenMatte` owns candidate generation, evaluation, and selection.
- The selected alpha sequence then continues through existing refine, despeckle, repair, temporal stabilization, decontamination, and encoding stages.

This avoids destabilizing current CLI and GUI behavior while allowing targeted testing.

### 6. Reporting

Extend the processing report with:

- Candidate model list and availability status.
- Per-model runtime.
- Per-model quality summary.
- Per-region selected model counts.
- Worst frames by each risk category.
- Paths to optional debug artifacts.

Debug artifacts should include:

- Candidate alpha contact sheet.
- Region selection overlay.
- Alpha difference heatmaps between selected output and each candidate.
- Local zoom sheets for high-risk regions.

The existing `output_debug` flag can control whether images are written. JSON summaries should be written even when image debug output is disabled.

### 7. Regression Suite

Add a fixed difficult-sample regression workflow under the existing evaluation structure.

Recommended layout:

```text
tests/fixtures/matting_quality/
  manifest.json
  green_screen/
  black_background/
  video_short/
```

The manifest should define:

- Input path
- Background mode
- Quality mode
- Enabled candidates
- Expected risk ceilings
- Optional baseline report path
- Optional regions of interest for zoom artifacts

Regression outputs:

- `quality_summary.json`
- `candidate_decisions.json`
- `p0_risks.json`
- `contact_sheet.png` when debug output is enabled
- `diff_heatmaps/` when debug output is enabled

Automated gates:

- No P0 risk category may regress beyond configured tolerance.
- Background residue and hair-edge-loss risk must not exceed sample-specific ceilings.
- Candidate selection must be deterministic for the same input and config.
- Missing optional models should mark tests as skipped or degraded according to the manifest, not as unrelated failures.

## Error Handling

The quality system should distinguish these cases:

- Requested model unavailable: fail fast with a clear model availability error.
- Optional candidate unavailable: record skipped candidate and continue.
- Candidate output shape mismatch: fail the quality selection stage.
- Candidate alpha contains invalid values: sanitize if finite, fail if NaN or infinite values remain.
- SAM2 guidance missing: skip SAM2-guided candidate in phase one and record the reason.
- No AI candidate succeeds: fall back to existing traditional behavior and record degraded mode.

## Testing Strategy

Unit tests:

- Candidate output normalization.
- Quality evaluator score direction and signal extraction.
- Selector behavior on synthetic subject, hair, transparent-effect, and background-residue masks.
- Deterministic selection when candidates have equal scores.
- Missing model handling.

Integration tests:

- Pipeline runs with `quality_selection_enable=False` and preserves existing behavior.
- Pipeline runs with `quality_selection_enable=True` using fake candidate generators.
- Processing report includes candidate decisions and quality summaries.
- Regression evaluator flags threshold and baseline regressions.

Visual/manual tests:

- Generate contact sheets for the current green-screen diagnostic sample.
- Compare local zoom regions for model disagreement, soft-edge/hair, and residue/contamination.

## Rollout Plan

Phase 1: Infrastructure without changing default behavior

- Add candidate contract.
- Add fake or lightweight candidate tests.
- Add quality evaluator and selector.
- Add report schema extensions.
- Keep `quality_selection_enable=False` by default.

Phase 2: Real model wrappers

- Wrap MatAnyone2, BiRefNet, and traditional matte as candidates.
- Add SAM2-guided candidate when guidance is provided.
- Run quality selection on short samples.

Phase 3: Regression and GUI visibility

- Add difficult-sample manifest.
- Generate contact sheets and local zoom artifacts.
- Surface candidate decisions in the GUI report view.

Phase 4: Quality tuning

- Tune region scores against the difficult-sample set.
- Decide whether MatAnyone2 should become the high-quality video default.
- Decide whether a learned quality evaluator is justified by remaining failures.

## Open Decisions Resolved for This Spec

- The first phase uses rule-based quality scoring, not a learned evaluator.
- The new path is opt-in through configuration until regression evidence is strong.
- SAM2 is guidance-first and should not be treated as a universal alpha generator without prompts.
- Selection happens by region, not by whole frame.
- Existing traditional matte remains a candidate, not a discarded fallback.

## Success Criteria

The design is successful when:

1. A single run can produce candidate outputs from MatAnyone2, BiRefNet, SAM2-guided mode when available, and traditional matte.
2. The report explains which candidate won for subject, hair edge, transparent effect, uncertain edge, and background residue regions.
3. A regression run can detect worse hair-edge loss, background residue, transparent-effect loss, or temporal instability before manual review.
4. Debug artifacts make model disagreement and edge failures visible without rerunning ad hoc scripts.
5. Current default behavior remains available while the quality system is validated.
