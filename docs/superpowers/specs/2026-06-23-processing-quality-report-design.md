# P3-A Processing Quality Report Design

## Summary

P3-A adds a structured processing quality report for each MatteFlow job. The report explains what the pipeline decided, which models and repair stages participated, what quality risks were detected, and what artifacts were written. It turns existing internal diagnostics from P0, P1, and P2 into a stable JSON contract that CLI, service, GUI, and later debug panels can consume.

This phase does not add new matting algorithms. It adds observability around the algorithms already present: `AlphaQualityAnalyzer`, `RegionOwnershipAnalyzer`, `ForegroundColorRecovery`, `FusionQualityGate`, `HybridMatte` fallback metrics, stage timings, and output/debug artifacts.

## Goals

- Write `processing_report.json` for successful pipeline runs.
- Include enough structured data to answer: background mode, active AI model, fallback/model decisions, per-stage timings, alpha quality metrics, region ownership statistics, foreground recovery statistics, fusion quality diagnostics, and generated artifact paths.
- Keep the report schema independent from GUI code so it can be used by CLI and Web GUI.
- Avoid making processing fail because report generation fails. Report generation errors should be recorded as warnings when possible and logged otherwise.
- Keep `output_debug=False` lightweight. The JSON report should be written by default, while large overlays remain controlled by `output_debug`.

## Non-Goals

- No GUI panel implementation in P3-A. P3-C can consume this report later.
- No model download or environment self-healing. That belongs to P3-B.
- No new matte selection behavior. Fusion and recovery behavior remain unchanged.
- No database or persistent job history beyond the output directory report file.
- No binary debug artifacts beyond existing debug overlays.

## User Value

When a result has green edges, missing light props, holes, or flicker, the report gives a concrete trail:

- which background path was used;
- which model actually produced the matte;
- whether fallback or fusion quality gates accepted or rejected candidates;
- whether region ownership detected luminous props or transparent effects;
- how much foreground color recovery was attempted or rejected;
- whether alpha quality metrics indicate holes, residue, speckles, or flicker.

This makes manual GUI testing and future bug reports more reproducible.

## Existing Context

Relevant current modules:

- `src/matteflow/pipeline.py`
  - Owns stage order and stage timings.
  - Already writes `debug/quality_report.txt` and overlays when `output_debug=True`.
  - Builds region context for downstream stages.
- `src/matteflow/analysis/alpha_quality.py`
  - Produces sequence-level alpha quality metrics.
- `src/matteflow/analysis/region_ownership.py`
  - Produces per-frame ownership masks.
- `src/matteflow/refine/color_decontaminate.py`
  - Writes `context["foreground_recovery"]` diagnostics.
- `src/matteflow/matte/fusion_quality_gate.py`
  - Produces fusion diagnostics for candidate selection.
- `src/matteflow/matte/hybrid_matte.py`
  - Exposes `last_active_ai_model`, `last_fallback_quality_metrics`, and green-screen debug layers.
- `src/matteflow/service.py`
  - Converts raw pipeline result into `ProcessResult`.
- `src/matteflow/diagnostics.py`
  - Already models environment and exception diagnostics.

## Recommended Approach

Implement a small report subsystem under `src/matteflow/reporting/` and call it from `MattingPipeline` near the end of processing.

Reasons:

- Keeps report schema, serialization, and summarization out of the already large pipeline module.
- Allows unit testing report assembly without running the full pipeline.
- Gives GUI and CLI a stable file contract without importing pipeline internals.
- Keeps P3-A limited to observability, avoiding behavior changes.

Rejected alternatives:

- Add report assembly directly inside `pipeline.py`: fastest, but grows an already central module and makes schema tests harder.
- Extend `diagnostics.py`: current diagnostics are mostly environment/error oriented; processing quality reports are a separate domain and should not overload the error model.
- Only write text reports: easier to read, but bad for GUI consumption and future automation.

## Report File Contract

File path:

```text
<output_dir>/processing_report.json
```

Encoding:

```text
UTF-8 JSON, pretty printed with stable key order
```

Top-level schema:

```json
{
  "schema_version": 1,
  "job": {
    "input_path": "input.mp4",
    "output_dir": "out",
    "frame_count": 120,
    "background_mode_requested": "auto",
    "background_mode_effective": "green_screen",
    "quality_mode": "high",
    "ai_model_requested": "auto",
    "ai_model_active": "gvm"
  },
  "timings": {
    "decode": 0.15,
    "analyze": 0.01,
    "matte": 12.4,
    "refine": 0.7,
    "despeckle": 0.1,
    "effect_prop_repair": 0.2,
    "stabilize": 0.6,
    "quality_debug": 0.0,
    "decontaminate": 0.3,
    "encode": 1.8,
    "total": 16.4
  },
  "quality": {
    "frame_count": 120,
    "overall_score": 0.91,
    "mean_edge_uncertainty": 0.08,
    "speckle_pixels": 12,
    "hole_pixels": 4,
    "background_residue": 0.01,
    "temporal_flicker": 0.03
  },
  "regions": {
    "subject_pixels": 123456,
    "hair_edge_pixels": 1200,
    "luminous_prop_pixels": 840,
    "transparent_effect_pixels": 6400,
    "background_residue_pixels": 300,
    "uncertain_edge_pixels": 8100
  },
  "model_decisions": {
    "active_ai_model": "gvm",
    "fallback_quality_metrics": {},
    "green_screen_layer_debug_available": true
  },
  "fusion": {
    "available": true,
    "selected_by_region": {
      "subject": "ai_core",
      "luminous_prop": "effect_prop_repair",
      "transparent_effect": "fx_detail",
      "background_residue": "green_key_base"
    },
    "rejected_takeovers": {
      "luminous_prop": 12,
      "background_residue": 5
    }
  },
  "foreground_recovery": {
    "frames": 120,
    "attempted_pixels": 5000,
    "accepted_pixels": 4100,
    "rejected_pixels": 900,
    "mean_weight": 0.42,
    "screen_rgb": [0.0, 210.0, 40.0]
  },
  "artifacts": {
    "processed_output": "processed_000000.png",
    "matte_output": "matte_000000.png",
    "debug_dir": "debug",
    "quality_report_txt": "debug/quality_report.txt"
  },
  "warnings": []
}
```

Fields may be `null` or omitted only where a stage is not applicable. Required top-level keys are:

- `schema_version`
- `job`
- `timings`
- `quality`
- `regions`
- `model_decisions`
- `fusion`
- `foreground_recovery`
- `artifacts`
- `warnings`

## Core Code Structure

### `src/matteflow/reporting/__init__.py`

Exports the report schema and writer:

```python
from .processing_report import (
    ProcessingReport,
    ProcessingReportBuilder,
    ProcessingReportWriter,
)

__all__ = [
    "ProcessingReport",
    "ProcessingReportBuilder",
    "ProcessingReportWriter",
]
```

### `src/matteflow/reporting/processing_report.py`

Owns report dataclasses, dict conversion, and JSON writing.

Proposed interfaces:

```python
@dataclass(frozen=True)
class ProcessingReport:
    schema_version: int
    job: dict[str, Any]
    timings: dict[str, float]
    quality: dict[str, Any]
    regions: dict[str, int]
    model_decisions: dict[str, Any]
    fusion: dict[str, Any]
    foreground_recovery: dict[str, Any]
    artifacts: dict[str, Any]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable dictionary matching the schema above."""


class ProcessingReportBuilder:
    def build(
        self,
        *,
        input_path: Path,
        output_dir: Path,
        config: MattingConfig,
        frame_count: int,
        background_mode_effective: BackgroundMode,
        timings: Mapping[str, float],
        quality_report: Any | None,
        region_context: Mapping[str, Any] | None,
        hybrid_matte: Any | None,
        decontaminate_context: Mapping[str, Any] | None,
        artifacts: Mapping[str, Any],
    ) -> ProcessingReport:
        """Assemble a ProcessingReport from pipeline stage evidence."""


class ProcessingReportWriter:
    def write(self, report: ProcessingReport, output_dir: Path) -> Path:
        """Write processing_report.json and return its path."""
```

### `src/matteflow/pipeline.py`

Pipeline integration should be narrow:

- Import `ProcessingReportBuilder` and `ProcessingReportWriter`.
- Keep `quality_report` from alpha analysis available even when `output_debug=False`.
- Keep `region_context` and `decontaminate_context` after post-processing.
- Call report writer after encoding, before returning the final result.
- Add `processing_report_path` to raw pipeline result.

Proposed integration shape:

```python
report = self.report_builder.build(
    input_path=input_path,
    output_dir=output_dir,
    config=self.config,
    frame_count=total_frames,
    background_mode_effective=bg_mode,
    timings=timings,
    quality_report=quality_report,
    region_context=decontaminate_context,
    hybrid_matte=getattr(self, "hybrid_matte", None),
    decontaminate_context=decontaminate_context,
    artifacts=self._collect_artifacts(output_dir),
)
report_path = self.report_writer.write(report, output_dir)
result["processing_report_path"] = str(report_path)
```

### `src/matteflow/service.py`

Extend `ProcessResult` with an optional report path:

```python
processing_report_path: Optional[Path] = None
```

Map `raw_result["processing_report_path"]` into `ProcessResult`.

### Tests

New tests:

- `tests/core/test_processing_report.py`
  - builder creates required top-level keys;
  - region stats aggregate masks correctly;
  - writer creates stable JSON;
  - missing optional diagnostics produce empty/default sections, not errors.
- `tests/core/test_pipeline_quality_report.py`
  - pipeline writes `processing_report.json` even when `output_debug=False`;
  - pipeline result includes `processing_report_path`;
  - report includes foreground recovery diagnostics when green-screen decontamination runs.
- `tests/core/test_service.py`
  - service maps `processing_report_path` into `ProcessResult`.

## Data Flow

```text
MattingPipeline.process()
  -> decode frames
  -> analyze background
  -> generate alpha
  -> refine/despeckle/repair/stabilize
  -> build region/decontamination contexts
  -> decontaminate frames
  -> encode outputs
  -> ProcessingReportBuilder.build(report inputs)
  -> ProcessingReportWriter.write(report, output_dir)
  -> return raw_result with processing_report_path
  -> MatteFlowService maps report path to ProcessResult
```

## Error Handling

Report generation must not fail the primary processing job.

Rules:

- If report building fails, log the exception and write a minimal report with a warning when possible.
- If report writing fails due to permission or IO errors, log the exception and return the processing result without `processing_report_path`.
- Do not raise report-specific exceptions from `MattingPipeline.process()` unless the primary output encoding also fails.

Minimal fallback report:

```json
{
  "schema_version": 1,
  "job": {},
  "timings": {},
  "quality": {},
  "regions": {},
  "model_decisions": {},
  "fusion": {"available": false},
  "foreground_recovery": {},
  "artifacts": {},
  "warnings": ["processing report generation failed: <message>"]
}
```

## Backward Compatibility

- Existing output files remain unchanged.
- Existing debug overlay behavior remains gated by `output_debug`.
- Existing tests that inspect `timings` or raw pipeline result should continue to pass.
- The report path is additive in pipeline and service results.

## Acceptance Criteria

- Every successful pipeline run writes `<output_dir>/processing_report.json`.
- `ProcessResult.processing_report_path` is populated when the report is written.
- Report JSON includes all required top-level sections.
- The report contains alpha quality metrics and region ownership pixel counts.
- Green-screen runs include foreground recovery diagnostics when available.
- Report generation failure does not fail the processing job.
- Full test suite passes.

## Implementation Notes

- Use only standard library JSON serialization.
- Convert NumPy scalars and arrays to native Python types before writing JSON.
- Use relative artifact paths where possible to keep reports portable.
- Keep schema version at integer `1`.
- Prefer explicit dictionaries for report sections over deeply nested dataclasses; this keeps evolution cheaper.

## Open Questions

- Whether `processing_report.json` should always be written for preview jobs as well as final export jobs. Recommended default: write it for every pipeline process call because previews are also useful for debugging.
- Whether GUI should expose a download link for the report in P3-C. Recommended default: yes, but out of scope for P3-A.
