# Changelog

## 1.0.0

### Added

- Unified CLI for image, sequence, and video matting with green-screen, black-background, and auto background modes.
- Quality preset support with `--quality-preset default` for compatibility and `--quality-preset best` for high-quality multi-candidate selection.
- Quality-driven matte selection with candidate ranking, region-level scoring, skipped-candidate diagnostics, and `processing_report.json` summaries.
- Black-background effect enhancement for smoke, glow, weak particles, and subject dark-edge continuity repair.
- Region weak supervision with `region_expectations` support for quality regression manifests.
- Edge reconstruction diagnostics and protected-region handling for transparent effects and luminous props.
- `quality-regression --reports` CLI command for aggregating `processing_report.json` outputs.
- Asset manifest update script for registering samples from `assets/`.

### Changed

- `processing_report.json` schema version is now `2`.
- Black-background processing now delegates effect preservation to the unified `BlackEffectEnhancer`.
- Unknown-background traditional fallback now guards against obvious green-screen samples being routed to black-background fallback.
- Quality regression now validates real selected-model contribution for `required_temporal_models`.

### Documentation

- Updated README CLI parameter descriptions, default values, report fields, and quality-regression examples.
- Updated technical route and architecture documentation for the current pipeline, quality selection flow, black-background enhancement, and report schema.

### Verification

- Core test suite, Ruff linting, and Python compilation are the required release gates for this version.
