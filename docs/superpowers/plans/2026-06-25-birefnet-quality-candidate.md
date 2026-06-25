# BiRefNet Quality Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 BiRefNet 从结构化 skip 推进到显式 opt-in 的真实 quality candidate，并保持默认不自动下载模型。

**Architecture:** 在 `BiRefNetCandidateGenerator` 内增加保守的懒加载能力，通过 `MattingConfig.quality_birefnet_auto_load` 和 CLI 参数显式开启。默认路径仍不加载外部模型；加载失败或推理失败时写入 `skipped_candidates`，由 `traditional` 候选兜底。

**Tech Stack:** Python、NumPy、pytest、现有 MatteFlow `BiRefNetMatte`、`QualityDrivenMatte`、`ProcessingReport.quality_selection`。

---

## 文件结构

- 修改 `src/matteflow/config.py`
  - 增加 `quality_birefnet_auto_load: bool = False`。
- 修改 `src/matteflow/cli_app.py`
  - 增加 `--quality-birefnet-auto-load` 参数，并写入 config。
- 修改 `src/matteflow/matte/candidates/birefnet.py`
  - 增加懒加载、engine factory、加载失败 skip、推理失败 skip。
- 修改 `src/matteflow/matte/quality_driven_matte.py`
  - 如有需要，将 engine factory 注入点保持可测试；默认无需改动主流程。
- 修改 `tests/core/test_candidate_contracts.py`
  - 增加 config 默认值测试。
- 修改或新增 `tests/core/test_birefnet_candidate.py`
  - 覆盖 BiRefNet generator 的默认 skip、fake engine、auto-load、失败降级。
- 修改 `tests/core/test_quality_driven_matte.py`
  - 覆盖 BiRefNet 失败时 traditional fallback 仍可用。
- 修改 `tests/core/test_cli_app.py`
  - 覆盖 CLI 参数解析。

---

### Task 1: 增加 BiRefNet auto-load 配置字段

**Files:**
- Modify: `src/matteflow/config.py`
- Modify: `tests/core/test_candidate_contracts.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_candidate_contracts.py` 增加：

```python
def test_quality_birefnet_auto_load_defaults_to_false():
    config = MattingConfig()

    assert config.quality_birefnet_auto_load is False
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```text
python -m pytest tests\core\test_candidate_contracts.py::test_quality_birefnet_auto_load_defaults_to_false -q
```

Expected: FAIL，提示 `MattingConfig` 没有 `quality_birefnet_auto_load`。

- [ ] **Step 3: 实现配置字段**

在 `src/matteflow/config.py` 的质量选择系统字段附近增加：

```python
quality_birefnet_auto_load: bool = False
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```text
python -m pytest tests\core\test_candidate_contracts.py::test_quality_birefnet_auto_load_defaults_to_false -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```text
git add src/matteflow/config.py tests/core/test_candidate_contracts.py
git commit -m "feat: add BiRefNet quality candidate config"
```

---

### Task 2: 扩展 BiRefNetCandidateGenerator 的懒加载和失败降级

**Files:**
- Modify: `src/matteflow/matte/candidates/birefnet.py`
- Create: `tests/core/test_birefnet_candidate.py`

- [ ] **Step 1: 写默认不自动加载的失败测试**

创建 `tests/core/test_birefnet_candidate.py`：

```python
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import MattingConfig  # noqa: E402
from matteflow.matte.candidates.birefnet import BiRefNetCandidateGenerator  # noqa: E402
from matteflow.matte.candidates.types import CandidateSkipReason  # noqa: E402


def test_birefnet_candidate_skips_without_engine_or_auto_load():
    generator = BiRefNetCandidateGenerator(MattingConfig())

    result = generator.generate(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        frame_shapes=[(2, 2)],
    )

    assert result.skipped is True
    assert result.skip_reason == CandidateSkipReason.MODEL_UNAVAILABLE
    assert "auto-load" in result.message
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```text
python -m pytest tests\core\test_birefnet_candidate.py::test_birefnet_candidate_skips_without_engine_or_auto_load -q
```

Expected: FAIL，message 仍是旧的 `"BiRefNet candidate engine is not available"`。

- [ ] **Step 3: 写 fake engine 可生成候选的测试**

在同一文件增加：

```python
class FakeBiRefNetEngine:
    model = object()

    def generate_sequence(self, frames, progress_callback=None):
        return [np.full(frame.shape[:2], 0.8, dtype=np.float32) for frame in frames]


def test_birefnet_candidate_generates_with_fake_engine():
    generator = BiRefNetCandidateGenerator(MattingConfig(), engine=FakeBiRefNetEngine())

    result = generator.generate(
        [np.zeros((3, 4, 3), dtype=np.uint8)],
        frame_shapes=[(3, 4)],
    )

    assert result.candidate is not None
    assert result.candidate.name == "birefnet"
    assert result.candidate.alphas[0].shape == (3, 4)
    assert float(result.candidate.alphas[0].mean()) == 0.8
```

- [ ] **Step 4: 写 auto-load 调用 factory 的测试**

```python
def test_birefnet_candidate_auto_loads_engine_when_enabled():
    config = MattingConfig()
    config.quality_birefnet_auto_load = True
    calls = []

    def factory(factory_config):
        calls.append(factory_config)
        return FakeBiRefNetEngine()

    generator = BiRefNetCandidateGenerator(config, engine_factory=factory)

    result = generator.generate(
        [np.zeros((2, 3, 3), dtype=np.uint8)],
        frame_shapes=[(2, 3)],
    )

    assert calls == [config]
    assert result.candidate is not None
    assert result.candidate.diagnostics["model"] == "birefnet"
```

- [ ] **Step 5: 写 auto-load 失败的测试**

```python
def test_birefnet_candidate_skips_when_auto_load_fails():
    config = MattingConfig()
    config.quality_birefnet_auto_load = True

    def factory(factory_config):
        raise RuntimeError("missing weights")

    generator = BiRefNetCandidateGenerator(config, engine_factory=factory)

    result = generator.generate(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        frame_shapes=[(2, 2)],
    )

    assert result.skipped is True
    assert result.skip_reason == CandidateSkipReason.MODEL_UNAVAILABLE
    assert "missing weights" in result.message
```

- [ ] **Step 6: 写推理失败的测试**

```python
class FailingBiRefNetEngine:
    model = object()

    def generate_sequence(self, frames, progress_callback=None):
        raise RuntimeError("inference failed")


def test_birefnet_candidate_skips_when_inference_fails():
    generator = BiRefNetCandidateGenerator(MattingConfig(), engine=FailingBiRefNetEngine())

    result = generator.generate(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        frame_shapes=[(2, 2)],
    )

    assert result.skipped is True
    assert result.skip_reason == CandidateSkipReason.GENERATION_FAILED
    assert "inference failed" in result.message
```

- [ ] **Step 7: 实现 generator**

在 `src/matteflow/matte/candidates/birefnet.py` 中实现：

```python
class BiRefNetCandidateGenerator(TimedCandidateGenerator):
    name = "birefnet"
    source = "birefnet"

    def __init__(self, config: MattingConfig, engine=None, engine_factory=None):
        self.config = config
        self.engine = engine
        self.engine_factory = engine_factory or self._default_engine_factory

    def _default_engine_factory(self, config: MattingConfig):
        from ..birefnet_matte import BiRefNetMatte

        return BiRefNetMatte(config)

    def _ensure_engine(self) -> CandidateGenerationResult | None:
        if self.engine is not None:
            if getattr(self.engine, "model", None) is not None:
                return None
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message="BiRefNet candidate engine is not available",
            )

        if not getattr(self.config, "quality_birefnet_auto_load", False):
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message="BiRefNet candidate engine is not available; auto-load is disabled",
            )

        try:
            self.engine = self.engine_factory(self.config)
        except Exception as exc:
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message=f"BiRefNet auto-load failed: {exc}",
            )

        if getattr(self.engine, "model", None) is None:
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
                message="BiRefNet auto-load completed but model is unavailable",
            )
        return None
```

并在 `generate()` 开头调用：

```python
skip_result = self._ensure_engine()
if skip_result is not None:
    return skip_result
```

推理调用包裹：

```python
try:
    alphas = self.engine.generate_sequence(list(frames), **kwargs)
except Exception as exc:
    return CandidateGenerationResult(
        candidate=None,
        skipped=True,
        skip_reason=CandidateSkipReason.GENERATION_FAILED,
        message=str(exc),
    )
```

- [ ] **Step 8: 运行测试确认通过**

Run:

```text
python -m pytest tests\core\test_birefnet_candidate.py -q
```

Expected: PASS。

- [ ] **Step 9: 提交**

```text
git add src/matteflow/matte/candidates/birefnet.py tests/core/test_birefnet_candidate.py
git commit -m "feat: enable opt-in BiRefNet quality candidate loading"
```

---

### Task 3: 集成 QualityDrivenMatte 的 BiRefNet fallback 行为

**Files:**
- Modify: `tests/core/test_quality_driven_matte.py`
- Modify: `src/matteflow/matte/quality_driven_matte.py` only if injection seams are insufficient

- [ ] **Step 1: 写 BiRefNet 推理失败后 traditional fallback 的测试**

在 `tests/core/test_quality_driven_matte.py` 增加：

```python
class _FailingBiRefNetGenerator:
    name = "birefnet"

    def generate(self, frames, *, frame_shapes, cancel_check=None, progress_callback=None):
        from matteflow.matte.candidates.types import CandidateGenerationResult, CandidateSkipReason

        return CandidateGenerationResult(
            candidate=None,
            skipped=True,
            skip_reason=CandidateSkipReason.GENERATION_FAILED,
            message="birefnet failed",
        )


def test_quality_driven_matte_uses_traditional_when_birefnet_candidate_fails():
    config = MattingConfig()
    frames = [np.full((4, 4, 3), [0, 255, 0], dtype=np.uint8)]
    matte = QualityDrivenMatte(
        config,
        background_mode=BackgroundMode.GREEN_SCREEN,
        generators=[
            _FailingBiRefNetGenerator(),
            TraditionalCandidateGenerator(config, background_mode=BackgroundMode.GREEN_SCREEN),
        ],
    )

    alphas = matte.generate_sequence(frames)

    assert len(alphas) == 1
    assert matte.last_quality_selection["candidate_count"] == 1
    assert matte.last_quality_selection["skipped_candidates"][0]["name"] == "birefnet"
    assert matte.last_quality_selection["skipped_candidates"][0]["reason"] == "generation_failed"
    assert matte.last_quality_selection["selected_model_counts"]["traditional"] >= 1
```

- [ ] **Step 2: 运行测试**

Run:

```text
python -m pytest tests\core\test_quality_driven_matte.py::test_quality_driven_matte_uses_traditional_when_birefnet_candidate_fails -q
```

Expected: PASS。如果 import 或 generator seam 不足，按最小改动补齐。

- [ ] **Step 3: 运行相关集成测试**

Run:

```text
python -m pytest tests\core\test_quality_driven_matte.py tests\core\test_birefnet_candidate.py -q
```

Expected: PASS。

- [ ] **Step 4: 提交**

```text
git add tests/core/test_quality_driven_matte.py src/matteflow/matte/quality_driven_matte.py
git commit -m "test: cover BiRefNet quality fallback behavior"
```

---

### Task 4: 增加 CLI 参数

**Files:**
- Modify: `src/matteflow/cli_app.py`
- Modify: `tests/core/test_cli_app.py`

- [ ] **Step 1: 写 CLI config 测试**

在 `tests/core/test_cli_app.py` 中增加：

```python
def test_build_config_sets_quality_birefnet_auto_load():
    args = argparse.Namespace(
        mode="green",
        quality="standard",
        ai_model="auto",
        no_ai=False,
        quality_selection=True,
        quality_birefnet_auto_load=True,
        mask=False,
        debug=False,
    )

    config = _build_config(args)

    assert config.quality_selection_enable is True
    assert config.quality_birefnet_auto_load is True
```

如果现有 helper 使用不同的 Namespace 字段，按文件内已有测试字段补齐。

- [ ] **Step 2: 运行测试确认失败**

Run:

```text
python -m pytest tests\core\test_cli_app.py::test_build_config_sets_quality_birefnet_auto_load -q
```

Expected: FAIL，字段未写入 config。

- [ ] **Step 3: 实现 CLI 参数**

在 `_add_process_arguments()` 增加：

```python
parser.add_argument(
    "--quality-birefnet-auto-load",
    action="store_true",
    help="质量选择启用时允许 BiRefNet candidate 显式懒加载模型",
)
```

在 `_build_config()` 增加：

```python
config.quality_birefnet_auto_load = bool(getattr(args, "quality_birefnet_auto_load", False))
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```text
python -m pytest tests\core\test_cli_app.py::test_build_config_sets_quality_birefnet_auto_load -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```text
git add src/matteflow/cli_app.py tests/core/test_cli_app.py
git commit -m "feat: add BiRefNet quality candidate CLI flag"
```

---

### Task 5: 最终回归和文档补充

**Files:**
- Modify: `docs/technical_route_and_architecture.md`
- Optional Modify: `docs/superpowers/specs/2026-06-25-birefnet-quality-candidate-design.md`

- [ ] **Step 1: 文档补充**

在 `docs/technical_route_and_architecture.md` 的质量选择系统段落补充：

```markdown
BiRefNet 候选默认不会自动加载模型。需要真实 BiRefNet candidate 时，必须同时启用质量选择并显式打开 `quality_birefnet_auto_load` 或 CLI 参数 `--quality-birefnet-auto-load`。加载或推理失败会记录为 `skipped_candidates`，不影响 traditional fallback。
```

- [ ] **Step 2: 运行核心测试**

Run:

```text
python -m pytest tests\core -q
```

Expected: PASS。

- [ ] **Step 3: 运行 manifest 回归**

Run:

```text
$env:PYTHONPATH='e:\ByteDance\Projects\Code\MatteFlow\src'; python -c "from matteflow.evaluation.matting_quality_regression import MattingQualityRegressionManifest, MattingQualityRegressionRunner; manifest=MattingQualityRegressionManifest.from_path('tests/fixtures/matting_quality/manifest.json'); summary=MattingQualityRegressionRunner(manifest=manifest).run('temp/birefnet_quality_candidate_regression'); print(summary)"
```

Expected: 生成 `temp\birefnet_quality_candidate_regression\quality_summary.json`。

- [ ] **Step 4: 运行 quality-regression**

Run:

```text
python scripts\run_matting.py quality-regression --reports temp\birefnet_quality_candidate_regression --min-overall-score 0.0 --max-hole-pixels 100 --max-background-residue 1.0 --max-mean-edge-uncertainty 1.0 --max-temporal-flicker 1.0
```

Expected:

```text
Status: PASS
Failed: 0
```

- [ ] **Step 5: 提交**

```text
git add docs/technical_route_and_architecture.md
git commit -m "docs: document BiRefNet quality candidate opt-in"
```

---

## 自查

- Spec 覆盖：配置字段、CLI、generator 懒加载、失败降级、fallback、report 可观测性和测试策略都有任务对应。
- 默认安全：所有真实模型加载都要求 `quality_birefnet_auto_load=True`，普通测试不触发下载。
- 类型一致性：配置字段统一为 `quality_birefnet_auto_load`；skip reason 使用已有 `CandidateSkipReason.MODEL_UNAVAILABLE` 和 `CandidateSkipReason.GENERATION_FAILED`。
- 范围控制：本计划不接 MatAnyone2/SAM2，不做权重管理系统，不做真实模型 benchmark。
