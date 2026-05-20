# P0-3 Diagnostics Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a structured diagnostics center for `MatteFlow` so environment checks, model availability checks, processing failures, and GUI error presentation all flow through one stable `DiagnosticReport` contract.

**Architecture:** Add a new `src/matteflow/diagnostics.py` module as the single translation layer between raw facts and user-facing diagnostics. Keep `ffmpeg_env.py`, `model_checker.py`, and `service.py` focused on facts or typed exceptions, then adapt `scripts/web_gui.py` to render diagnostics by severity instead of relying on ad-hoc strings.

**Tech Stack:** Python 3.10+, dataclasses, `Enum`, `pytest`, Gradio, existing `MatteFlowService`, existing model/runtime discovery utilities.

---

## File Map

### New file

- `src/matteflow/diagnostics.py`
  - Owns `DiagnosticSeverity`, `DiagnosticCode`, `DiagnosticItem`, `DiagnosticReport`
  - Owns `from_media_tools()`, `from_model_status()`, `from_exception()`, `merge_reports()`, `report_to_user_text()`

### Existing files to modify

- `src/matteflow/ffmpeg_env.py`
  - Keep media-tool discovery as fact collection
  - Add only the minimal fact fields needed by diagnostics mapping

- `src/matteflow/utils/model_checker.py`
  - Keep current `check_all_models()` and `get_ui_choices()`
  - Add a diagnostics-oriented fact interface that is stable for tests and for `diagnostics.py`

- `src/matteflow/service.py`
  - Keep `ProcessingError` as the service-facing typed error
  - Route runtime exceptions through diagnostics mapping

- `scripts/web_gui.py`
  - Add helpers to collect environment diagnostics
  - Add helpers to format `DiagnosticReport` for GUI status/error regions
  - Keep existing workflow and queue logic intact

### Test files

- Create: `tests/test_diagnostics.py`
- Modify: `tests/test_model_checker_runtime.py`
- Modify: `tests/test_service.py`
- Modify: `tests/test_web_gui_defaults.py`
- Keep green: `tests/test_ffmpeg_env.py`

## Implementation Order

1. Create the diagnostics core and tests first
2. Integrate FFmpeg/media-tool mapping
3. Add model-checker fact export and mapping
4. Route service exceptions through diagnostics
5. Adapt Web GUI presentation and run focused regressions

### Task 1: Create Diagnostics Core

**Files:**
- Create: `src/matteflow/diagnostics.py`
- Test: `tests/test_diagnostics.py`

- [ ] **Step 1: Write the failing diagnostics-core tests**

Add `tests/test_diagnostics.py` with these starter tests:

```python
from pathlib import Path

from matteflow.diagnostics import (
    DiagnosticCode,
    DiagnosticSeverity,
    from_exception,
    from_media_tools,
    merge_reports,
)
from matteflow.errors import ProcessingError
from matteflow.ffmpeg_env import MediaToolDiscoveryResult


def test_from_media_tools_flags_missing_ffprobe():
    report = from_media_tools(
        MediaToolDiscoveryResult(
            ffmpeg_path="C:/ffmpeg/bin/ffmpeg.exe",
            ffprobe_path=None,
            bin_dir="C:/ffmpeg/bin",
            source="imageio_ffmpeg",
            complete=False,
            download_required=True,
        )
    )

    assert report.ok is False
    assert report.blocking_count == 1
    assert report.items[0].code is DiagnosticCode.FFPROBE_NOT_FOUND
    assert report.items[0].severity is DiagnosticSeverity.ERROR


def test_from_media_tools_flags_missing_toolchain():
    report = from_media_tools(
        MediaToolDiscoveryResult(
            ffmpeg_path=None,
            ffprobe_path=None,
            bin_dir=None,
            source=None,
            complete=False,
            download_required=True,
        )
    )

    assert report.ok is False
    assert report.items[0].code is DiagnosticCode.FFMPEG_NOT_FOUND


def test_from_exception_maps_cuda_oom():
    report = from_exception(RuntimeError("CUDA out of memory"))

    assert report.ok is False
    assert report.items[0].code is DiagnosticCode.GPU_OUT_OF_MEMORY
    assert report.items[0].blocking is True


def test_from_exception_maps_unknown_processing_error():
    report = from_exception(ProcessingError("boom"))

    assert report.items[0].code is DiagnosticCode.UNKNOWN_PROCESSING_ERROR


def test_merge_reports_deduplicates_by_code_and_summary():
    report_a = from_exception(RuntimeError("CUDA out of memory"))
    report_b = from_exception(RuntimeError("CUDA out of memory"))

    merged = merge_reports(report_a, report_b)

    assert len(merged.items) == 1
    assert merged.blocking_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_diagnostics.py -v
```

Expected:

```text
E   ModuleNotFoundError: No module named 'matteflow.diagnostics'
```

- [ ] **Step 3: Write the minimal diagnostics core**

Create `src/matteflow/diagnostics.py` with the following initial implementation:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class DiagnosticSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DiagnosticCode(str, Enum):
    FFMPEG_NOT_FOUND = "FFMPEG_NOT_FOUND"
    FFPROBE_NOT_FOUND = "FFPROBE_NOT_FOUND"
    FFMPEG_INCOMPLETE_TOOLCHAIN = "FFMPEG_INCOMPLETE_TOOLCHAIN"
    MODEL_MISSING = "MODEL_MISSING"
    MODEL_RUNTIME_IMPORT_FAILED = "MODEL_RUNTIME_IMPORT_FAILED"
    MODEL_GPU_REQUIRED = "MODEL_GPU_REQUIRED"
    GPU_OUT_OF_MEMORY = "GPU_OUT_OF_MEMORY"
    INPUT_INVALID = "INPUT_INVALID"
    OUTPUT_DIR_UNWRITABLE = "OUTPUT_DIR_UNWRITABLE"
    UNKNOWN_PROCESSING_ERROR = "UNKNOWN_PROCESSING_ERROR"


@dataclass(frozen=True)
class DiagnosticItem:
    code: DiagnosticCode
    severity: DiagnosticSeverity
    title: str
    summary: str
    details: str | None = None
    actions: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    blocking: bool = True


@dataclass(frozen=True)
class DiagnosticReport:
    items: tuple[DiagnosticItem, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(item.blocking for item in self.items)

    @property
    def blocking_count(self) -> int:
        return sum(1 for item in self.items if item.blocking)

    @property
    def warning_count(self) -> int:
        return sum(1 for item in self.items if item.severity is DiagnosticSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for item in self.items if item.severity is DiagnosticSeverity.INFO)


def from_media_tools(discovery_result) -> DiagnosticReport:
    if discovery_result.ffmpeg_path is None:
        return DiagnosticReport(
            items=(
                DiagnosticItem(
                    code=DiagnosticCode.FFMPEG_NOT_FOUND,
                    severity=DiagnosticSeverity.ERROR,
                    title="未找到 FFmpeg",
                    summary="未检测到可用的 ffmpeg/ffprobe 工具链。",
                    actions=("运行 configure_ffmpeg.ps1", "检查 PATH 中的 ffmpeg 安装"),
                    evidence={"source": discovery_result.source},
                    blocking=True,
                ),
            )
        )
    if discovery_result.ffprobe_path is None:
        return DiagnosticReport(
            items=(
                DiagnosticItem(
                    code=DiagnosticCode.FFPROBE_NOT_FOUND,
                    severity=DiagnosticSeverity.ERROR,
                    title="FFprobe 缺失",
                    summary="已找到 ffmpeg，但缺少 ffprobe，当前工具链不完整。",
                    actions=("下载完整 FFmpeg 套件", "重新运行 FFmpeg bootstrap"),
                    evidence={"ffmpeg_path": discovery_result.ffmpeg_path, "source": discovery_result.source},
                    blocking=True,
                ),
            )
        )
    return DiagnosticReport()


def from_exception(exc: Exception, context: dict[str, Any] | None = None) -> DiagnosticReport:
    raw_message = str(exc)
    lowered = raw_message.lower()
    if "cuda out of memory" in lowered or "outofmemory" in lowered:
        return DiagnosticReport(
            items=(
                DiagnosticItem(
                    code=DiagnosticCode.GPU_OUT_OF_MEMORY,
                    severity=DiagnosticSeverity.ERROR,
                    title="GPU 显存不足",
                    summary="当前任务执行时显存不足，无法继续处理。",
                    actions=("关闭其他 GPU 程序", "降低质量或分辨率", "切换到更轻量模型"),
                    evidence={"message": raw_message, "context": context or {}},
                    blocking=True,
                ),
            )
        )
    return DiagnosticReport(
        items=(
            DiagnosticItem(
                code=DiagnosticCode.UNKNOWN_PROCESSING_ERROR,
                severity=DiagnosticSeverity.ERROR,
                title="处理失败",
                summary="MatteFlow 在处理任务时发生未分类错误。",
                actions=("查看日志输出", "检查输入素材和模型状态"),
                evidence={"message": raw_message, "context": context or {}},
                blocking=True,
            ),
        )
    )


def merge_reports(*reports: DiagnosticReport) -> DiagnosticReport:
    seen: set[tuple[str, str]] = set()
    merged: list[DiagnosticItem] = []
    for report in reports:
        for item in report.items:
            key = (item.code.value, item.summary)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return DiagnosticReport(items=tuple(merged))


def report_to_user_text(report: DiagnosticReport) -> str:
    if not report.items:
        return "未检测到阻断问题。"
    lines: list[str] = []
    for item in report.items:
        lines.append(f"[{item.severity.value}] {item.title}: {item.summary}")
        for action in item.actions[:3]:
            lines.append(f"- {action}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify the diagnostics core passes**

Run:

```bash
pytest tests/test_diagnostics.py -v
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Commit**

```bash
git add src/matteflow/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: add diagnostics core models"
```

### Task 2: Integrate Media-Tool Diagnostics

**Files:**
- Modify: `src/matteflow/ffmpeg_env.py`
- Test: `tests/test_ffmpeg_env.py`
- Test: `tests/test_diagnostics.py`

- [ ] **Step 1: Add a failing media-tool diagnostics test**

Append this test to `tests/test_diagnostics.py`:

```python
def test_from_media_tools_accepts_complete_toolchain():
    report = from_media_tools(
        MediaToolDiscoveryResult(
            ffmpeg_path="C:/ffmpeg/bin/ffmpeg.exe",
            ffprobe_path="C:/ffmpeg/bin/ffprobe.exe",
            bin_dir="C:/ffmpeg/bin",
            source="path",
            complete=True,
            download_required=False,
        )
    )

    assert report.ok is True
    assert report.items == ()
```

Append this discovery-fact test to `tests/test_ffmpeg_env.py`:

```python
def test_discover_media_tools_marks_imageio_ffmpeg_as_incomplete(monkeypatch):
    result = discover_media_tools(
        ffmpeg_which=lambda _: None,
        ffprobe_which=lambda _: None,
        path_exists=lambda path: path.endswith("ffmpeg.exe"),
        imageio_ffmpeg_getter=lambda: "C:/imageio/ffmpeg.exe",
        common_candidate_dirs=[],
    )

    assert result.ffmpeg_path == "C:/imageio/ffmpeg.exe"
    assert result.ffprobe_path is None
    assert result.complete is False
    assert result.download_required is True
```

- [ ] **Step 2: Run the focused tests to verify current behavior**

Run:

```bash
pytest tests/test_diagnostics.py::test_from_media_tools_accepts_complete_toolchain tests/test_ffmpeg_env.py::test_discover_media_tools_marks_imageio_ffmpeg_as_incomplete -v
```

Expected:

```text
FAIL at least one assertion if the discovery facts or complete-toolchain mapping are not stable yet
```

- [ ] **Step 3: Tighten `ffmpeg_env.py` only at the fact layer**

Keep the API shape stable and add only the minimal normalization needed:

```python
@dataclass(frozen=True)
class MediaToolDiscoveryResult:
    ffmpeg_path: str | None
    ffprobe_path: str | None
    bin_dir: str | None
    source: str | None
    complete: bool
    download_required: bool

    @property
    def missing_ffmpeg(self) -> bool:
        return self.ffmpeg_path is None

    @property
    def missing_ffprobe(self) -> bool:
        return self.ffmpeg_path is not None and self.ffprobe_path is None
```

Then make `from_media_tools()` explicitly handle the complete case:

```python
def from_media_tools(discovery_result) -> DiagnosticReport:
    if discovery_result.complete:
        return DiagnosticReport()
    if discovery_result.missing_ffmpeg:
        ...
    if discovery_result.missing_ffprobe:
        ...
```

- [ ] **Step 4: Run media-tool regression tests**

Run:

```bash
pytest tests/test_ffmpeg_env.py tests/test_diagnostics.py -v
```

Expected:

```text
all tests in both files pass
```

- [ ] **Step 5: Commit**

```bash
git add src/matteflow/ffmpeg_env.py src/matteflow/diagnostics.py tests/test_ffmpeg_env.py tests/test_diagnostics.py
git commit -m "feat: map media tool discovery into diagnostics"
```

### Task 3: Add Model-Checker Fact Export and Diagnostics Mapping

**Files:**
- Modify: `src/matteflow/utils/model_checker.py`
- Modify: `src/matteflow/diagnostics.py`
- Test: `tests/test_model_checker_runtime.py`
- Test: `tests/test_diagnostics.py`

- [ ] **Step 1: Write failing model-diagnostics tests**

Append this test to `tests/test_model_checker_runtime.py`:

```python
def test_collect_model_facts_exposes_reason_and_path(monkeypatch, tmp_path):
    checker = ModelChecker()
    checker.cache_dir = tmp_path
    checker.matteflow_dir = tmp_path

    facts = checker.collect_model_facts()

    assert "gvm" in facts
    assert set(facts["gvm"]) >= {"available", "path", "reason", "display_name", "auto_download"}
```

Append these tests to `tests/test_diagnostics.py`:

```python
from matteflow.diagnostics import from_model_status


def test_from_model_status_maps_missing_model():
    report = from_model_status(
        {
            "corridorkey": {
                "display_name": "CorridorKey",
                "available": False,
                "path": "C:/models/corridorkey.pth",
                "reason": "需要手动下载",
                "auto_download": False,
            }
        }
    )

    assert report.ok is False
    assert report.items[0].code is DiagnosticCode.MODEL_MISSING


def test_from_model_status_maps_runtime_import_failure():
    report = from_model_status(
        {
            "gvm": {
                "display_name": "GVM",
                "available": False,
                "path": "C:/models/gvm",
                "reason": "GVM vendored runtime 不可导入",
                "auto_download": False,
            }
        }
    )

    assert report.items[0].code is DiagnosticCode.MODEL_RUNTIME_IMPORT_FAILED
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_model_checker_runtime.py::test_collect_model_facts_exposes_reason_and_path tests/test_diagnostics.py::test_from_model_status_maps_missing_model tests/test_diagnostics.py::test_from_model_status_maps_runtime_import_failure -v
```

Expected:

```text
E   AttributeError: 'ModelChecker' object has no attribute 'collect_model_facts'
```

- [ ] **Step 3: Add a diagnostics-oriented fact interface**

In `src/matteflow/utils/model_checker.py`, add:

```python
class ModelChecker:
    ...
    def collect_model_facts(self) -> Dict[str, Dict]:
        results = self.check_all_models()
        facts: Dict[str, Dict] = {}
        for key, info in results.items():
            facts[key] = {
                "model_key": key,
                "display_name": info["name"],
                "available": bool(info["available"]),
                "path": info["path"],
                "reason": info["reason"],
                "auto_download": bool(info["auto_download"]),
            }
        return facts
```

In `src/matteflow/diagnostics.py`, add:

```python
def from_model_status(model_status_dict: dict[str, dict[str, Any]]) -> DiagnosticReport:
    items: list[DiagnosticItem] = []
    for model_key, info in model_status_dict.items():
        if info.get("available"):
            continue
        reason = info.get("reason") or ""
        if "不可导入" in reason:
            code = DiagnosticCode.MODEL_RUNTIME_IMPORT_FAILED
            title = f"{info['display_name']} runtime 不可用"
            summary = f"{info['display_name']} 已安装，但运行时依赖无法导入。"
        elif "仅支持 CUDA" in reason:
            code = DiagnosticCode.MODEL_GPU_REQUIRED
            title = f"{info['display_name']} 需要 CUDA"
            summary = f"{info['display_name']} 仅支持 CUDA GPU。"
        else:
            code = DiagnosticCode.MODEL_MISSING
            title = f"{info['display_name']} 不可用"
            summary = f"{info['display_name']} 当前不可用，通常是模型文件缺失。"
        items.append(
            DiagnosticItem(
                code=code,
                severity=DiagnosticSeverity.WARNING if info.get("auto_download") else DiagnosticSeverity.ERROR,
                title=title,
                summary=summary,
                actions=("检查模型目录", "确认依赖是否已安装"),
                evidence={"model_key": model_key, "path": info.get("path"), "reason": reason},
                blocking=not bool(info.get("auto_download")),
            )
        )
    return DiagnosticReport(items=tuple(items))
```

- [ ] **Step 4: Run model-checker and diagnostics tests**

Run:

```bash
pytest tests/test_model_checker_runtime.py tests/test_diagnostics.py -v
```

Expected:

```text
all tests in both files pass
```

- [ ] **Step 5: Commit**

```bash
git add src/matteflow/utils/model_checker.py src/matteflow/diagnostics.py tests/test_model_checker_runtime.py tests/test_diagnostics.py
git commit -m "feat: add model diagnostics fact mapping"
```

### Task 4: Route Service Exceptions Through Diagnostics

**Files:**
- Modify: `src/matteflow/service.py`
- Modify: `src/matteflow/diagnostics.py`
- Test: `tests/test_service.py`
- Test: `tests/test_diagnostics.py`

- [ ] **Step 1: Write failing service-level diagnostics tests**

Append these tests to `tests/test_service.py`:

```python
from matteflow.diagnostics import DiagnosticCode, from_exception


def test_service_unknown_pipeline_error_stays_wrapped_and_mappable(tmp_path):
    class FailingPipeline:
        def __init__(self, config):
            pass

        def process(self, input_path, output_dir, progress_callback=None):
            raise RuntimeError("decoder exploded")

    service = MatteFlowService(pipeline_factory=FailingPipeline)
    params = ProcessJobParams(input_path=tmp_path / "input.png", output_dir=tmp_path / "out")

    with pytest.raises(ProcessingError) as exc_info:
        service.process(params)

    report = from_exception(exc_info.value, context={"stage": "process"})
    assert report.items[0].code is DiagnosticCode.UNKNOWN_PROCESSING_ERROR


def test_service_oom_error_maps_to_gpu_out_of_memory(tmp_path):
    class FailingPipeline:
        def __init__(self, config):
            pass

        def process(self, input_path, output_dir, progress_callback=None):
            raise RuntimeError("CUDA out of memory")

    service = MatteFlowService(pipeline_factory=FailingPipeline)
    params = ProcessJobParams(input_path=tmp_path / "input.png", output_dir=tmp_path / "out")

    with pytest.raises(ProcessingError) as exc_info:
        service.process(params)

    report = from_exception(exc_info.value, context={"stage": "process"})
    assert report.items[0].code is DiagnosticCode.GPU_OUT_OF_MEMORY
```

- [ ] **Step 2: Run the focused service tests**

Run:

```bash
pytest tests/test_service.py::test_service_unknown_pipeline_error_stays_wrapped_and_mappable tests/test_service.py::test_service_oom_error_maps_to_gpu_out_of_memory -v
```

Expected:

```text
FAIL if the wrapper text loses the original signal or if exception mapping is incomplete
```

- [ ] **Step 3: Keep the service API stable and add diagnostics hooks**

In `src/matteflow/service.py`, keep `ProcessingError` as the public exception, but preserve richer mapping context:

```python
from .diagnostics import from_exception
...
    @staticmethod
    def _format_processing_error(exc: Exception) -> str:
        report = from_exception(exc, context={"stage": "process"})
        first_item = report.items[0]
        return f"{first_item.title}: {first_item.summary} Original error: {exc}"
```

Do not add repair logic here. Do not add GUI formatting here. Only ensure:

- OOM messages still mention GPU memory
- unknown failures still preserve the original raw error text
- diagnostics mapping can consume `ProcessingError` and raw runtime exceptions consistently

- [ ] **Step 4: Run all service and diagnostics tests**

Run:

```bash
pytest tests/test_service.py tests/test_diagnostics.py -v
```

Expected:

```text
all tests in both files pass
```

- [ ] **Step 5: Commit**

```bash
git add src/matteflow/service.py src/matteflow/diagnostics.py tests/test_service.py tests/test_diagnostics.py
git commit -m "feat: route service failures through diagnostics"
```

### Task 5: Adapt Web GUI to Consume Diagnostic Reports

**Files:**
- Modify: `scripts/web_gui.py`
- Modify: `tests/test_web_gui_defaults.py`
- Optional modify: `tests/test_web_gui_preview.py`

- [ ] **Step 1: Write failing GUI diagnostics tests**

Append these tests to `tests/test_web_gui_defaults.py`:

```python
from matteflow.diagnostics import DiagnosticCode, DiagnosticItem, DiagnosticReport, DiagnosticSeverity


def test_format_diagnostic_report_prioritizes_errors():
    report = DiagnosticReport(
        items=(
            DiagnosticItem(
                code=DiagnosticCode.GPU_OUT_OF_MEMORY,
                severity=DiagnosticSeverity.ERROR,
                title="GPU 显存不足",
                summary="当前任务执行时显存不足。",
                actions=("关闭其他 GPU 程序",),
                blocking=True,
            ),
            DiagnosticItem(
                code=DiagnosticCode.MODEL_MISSING,
                severity=DiagnosticSeverity.WARNING,
                title="模型缺失",
                summary="部分模型当前不可用。",
                actions=("检查模型目录",),
                blocking=False,
            ),
        )
    )

    text = web_gui._format_diagnostic_report(report)

    assert "GPU 显存不足" in text
    assert "关闭其他 GPU 程序" in text
    assert "模型缺失" in text


def test_collect_environment_diagnostics_returns_blocking_report(monkeypatch):
    from matteflow.ffmpeg_env import MediaToolDiscoveryResult

    monkeypatch.setattr(
        web_gui,
        "discover_media_tools",
        lambda: MediaToolDiscoveryResult(
            ffmpeg_path=None,
            ffprobe_path=None,
            bin_dir=None,
            source=None,
            complete=False,
            download_required=True,
        ),
    )

    class FakeChecker:
        def collect_model_facts(self):
            return {}

    report = web_gui._collect_environment_diagnostics(model_checker=FakeChecker())

    assert report.ok is False
    assert report.blocking_count == 1
```

- [ ] **Step 2: Run the focused GUI tests**

Run:

```bash
pytest tests/test_web_gui_defaults.py::test_format_diagnostic_report_prioritizes_errors tests/test_web_gui_defaults.py::test_collect_environment_diagnostics_returns_blocking_report -v
```

Expected:

```text
E   AttributeError: module 'scripts.web_gui' has no attribute '_format_diagnostic_report'
```

- [ ] **Step 3: Add GUI diagnostics helpers without rewriting the page**

In `scripts/web_gui.py`, add these imports near the top:

```python
from matteflow.diagnostics import (
    DiagnosticReport,
    from_media_tools,
    from_model_status,
    from_exception,
    merge_reports,
)
from matteflow.ffmpeg_env import discover_media_tools
```

Add these helpers:

```python
def _collect_environment_diagnostics(model_checker=None) -> DiagnosticReport:
    checker = model_checker or _model_checker
    media_report = from_media_tools(discover_media_tools())
    model_report = from_model_status(checker.collect_model_facts())
    return merge_reports(media_report, model_report)


def _format_diagnostic_report(report: DiagnosticReport) -> str:
    if not report.items:
        return "未检测到阻断问题。"
    lines = []
    for item in report.items:
        prefix = {
            "error": "ERROR",
            "warning": "WARNING",
            "info": "INFO",
        }[item.severity.value]
        lines.append(f"**{prefix}** {item.title}")
        lines.append(item.summary)
        for action in item.actions[:3]:
            lines.append(f"- {action}")
    return "\n".join(lines)
```

Then wire it into existing failure paths:

```python
    try:
        result = service.process(...)
    except Exception as exc:
        report = from_exception(exc, context={"stage": "process_video"})
        return None, None, _format_diagnostic_report(report), ...
```

For pre-run environment checks, do not block import-time module loading. Call `_collect_environment_diagnostics()` inside runtime handlers such as `process_video()` or a dedicated status refresh helper.

- [ ] **Step 4: Run GUI-focused regressions**

Run:

```bash
pytest tests/test_web_gui_defaults.py tests/test_service.py tests/test_diagnostics.py -v
```

Expected:

```text
all tests in the focused GUI/service/diagnostics set pass
```

- [ ] **Step 5: Commit**

```bash
git add scripts/web_gui.py tests/test_web_gui_defaults.py tests/test_service.py tests/test_diagnostics.py
git commit -m "feat: surface structured diagnostics in web gui"
```

### Task 6: Final Regression Sweep and Documentation Check

**Files:**
- Verify only: `src/matteflow/diagnostics.py`
- Verify only: `src/matteflow/ffmpeg_env.py`
- Verify only: `src/matteflow/utils/model_checker.py`
- Verify only: `src/matteflow/service.py`
- Verify only: `scripts/web_gui.py`
- Verify only: `docs/superpowers/specs/2026-05-20-p0-3-diagnostics-center-design.md`

- [ ] **Step 1: Run the full focused regression bundle**

Run:

```bash
pytest tests/test_ffmpeg_env.py tests/test_model_checker_runtime.py tests/test_service.py tests/test_web_gui_defaults.py tests/test_diagnostics.py -v
```

Expected:

```text
all selected tests pass
```

- [ ] **Step 2: Run editor diagnostics on edited files**

Check diagnostics for:

```text
src/matteflow/diagnostics.py
src/matteflow/ffmpeg_env.py
src/matteflow/utils/model_checker.py
src/matteflow/service.py
scripts/web_gui.py
```

Expected:

```text
no new linter or type diagnostics introduced by the diagnostics-center changes
```

- [ ] **Step 3: Manual acceptance check**

Verify these manual outcomes:

```text
1. Missing ffmpeg/ffprobe no longer shows only raw traceback text.
2. Missing model/runtime issues can be summarized as structured GUI messages.
3. CUDA OOM path shows an actionable message with next steps.
4. Complete ffmpeg + ffprobe toolchain yields no blocking diagnostics.
5. Existing queue and result-summary behavior stays unchanged.
```

- [ ] **Step 4: Commit the final integration sweep**

```bash
git add src/matteflow/diagnostics.py src/matteflow/ffmpeg_env.py src/matteflow/utils/model_checker.py src/matteflow/service.py scripts/web_gui.py tests/test_ffmpeg_env.py tests/test_model_checker_runtime.py tests/test_service.py tests/test_web_gui_defaults.py tests/test_diagnostics.py
git commit -m "test: verify diagnostics center integration"
```

## Spec Coverage Check

- Structured diagnostics model: covered by Task 1
- Media-tool diagnostics mapping: covered by Task 2
- Model-fact export and model diagnostics mapping: covered by Task 3
- Service exception mapping: covered by Task 4
- GUI severity-based presentation: covered by Task 5
- Focused regression and acceptance: covered by Task 6
- `P0-4` boundary kept intact: no task introduces repair, startup wizard, or first-run flow

## Notes for the Implementer

- Keep `ffmpeg_env.py` focused on discovery facts; do not move user-facing strings there
- Keep `model_checker.py` backward compatible for `get_ui_choices()`
- Keep `service.py` free of GUI formatting or repair operations
- In `web_gui.py`, prefer helper functions instead of spreading diagnostics formatting across multiple branches
- If a test forces a broader UI refactor, stop and split the work before proceeding
