# P0-4 最小版启动自检 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `MatteFlow` 增加第一阶段最小版启动自检能力，在启动时统一检查媒体工具、模型和默认输出目录，并将结构化诊断结果展示到 Web GUI 顶部。

**Architecture:** 新增 `src/matteflow/bootstrap.py` 作为启动检查聚合入口，复用 `discover_media_tools()`、`ModelChecker.collect_model_facts()` 和 `diagnostics.py`，统一返回 `DiagnosticReport`。`scripts/start_gui.ps1` 负责触发检查与打印摘要，`scripts/web_gui.py` 负责消费和展示启动报告，不在本期引入完整向导或 repair 逻辑。

**Tech Stack:** Python, PowerShell 5, pytest, existing `matteflow.diagnostics`, Gradio Web GUI

---

## File Structure

- Create: `src/matteflow/bootstrap.py`
  - 负责聚合启动检查，返回 `DiagnosticReport`
- Modify: `src/matteflow/diagnostics.py`
  - 补充输出目录不可写映射和启动检查聚合 helper
- Modify: `src/matteflow/utils/model_checker.py`
  - 复用现有事实导出，必要时做稳定接口补充
- Modify: `scripts/web_gui.py`
  - 增加启动状态区域的格式化与展示接线
- Modify: `scripts/start_gui.ps1`
  - 启动前调用 Python 侧检查入口，并打印简洁摘要
- Create: `tests/test_bootstrap.py`
  - 覆盖启动报告聚合行为
- Modify: `tests/test_diagnostics.py`
  - 补充输出目录不可写和聚合逻辑测试
- Modify: `tests/test_web_gui_defaults.py`
  - 覆盖 GUI 顶部启动状态展示

## Task 1: 建立启动检查聚合入口

**Files:**
- Create: `src/matteflow/bootstrap.py`
- Test: `tests/test_bootstrap.py`

- [ ] **Step 1: 写失败测试，锁定启动检查最小返回协议**

```python
from pathlib import Path

from matteflow.diagnostics import DiagnosticCode
from matteflow.bootstrap import collect_startup_report


def test_collect_startup_report_returns_ok_when_dependencies_are_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "matteflow.bootstrap.discover_media_tools",
        lambda: type(
            "Result",
            (),
            {
                "ffmpeg_path": "C:/ffmpeg/bin/ffmpeg.exe",
                "ffprobe_path": "C:/ffmpeg/bin/ffprobe.exe",
                "missing_ffmpeg": False,
                "missing_ffprobe": False,
                "is_complete_toolchain": True,
                "source": "system",
            },
        )(),
    )
    monkeypatch.setattr(
        "matteflow.bootstrap.collect_model_facts",
        lambda checker=None: [],
    )

    report = collect_startup_report(default_output_dir=tmp_path)

    assert report.ok is True
    assert report.blocking_count == 0
    assert report.warning_count == 0
    assert report.items == []


def test_collect_startup_report_includes_output_dir_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "matteflow.bootstrap.discover_media_tools",
        lambda: type(
            "Result",
            (),
            {
                "ffmpeg_path": "C:/ffmpeg/bin/ffmpeg.exe",
                "ffprobe_path": "C:/ffmpeg/bin/ffprobe.exe",
                "missing_ffmpeg": False,
                "missing_ffprobe": False,
                "is_complete_toolchain": True,
                "source": "system",
            },
        )(),
    )
    monkeypatch.setattr(
        "matteflow.bootstrap.collect_model_facts",
        lambda checker=None: [],
    )
    monkeypatch.setattr(
        "matteflow.bootstrap.check_output_dir_writable",
        lambda path: {
            "ok": False,
            "path": str(path),
            "reason": "permission denied",
        },
    )

    report = collect_startup_report(default_output_dir=tmp_path)

    assert report.ok is False
    assert report.blocking_count == 1
    assert report.items[0].code is DiagnosticCode.OUTPUT_DIR_UNWRITABLE
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_bootstrap.py -v
```

Expected:

- `ModuleNotFoundError: No module named 'matteflow.bootstrap'`

- [ ] **Step 3: 写最小实现，新增启动检查聚合入口**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from matteflow.diagnostics import DiagnosticReport, from_media_tools, from_output_dir_status, merge_reports
from matteflow.ffmpeg_env import discover_media_tools
from matteflow.utils.model_checker import ModelChecker, collect_model_facts


def check_output_dir_writable(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".matteflow_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return {"ok": False, "path": str(target), "reason": str(exc)}
    return {"ok": True, "path": str(target)}


def collect_startup_report(
    default_output_dir: str | Path,
    *,
    checker: ModelChecker | None = None,
) -> DiagnosticReport:
    reports = [
        from_media_tools(discover_media_tools()),
        from_output_dir_status(check_output_dir_writable(default_output_dir)),
    ]
    reports.extend(collect_model_facts(checker=checker))
    return merge_reports(*reports)
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_bootstrap.py -v
```

Expected:

- `2 passed`

- [ ] **Step 5: 提交**

```bash
git add src/matteflow/bootstrap.py tests/test_bootstrap.py
git commit -m "feat: add startup report bootstrap"
```

## Task 2: 补齐输出目录不可写 diagnostics 映射

**Files:**
- Modify: `src/matteflow/diagnostics.py`
- Test: `tests/test_diagnostics.py`

- [ ] **Step 1: 写失败测试，锁定输出目录不可写的诊断语义**

```python
from matteflow.diagnostics import DiagnosticCode, DiagnosticSeverity, from_output_dir_status


def test_from_output_dir_status_maps_unwritable_directory():
    report = from_output_dir_status(
        {
            "ok": False,
            "path": "E:/MatteFlow/output",
            "reason": "permission denied",
        }
    )

    assert report.ok is False
    assert report.blocking_count == 1
    item = report.items[0]
    assert item.code is DiagnosticCode.OUTPUT_DIR_UNWRITABLE
    assert item.severity is DiagnosticSeverity.ERROR
    assert item.blocking is True
    assert "E:/MatteFlow/output" in item.summary
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_diagnostics.py::test_from_output_dir_status_maps_unwritable_directory -v
```

Expected:

- `ImportError` 或 `AttributeError`
- 提示 `from_output_dir_status` 尚不存在

- [ ] **Step 3: 写最小实现**

```python
def from_output_dir_status(status: dict[str, object]) -> DiagnosticReport:
    if status.get("ok"):
        return DiagnosticReport.ok_report()

    path = str(status.get("path", ""))
    reason = str(status.get("reason", "unknown error"))
    return DiagnosticReport(
        ok=False,
        items=[
            DiagnosticItem(
                code=DiagnosticCode.OUTPUT_DIR_UNWRITABLE,
                severity=DiagnosticSeverity.ERROR,
                title="输出目录不可写",
                summary=f"默认输出目录不可写：{path}",
                details="MatteFlow 无法在当前默认输出目录创建或写入结果文件。",
                actions=[
                    "检查目录权限",
                    "修改输出目录",
                    "确认磁盘和父目录可写",
                ],
                evidence={"path": path, "reason": reason},
                blocking=True,
            )
        ],
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_diagnostics.py::test_from_output_dir_status_maps_unwritable_directory -v
```

Expected:

- `1 passed`

- [ ] **Step 5: 提交**

```bash
git add src/matteflow/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: add output directory startup diagnostics"
```

## Task 3: 统一模型事实到启动报告的适配方式

**Files:**
- Modify: `src/matteflow/bootstrap.py`
- Modify: `src/matteflow/utils/model_checker.py`
- Test: `tests/test_bootstrap.py`

- [ ] **Step 1: 写失败测试，锁定模型事实会被纳入启动报告**

```python
from matteflow.diagnostics import DiagnosticCode, DiagnosticItem, DiagnosticReport, DiagnosticSeverity
from matteflow.bootstrap import collect_startup_report


def test_collect_startup_report_merges_model_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "matteflow.bootstrap.discover_media_tools",
        lambda: type(
            "Result",
            (),
            {
                "ffmpeg_path": "C:/ffmpeg/bin/ffmpeg.exe",
                "ffprobe_path": "C:/ffmpeg/bin/ffprobe.exe",
                "missing_ffmpeg": False,
                "missing_ffprobe": False,
                "is_complete_toolchain": True,
                "source": "system",
            },
        )(),
    )
    monkeypatch.setattr(
        "matteflow.bootstrap.check_output_dir_writable",
        lambda path: {"ok": True, "path": str(path)},
    )
    monkeypatch.setattr(
        "matteflow.bootstrap.collect_model_report",
        lambda checker=None: DiagnosticReport(
            ok=False,
            items=[
                DiagnosticItem(
                    code=DiagnosticCode.MODEL_MISSING,
                    severity=DiagnosticSeverity.ERROR,
                    title="模型缺失",
                    summary="缺少 GVM 模型文件",
                    details="startup check",
                    actions=["下载模型"],
                    evidence={"model": "gvm"},
                    blocking=True,
                )
            ],
        ),
    )

    report = collect_startup_report(default_output_dir=tmp_path)

    assert report.ok is False
    assert any(item.code is DiagnosticCode.MODEL_MISSING for item in report.items)
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_bootstrap.py::test_collect_startup_report_merges_model_diagnostics -v
```

Expected:

- 失败提示 `collect_model_report` 尚不存在

- [ ] **Step 3: 写最小实现**

```python
from matteflow.diagnostics import from_model_status, merge_reports


def collect_model_report(*, checker: ModelChecker | None = None) -> DiagnosticReport:
    reports: list[DiagnosticReport] = []
    for fact in collect_model_facts(checker=checker):
        reports.append(from_model_status(fact))
    return merge_reports(*reports)


def collect_startup_report(
    default_output_dir: str | Path,
    *,
    checker: ModelChecker | None = None,
) -> DiagnosticReport:
    reports = [
        from_media_tools(discover_media_tools()),
        from_output_dir_status(check_output_dir_writable(default_output_dir)),
        collect_model_report(checker=checker),
    ]
    return merge_reports(*reports)
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_bootstrap.py::test_collect_startup_report_merges_model_diagnostics -v
```

Expected:

- `1 passed`

- [ ] **Step 5: 提交**

```bash
git add src/matteflow/bootstrap.py src/matteflow/utils/model_checker.py tests/test_bootstrap.py
git commit -m "feat: merge model diagnostics into startup report"
```

## Task 4: 在 Web GUI 顶部展示启动状态

**Files:**
- Modify: `scripts/web_gui.py`
- Test: `tests/test_web_gui_defaults.py`

- [ ] **Step 1: 写失败测试，锁定启动报告格式化与展示规则**

```python
from matteflow.diagnostics import DiagnosticCode, DiagnosticItem, DiagnosticReport, DiagnosticSeverity
from scripts import web_gui


def test_format_startup_report_returns_ready_message():
    report = DiagnosticReport(ok=True, items=[])
    rendered = web_gui._format_startup_report(report)
    assert "环境已就绪" in rendered


def test_format_startup_report_prioritizes_blocking_errors():
    report = DiagnosticReport(
        ok=False,
        items=[
            DiagnosticItem(
                code=DiagnosticCode.FFPROBE_NOT_FOUND,
                severity=DiagnosticSeverity.ERROR,
                title="FFprobe 缺失",
                summary="当前环境缺少 FFprobe。",
                details="startup",
                actions=["重新配置 FFmpeg 工具链"],
                evidence={"tool": "ffprobe"},
                blocking=True,
            )
        ],
    )
    rendered = web_gui._format_startup_report(report)
    assert "**ERROR** FFprobe 缺失" in rendered
    assert "- 重新配置 FFmpeg 工具链" in rendered
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_web_gui_defaults.py::test_format_startup_report_returns_ready_message tests/test_web_gui_defaults.py::test_format_startup_report_prioritizes_blocking_errors -v
```

Expected:

- 提示 `_format_startup_report` 尚不存在

- [ ] **Step 3: 写最小实现**

```python
from matteflow.bootstrap import collect_startup_report


def _format_startup_report(report):
    if report.ok and not report.items:
        return "**READY** 环境已就绪"
    return report_to_user_text(report)


def _collect_startup_report(output_dir):
    return collect_startup_report(default_output_dir=output_dir)
```

并在页面顶部增加一个只读 `Markdown` 状态区，初始内容来自 `_collect_startup_report(...)`。

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_web_gui_defaults.py::test_format_startup_report_returns_ready_message tests/test_web_gui_defaults.py::test_format_startup_report_prioritizes_blocking_errors -v
```

Expected:

- `2 passed`

- [ ] **Step 5: 提交**

```bash
git add scripts/web_gui.py tests/test_web_gui_defaults.py
git commit -m "feat: show startup diagnostics in web gui"
```

## Task 5: 在启动脚本中接入检查并打印摘要

**Files:**
- Modify: `scripts/start_gui.ps1`
- Modify: `src/matteflow/bootstrap.py`
- Test: `tests/test_bootstrap.py`

- [ ] **Step 1: 写失败测试，锁定启动摘要格式**

```python
from matteflow.bootstrap import format_startup_summary
from matteflow.diagnostics import DiagnosticCode, DiagnosticItem, DiagnosticReport, DiagnosticSeverity


def test_format_startup_summary_reports_blocking_count():
    report = DiagnosticReport(
        ok=False,
        items=[
            DiagnosticItem(
                code=DiagnosticCode.FFMPEG_NOT_FOUND,
                severity=DiagnosticSeverity.ERROR,
                title="FFmpeg 缺失",
                summary="当前环境缺少 FFmpeg。",
                details="startup",
                actions=["运行 configure_ffmpeg.ps1"],
                evidence={"tool": "ffmpeg"},
                blocking=True,
            )
        ],
    )

    text = format_startup_summary(report)

    assert "startup diagnostics" in text.lower()
    assert "1 blocking" in text.lower()
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
pytest tests/test_bootstrap.py::test_format_startup_summary_reports_blocking_count -v
```

Expected:

- 提示 `format_startup_summary` 尚不存在

- [ ] **Step 3: 写最小实现**

```python
def format_startup_summary(report: DiagnosticReport) -> str:
    return (
        "Startup diagnostics: "
        f"{report.blocking_count} blocking, "
        f"{report.warning_count} warning, "
        f"{report.info_count} info"
    )
```

并在 `scripts/start_gui.ps1` 中加入一段调用 Python 模块的逻辑，打印摘要但不阻断 GUI 启动：

```powershell
Write-Host "Running startup diagnostics..."
$startupSummary = & $PythonExe -c "from matteflow.bootstrap import collect_startup_report, format_startup_summary; import os; report = collect_startup_report(default_output_dir=os.path.join(os.getcwd(), 'outputs')); print(format_startup_summary(report))"
Write-Host $startupSummary
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
pytest tests/test_bootstrap.py::test_format_startup_summary_reports_blocking_count -v
```

Expected:

- `1 passed`

- [ ] **Step 5: 提交**

```bash
git add src/matteflow/bootstrap.py scripts/start_gui.ps1 tests/test_bootstrap.py
git commit -m "feat: print startup diagnostics summary"
```

## Task 6: 回归、手动验收与收口

**Files:**
- Verify: `src/matteflow/bootstrap.py`
- Verify: `src/matteflow/diagnostics.py`
- Verify: `scripts/web_gui.py`
- Verify: `scripts/start_gui.ps1`
- Verify: `tests/test_bootstrap.py`
- Verify: `tests/test_diagnostics.py`
- Verify: `tests/test_web_gui_defaults.py`

- [ ] **Step 1: 跑聚焦回归**

Run:

```bash
pytest tests/test_bootstrap.py tests/test_diagnostics.py tests/test_ffmpeg_env.py tests/test_model_checker_runtime.py tests/test_web_gui_defaults.py -q
```

Expected:

- 全绿，或仅保留已知且与本任务无关的既有基线失败

- [ ] **Step 2: 获取编辑器诊断**

Check:

```text
src/matteflow/bootstrap.py
src/matteflow/diagnostics.py
scripts/web_gui.py
scripts/start_gui.ps1
tests/test_bootstrap.py
tests/test_diagnostics.py
tests/test_web_gui_defaults.py
```

Expected:

- 无新增 diagnostics

- [ ] **Step 3: 手动验收**

Run:

```bash
powershell -ExecutionPolicy Bypass -File scripts/start_gui.ps1
```

Check:

- 正常环境下终端打印 startup summary
- GUI 页面顶部显示环境状态区
- 缺模型或工具链异常时，顶部状态区能显示对应说明

- [ ] **Step 4: 最终提交**

```bash
git add src/matteflow/bootstrap.py src/matteflow/diagnostics.py src/matteflow/utils/model_checker.py scripts/web_gui.py scripts/start_gui.ps1 tests/test_bootstrap.py tests/test_diagnostics.py tests/test_web_gui_defaults.py
git commit -m "feat: add minimal startup diagnostics flow"
```

## Spec Coverage Check

- 已覆盖媒体工具完整性检查：`Task 1`、`Task 5`
- 已覆盖模型可用性检查：`Task 3`
- 已覆盖默认输出目录可写性检查：`Task 1`、`Task 2`
- 已覆盖统一 `DiagnosticReport` 聚合：`Task 1`、`Task 2`、`Task 3`
- 已覆盖 GUI 顶部展示：`Task 4`
- 已覆盖脚本层启动摘要：`Task 5`
- 已覆盖聚焦回归与手动验收：`Task 6`
- 未越界到完整首启向导、repair、模型下载或更新检查
