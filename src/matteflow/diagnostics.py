from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
                    evidence={
                        "ffmpeg_path": discovery_result.ffmpeg_path,
                        "source": discovery_result.source,
                    },
                    blocking=True,
                ),
            )
        )

    return DiagnosticReport()


def from_model_status(model_status_dict: dict[str, dict[str, Any]]) -> DiagnosticReport:
    items: list[DiagnosticItem] = []
    for model_key, info in model_status_dict.items():
        if info.get("available"):
            continue

        reason = info.get("reason") or ""
        display_name = info.get("display_name", model_key)

        if "不可导入" in reason:
            code = DiagnosticCode.MODEL_RUNTIME_IMPORT_FAILED
            title = f"{display_name} runtime 不可用"
            summary = f"{display_name} 已安装，但运行时依赖无法导入。"
        elif "仅支持 CUDA" in reason:
            code = DiagnosticCode.MODEL_GPU_REQUIRED
            title = f"{display_name} 需要 CUDA"
            summary = f"{display_name} 仅支持 CUDA GPU。"
        else:
            code = DiagnosticCode.MODEL_MISSING
            title = f"{display_name} 不可用"
            summary = f"{display_name} 当前不可用，通常是模型文件缺失。"

        auto_download = bool(info.get("auto_download"))
        items.append(
            DiagnosticItem(
                code=code,
                severity=DiagnosticSeverity.WARNING if auto_download else DiagnosticSeverity.ERROR,
                title=title,
                summary=summary,
                actions=("检查模型目录", "确认依赖是否已安装"),
                evidence={
                    "model_key": model_key,
                    "path": info.get("path"),
                    "reason": reason,
                },
                blocking=not auto_download,
            )
        )

    return DiagnosticReport(items=tuple(items))


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
