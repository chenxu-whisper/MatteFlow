"""GUI-friendly views for structured processing reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

UNAVAILABLE_REPORT_TEXT = "处理诊断报告暂不可用。"


@dataclass(frozen=True)
class ProcessingReportView:
    """Text summaries ready for GUI rendering."""

    report_path: Path | None
    title: str
    quality_summary: str
    model_summary: str
    region_summary: str
    recovery_summary: str
    fusion_summary: str = ""
    quality_selection_summary: str = ""
    warnings: tuple[str, ...] = ()

    def to_markdown(self) -> str:
        """Render the report view as compact Markdown."""
        if self.quality_summary == UNAVAILABLE_REPORT_TEXT:
            return UNAVAILABLE_REPORT_TEXT

        sections = [
            f"### {self.title}",
            self.quality_summary,
            self.model_summary,
            self.region_summary,
            self.quality_selection_summary,
            self.recovery_summary,
            self.fusion_summary,
            self._warnings_markdown(),
        ]
        return "\n\n".join(section for section in sections if section)

    def _warnings_markdown(self) -> str:
        if not self.warnings:
            return ""
        lines = ["**警告**"]
        lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


class ProcessingReportViewBuilder:
    """Build GUI report summaries from processing report JSON."""

    def from_path(self, report_path: Path | str | None) -> ProcessingReportView:
        """Load a processing report JSON file and build a display view."""
        if report_path is None:
            return self.unavailable(None)

        path = Path(report_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return self.unavailable(path)
        if not isinstance(payload, Mapping):
            return self.unavailable(path)
        return self.from_payload(payload, report_path=path)

    def from_payload(
        self,
        payload: Mapping[str, Any],
        report_path: Path | str | None = None,
    ) -> ProcessingReportView:
        """Build a display view from an in-memory report payload."""
        return ProcessingReportView(
            report_path=Path(report_path) if report_path is not None else None,
            title="处理诊断报告",
            quality_summary=self._quality_summary(payload),
            model_summary=self._model_summary(payload),
            region_summary=self._region_summary(payload),
            recovery_summary=self._recovery_summary(payload),
            fusion_summary=self._fusion_summary(payload),
            quality_selection_summary=format_quality_selection_summary(payload),
            warnings=tuple(str(item) for item in payload.get("warnings", ()) or ()),
        )

    @staticmethod
    def unavailable(report_path: Path | str | None = None) -> ProcessingReportView:
        """Return a stable placeholder view when report data cannot be loaded."""
        return ProcessingReportView(
            report_path=Path(report_path) if report_path is not None else None,
            title="处理诊断报告",
            quality_summary=UNAVAILABLE_REPORT_TEXT,
            model_summary="",
            region_summary="",
            recovery_summary="",
            fusion_summary="",
            quality_selection_summary="",
            warnings=(),
        )

    @staticmethod
    def _quality_summary(payload: Mapping[str, Any]) -> str:
        quality = _mapping(payload.get("quality"))
        job = _mapping(payload.get("job"))
        timings = _mapping(payload.get("timings"))
        return "\n".join(
            [
                "**质量摘要**",
                f"质量评分：{_format_float(quality.get('overall_score'))}",
                f"边缘不确定性：{_format_float(quality.get('mean_edge_uncertainty'))}",
                f"孔洞像素：{_format_int(quality.get('hole_pixels'))}",
                f"噪点像素：{_format_int(quality.get('speckle_pixels'))}",
                f"背景残留：{_format_float(quality.get('background_residue'))}",
                f"时序闪烁：{_format_float(quality.get('temporal_flicker'))}",
                f"帧数：{_format_int(job.get('frame_count'))}",
                f"总耗时：{_format_seconds(timings.get('total'))}",
            ]
        )

    @staticmethod
    def _model_summary(payload: Mapping[str, Any]) -> str:
        job = _mapping(payload.get("job"))
        decisions = _mapping(payload.get("model_decisions"))
        active_model = decisions.get("active_ai_model") or job.get("ai_model_active")
        return "\n".join(
            [
                "**模型决策**",
                f"请求模型：{_format_value(job.get('ai_model_requested'))}",
                f"实际模型：{_format_value(active_model)}",
                f"背景模式：{_format_value(job.get('background_mode_effective'))}",
                f"质量模式：{_format_value(job.get('quality_mode'))}",
            ]
        )

    @staticmethod
    def _region_summary(payload: Mapping[str, Any]) -> str:
        regions = _mapping(payload.get("regions"))
        labels = [
            ("subject_pixels", "主体"),
            ("hair_edge_pixels", "发丝边缘"),
            ("luminous_prop_pixels", "发光道具"),
            ("transparent_effect_pixels", "透明光效"),
            ("background_residue_pixels", "背景残留"),
            ("uncertain_edge_pixels", "不确定边缘"),
        ]
        lines = ["**区域统计**"]
        lines.extend(f"{label}：{_format_int(regions.get(key))} px" for key, label in labels)
        return "\n".join(lines)

    @staticmethod
    def _recovery_summary(payload: Mapping[str, Any]) -> str:
        recovery = _mapping(payload.get("foreground_recovery"))
        if not recovery:
            return "**前景恢复**\n无前景恢复诊断。"
        attempted = _format_int(recovery.get("attempted_pixels"))
        accepted = _format_int(recovery.get("accepted_pixels"))
        return "\n".join(
            [
                "**前景恢复**",
                f"处理帧数：{_format_int(recovery.get('frames'))}",
                f"接受像素：{accepted} / {attempted}",
                f"估计幕布色：{_format_rgb(recovery.get('screen_rgb'))}",
            ]
        )

    @staticmethod
    def _fusion_summary(payload: Mapping[str, Any]) -> str:
        fusion = _mapping(payload.get("fusion"))
        if not fusion or not fusion.get("available"):
            return "**融合诊断**\n无融合诊断。"

        selected = _mapping(fusion.get("selected_by_region"))
        rejected = _mapping(fusion.get("rejected_takeovers"))
        lines = ["**融合诊断**"]
        if selected:
            lines.append("区域选择：")
            lines.extend(f"- {key}: {value}" for key, value in selected.items())
        if rejected:
            lines.append("拒绝抢占：")
            lines.extend(f"- {key}: {value}" for key, value in rejected.items())
        return "\n".join(lines)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def format_quality_selection_summary(payload: Mapping[str, Any]) -> str:
    quality_selection = _mapping(payload.get("quality_selection"))
    if not quality_selection.get("available"):
        return "**质量选择**\n质量选择: 未启用"

    lines = [
        "**质量选择**",
        "质量选择: 已启用",
        f"候选数量: {_format_int(quality_selection.get('candidate_count'))}",
    ]
    selected_counts = _mapping(quality_selection.get("selected_model_counts"))
    if selected_counts:
        lines.append("选中模型统计:")
        lines.extend(f"- {model_name}: {_format_int(count)}" for model_name, count in selected_counts.items())

    skipped = quality_selection.get("skipped_candidates") or []
    if skipped:
        lines.append("跳过候选:")
        for item in skipped:
            if not isinstance(item, Mapping):
                continue
            lines.append(
                f"- {_format_value(item.get('name'))}: {_format_value(item.get('reason'))}"
            )
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


def _format_float(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def _format_int(value: Any) -> str:
    if value is None:
        return "0"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "0"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "N/A"


def _format_rgb(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return "N/A"
    try:
        channels = [int(round(float(channel))) for channel in value[:3]]
    except (TypeError, ValueError):
        return "N/A"
    return f"rgb({channels[0]}, {channels[1]}, {channels[2]})"
