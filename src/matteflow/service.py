"""Service-layer entry points for MatteFlow processing jobs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

from .config import BackgroundMode, MattingConfig, QualityMode
from .diagnostics import from_exception
from .errors import MatteFlowError, ProcessingError

ProgressCallback = Callable[[int, int, str], None]
PipelineFactory = Callable[[MattingConfig], Any]


@dataclass(frozen=True)
class ProcessJobParams:
    """Immutable snapshot of one submitted processing job."""

    input_path: str | Path
    output_dir: str | Path
    background_mode: BackgroundMode = BackgroundMode.AUTO
    quality_mode: QualityMode = QualityMode.STANDARD
    use_ai: bool = True
    ai_model: str = "auto"
    config_overrides: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(
            self,
            "background_mode",
            self._coerce_background_mode(self.background_mode),
        )
        object.__setattr__(
            self,
            "quality_mode",
            self._coerce_quality_mode(self.quality_mode),
        )
        frozen_overrides = MappingProxyType(copy.deepcopy(dict(self.config_overrides)))
        object.__setattr__(self, "config_overrides", frozen_overrides)

    @staticmethod
    def _coerce_background_mode(value: BackgroundMode | str) -> BackgroundMode:
        if isinstance(value, BackgroundMode):
            return value
        return BackgroundMode(value)

    @staticmethod
    def _coerce_quality_mode(value: QualityMode | str) -> QualityMode:
        if isinstance(value, QualityMode):
            return value
        return QualityMode(value)


@dataclass(frozen=True)
class ProcessOutputConfig:
    """Snapshot of output toggles for future queue/manifest integration."""

    output_format: str = "png"
    output_fg: bool = False
    output_matte: bool = True
    output_comp: bool = False
    output_processed: bool = True
    exr_compression: str = "dwab"


@dataclass(frozen=True)
class ProcessResult:
    """Structured result returned by the service layer."""

    success: bool
    input_path: Path
    output_dir: Path
    background_mode: str
    frame_count: int = 0
    processing_time: float = 0.0
    timings: Mapping[str, float] = field(default_factory=dict)
    processing_report_path: Optional[Path] = None
    error_message: Optional[str] = None


class MatteFlowService:
    """Stable API between UI/CLI callers and the matting pipeline."""

    def __init__(self, pipeline_factory: Optional[PipelineFactory] = None) -> None:
        self._pipeline_factory = pipeline_factory or self._default_pipeline_factory

    def process(
        self,
        params: ProcessJobParams,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> ProcessResult:
        """Run a processing job from an immutable parameter snapshot."""
        try:
            config = self._build_config(params)
            pipeline = self._pipeline_factory(config)
            process_kwargs: dict[str, Any] = {"progress_callback": progress_callback}
            if cancel_check is not None:
                process_kwargs["cancel_check"] = cancel_check
            raw_result = pipeline.process(params.input_path, params.output_dir, **process_kwargs)
        except MatteFlowError:
            raise
        except Exception as exc:
            raise ProcessingError(self._format_processing_error(exc)) from exc

        return self._to_process_result(params, raw_result)

    @staticmethod
    def _default_pipeline_factory(config: MattingConfig) -> Any:
        from .pipeline import MattingPipeline

        return MattingPipeline(config)

    @staticmethod
    def _build_config(params: ProcessJobParams) -> MattingConfig:
        config = MattingConfig()
        config.background_mode = params.background_mode
        config.quality_mode = params.quality_mode
        config.use_ai = params.use_ai
        config.ai_model = params.ai_model

        for name, value in params.config_overrides.items():
            if not hasattr(config, name):
                raise ProcessingError(f"Unknown MatteFlow config option: {name}")
            setattr(config, name, copy.deepcopy(value))
        return config

    @staticmethod
    def _to_process_result(params: ProcessJobParams, raw_result: Mapping[str, Any]) -> ProcessResult:
        timings = raw_result.get("timings") or {}
        frame_count = raw_result.get("frame_count")
        if frame_count is None:
            frame_count = raw_result.get("frames_processed", 0)

        processing_time = raw_result.get("processing_time")
        if processing_time is None:
            processing_time = raw_result.get("elapsed_time", 0.0)
        processing_report_path = raw_result.get("processing_report_path")

        return ProcessResult(
            success=True,
            input_path=params.input_path,
            output_dir=params.output_dir,
            background_mode=str(raw_result.get("background_mode", params.background_mode.value)),
            frame_count=int(frame_count),
            processing_time=float(processing_time),
            timings=MappingProxyType(copy.deepcopy(dict(timings))),
            processing_report_path=Path(processing_report_path) if processing_report_path else None,
        )

    @staticmethod
    def _format_processing_error(exc: Exception) -> str:
        report = from_exception(exc, context={"stage": "process"})
        first_item = report.items[0]
        return f"{first_item.title}: {first_item.summary} Original error: {exc}"
