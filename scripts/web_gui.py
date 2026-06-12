"""
用法:
    python scripts/web_gui.py
    python scripts/web_gui.py --port 7860
    python scripts/web_gui.py --port 7862 --debug
    python scripts/web_gui.py --share
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
import inspect
import logging
import shutil
import tempfile
from threading import RLock
import time
import zipfile
from uuid import uuid4
import cv2
import numpy as np
import gradio as gr
from PIL import Image

from matteflow import MattingPipeline, MattingConfig, QualityMode, BackgroundMode
from matteflow.auto_params import apply_suggestion, suggest_input_params
from matteflow.diagnostics import (
    DiagnosticReport,
    from_exception,
    from_media_tools,
    from_model_status,
    merge_reports,
)
from matteflow.ffmpeg_env import discover_media_tools
from matteflow.input import ImageDecoder, InputKind, SequenceDecoder, detect_input_kind
from matteflow.input.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from matteflow.job_queue import GPUJob, GPUJobQueue, JobStatus, JobType
from matteflow.job_worker import JobWorker
from matteflow.service import MatteFlowService, ProcessJobParams
from matteflow.utils.cv_compat import video_writer_fourcc
from matteflow.utils.model_checker import ModelChecker
from matteflow.utils.output_paths import (
    resolve_project_output_dir,
    sanitize_output_name,
)

logger = logging.getLogger(__name__)
GRADIO_PREVIEW_MAX_AGE_SECONDS = 3600
SESSION_REGISTRY_MAX_AGE_SECONDS = 3600
SUPPORTED_UPLOAD_EXTENSIONS = sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
PREVIEW_OUTPUT_EXTENSIONS = IMAGE_EXTENSIONS | {".tga"}
GUI_INPUT_PATH_NOT_ALLOWED_MESSAGE = "请通过上传组件选择素材，禁止直接引用服务器本地路径"
GUI_INPUT_PATH_MISSING_MESSAGE = "输入素材不存在"
PREVIEW_FAST_FRAME_LIMIT = 60
GUI_DEFAULTS = {
    "mode": "green",
    "quality": "standard",
    "preferred_ai": "gvm",
    "pure_color": True,
    "use_filter": False,
    "green_similarity": 0.4,
    "green_despill": 0.7,
    "green_hair": 0.8,
    "white_protect_brightness": 180,
    "white_protect_saturation": 25,
    "edge_despill_factor": 1.2,
    "screen_color": "auto",
    "key_strength": 1.0,
    "clip_black": 0.0,
    "clip_white": 1.0,
    "shrink_grow": 0,
    "edge_blur": 0,
    "despill_enable": True,
    "despill_strength": 0.7,
    "despill_color": "green",
    "despeckle_enable": True,
    "despeckle_radius": 2,
    "despeckle_threshold": 0.0,
    "transparency_preserve": 0.7,
    "gvm_max_internal_size": 768,
    "auto_optimize": False,
    "generate_zip": False,
    "output_fg": False,
    "output_matte": True,
    "output_comp": False,
    "output_processed": True,
}

GUI_PRIMARY_CONTROL_KEYS = [
    "use_ai",
    "quality",
    "key_strength",
    "transparency_preserve",
    "green_despill",
    "edge_despill_factor",
    "shrink_grow",
    "edge_blur",
    "gvm_max_internal_size",
]

GUI_FIXED_PARAMETER_DEFAULTS = {
    "pure_color": GUI_DEFAULTS["pure_color"],
    "use_filter": GUI_DEFAULTS["use_filter"],
    "edge_softness": 0.0,
    "temporal_strength": 0.5,
    "color_space": "sRGB",
    "despill_enable": GUI_DEFAULTS["despill_enable"],
    "despill_color": GUI_DEFAULTS["despill_color"],
}

RECOMMENDED_PRESET_OUTPUT_KEYS = [
    "mode",
    "quality",
    "use_ai",
    "pure_color_mode",
    "use_guided_filter",
    "green_similarity",
    "green_despill",
    "green_hair",
    "white_protect_thresh",
    "white_protect_sat",
    "edge_despill_factor",
    "screen_color",
    "key_strength",
    "clip_black",
    "clip_white",
    "shrink_grow",
    "edge_blur",
    "despill_enable",
    "despill_strength",
    "despill_color",
    "despeckle_enable",
    "despeckle_radius",
    "despeckle_threshold",
    "transparency_preserve",
    "gvm_max_internal_size",
    "auto_optimize",
    "generate_zip",
    "output_fg",
    "output_matte",
    "output_comp",
    "output_processed",
]

# 全局状态
_output_dir = None
_current_preview_index = 0


class _SessionQueueRegistry:
    def __init__(self) -> None:
        self._queues: dict[str, GPUJobQueue] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = RLock()

    def _prune_stale(self, now: float) -> list[str]:
        stale_before = now - SESSION_REGISTRY_MAX_AGE_SECONDS
        pruned_session_ids: list[str] = []
        for session_id, last_seen in list(self._last_seen.items()):
            if last_seen > stale_before:
                continue
            queue = self._queues.get(session_id)
            if queue is None:
                self._last_seen.pop(session_id, None)
                continue
            if queue.running_job is not None or queue.queued_snapshot:
                continue
            self._queues.pop(session_id, None)
            self._last_seen.pop(session_id, None)
            pruned_session_ids.append(session_id)
        return pruned_session_ids

    def get_queue(self, session_id: str) -> GPUJobQueue:
        with self._lock:
            now = time.time()
            pruned_session_ids = self._prune_stale(now)
            queue = self._queues.get(session_id)
            if queue is None:
                queue = GPUJobQueue()
                self._queues[session_id] = queue
            self._last_seen[session_id] = now
        _cleanup_expired_session_preview_artifacts(pruned_session_ids)
        return queue


_session_queue_registry = _SessionQueueRegistry()


class _SessionUploadRegistry:
    def __init__(self) -> None:
        self._paths: dict[str, set[Path]] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = RLock()

    def _prune_stale(self, now: float) -> list[str]:
        stale_before = now - SESSION_REGISTRY_MAX_AGE_SECONDS
        pruned_session_ids: list[str] = []
        for session_id, last_seen in list(self._last_seen.items()):
            if last_seen > stale_before:
                continue
            self._paths.pop(session_id, None)
            self._last_seen.pop(session_id, None)
            pruned_session_ids.append(session_id)
        return pruned_session_ids

    def register_path(self, session_id: str, path: Path) -> None:
        resolved_path = path.resolve(strict=False)
        with self._lock:
            now = time.time()
            pruned_session_ids = self._prune_stale(now)
            self._paths.setdefault(session_id, set()).add(resolved_path)
            self._last_seen[session_id] = now
        _cleanup_expired_session_preview_artifacts(pruned_session_ids)

    def is_registered(self, session_id: str, path: Path) -> bool:
        resolved_path = path.resolve(strict=False)
        with self._lock:
            now = time.time()
            pruned_session_ids = self._prune_stale(now)
            is_allowed = resolved_path in self._paths.get(session_id, set())
            if is_allowed:
                self._last_seen[session_id] = now
        _cleanup_expired_session_preview_artifacts(pruned_session_ids)
        return is_allowed


class _SessionPreviewArtifactRegistry:
    def __init__(self) -> None:
        self._paths: dict[str, set[Path]] = {}
        self._lock = RLock()

    def register_path(self, session_id: str | None, path: Path) -> None:
        if session_id is None:
            return
        resolved_path = path.resolve(strict=False)
        with self._lock:
            self._paths.setdefault(session_id, set()).add(resolved_path)

    def prune_session(self, session_id: str) -> None:
        with self._lock:
            paths = list(self._paths.pop(session_id, set()))

        for path in paths:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except FileNotFoundError:
                continue
            except Exception:
                logger.warning("Failed to remove stale session preview artifact: %s", path, exc_info=True)


_session_upload_registry = _SessionUploadRegistry()
_session_preview_artifact_registry = _SessionPreviewArtifactRegistry()


def _cleanup_expired_session_preview_artifacts(session_ids: list[str]) -> None:
    for session_id in session_ids:
        _session_preview_artifact_registry.prune_session(session_id)

# 检查可用模型
_model_checker = ModelChecker()
_available_models = _model_checker.get_available_models()
_ui_choices = _model_checker.get_ui_choices()


def _configure_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _sanitize_output_name(name: str) -> str:
    return sanitize_output_name(name)


def _resolve_gui_output_dir(
    video_path,
    output_root: Path | None = None,
    job_token: str | None = None,
) -> Path:
    return resolve_project_output_dir(
        Path(video_path),
        project_root=project_root,
        output_root=output_root,
        job_token=job_token,
    )


def _resolve_request_output_dir(video_path, request_token: str) -> Path:
    resolver = _resolve_gui_output_dir
    try:
        signature = inspect.signature(resolver)
    except (TypeError, ValueError):
        return resolver(video_path, job_token=request_token)

    if "job_token" in signature.parameters:
        return resolver(video_path, job_token=request_token)
    return resolver(video_path)


def _gradio_preview_root() -> Path:
    root = Path(tempfile.gettempdir()) / "gradio" / "matteflow_previews"
    root.mkdir(parents=True, exist_ok=True)
    _prune_stale_gradio_preview_artifacts(root)
    return root


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_gui_input_path(file_path, session_id: str | None = None, require_registered: bool = False) -> Path:
    path = Path(file_path)
    resolved_path = path.resolve(strict=False)
    allowed_roots = (
        Path(tempfile.gettempdir()).resolve(),
        _gradio_preview_root().resolve(),
    )
    if not any(_is_within_root(resolved_path, root) for root in allowed_roots):
        raise ValueError(GUI_INPUT_PATH_NOT_ALLOWED_MESSAGE)
    if require_registered and session_id is not None:
        if not _session_upload_registry.is_registered(session_id, resolved_path):
            raise ValueError(GUI_INPUT_PATH_NOT_ALLOWED_MESSAGE)
    return path


def _register_gui_upload_path(file_path, session_id: str | None = None) -> None:
    if not file_path or session_id is None:
        return
    _session_upload_registry.register_path(session_id, Path(file_path))


def _is_stale_preview_artifact(path: Path, now: float) -> bool:
    try:
        age_seconds = now - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age_seconds > GRADIO_PREVIEW_MAX_AGE_SECONDS


def _prune_stale_preview_cache_dirs(cache_root: Path, active_job_id: str | None = None) -> None:
    if not cache_root.exists():
        return
    now = time.time()
    for candidate in cache_root.iterdir():
        if candidate.name == "downloads" or not candidate.is_dir():
            continue
        if active_job_id is not None and candidate.name == active_job_id:
            continue
        if not _is_stale_preview_artifact(candidate, now):
            continue
        try:
            shutil.rmtree(candidate)
        except FileNotFoundError:
            continue
        except Exception:
            logger.warning("Failed to remove stale Gradio preview cache: %s", candidate, exc_info=True)


def _prune_stale_preview_downloads(cache_root: Path) -> None:
    downloads_dir = cache_root / "downloads"
    if not downloads_dir.exists():
        return
    now = time.time()
    for candidate in downloads_dir.iterdir():
        if candidate.is_dir():
            continue
        if not _is_stale_preview_artifact(candidate, now):
            continue
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        except Exception:
            logger.warning("Failed to remove stale preview download: %s", candidate, exc_info=True)


def _prune_stale_gradio_preview_artifacts(cache_root: Path, active_job_id: str | None = None) -> None:
    _prune_stale_preview_cache_dirs(cache_root, active_job_id=active_job_id)
    _prune_stale_preview_downloads(cache_root)


def _copy_preview_into_gradio_cache(
    job: GPUJob,
    preview_path: Path,
    job_output_dir: Path,
    session_id: str | None = None,
) -> str:
    cache_root = _gradio_preview_root()
    _prune_stale_gradio_preview_artifacts(cache_root, active_job_id=job.id)
    job_cache_dir = cache_root / job.id
    job_cache_dir.mkdir(parents=True, exist_ok=True)
    _session_preview_artifact_registry.register_path(session_id, job_cache_dir)
    preview_name = f"preview_{job_output_dir.name}.mp4"
    gradio_preview_path = job_cache_dir / preview_name
    shutil.copy2(str(preview_path), str(gradio_preview_path))
    return str(gradio_preview_path.resolve())


def _create_upload_preview(file_path, session_id=None):
    hide_image = gr.update(value=None, visible=False)
    hide_video = gr.update(value=None, visible=False)
    if not file_path:
        return hide_image, hide_video, "未选择素材"

    try:
        path = _validate_gui_input_path(file_path)
    except ValueError as exc:
        return hide_image, hide_video, str(exc)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        if not path.exists():
            return hide_image, hide_video, f"图片文件不存在: {path.name}"
        _register_gui_upload_path(path, session_id=session_id)
        return gr.update(value=str(path), visible=True), hide_video, f"已选择图片: {path.name}"

    if suffix in VIDEO_EXTENSIONS:
        if not path.exists():
            return hide_image, hide_video, f"视频文件不存在: {path.name}"
        _register_gui_upload_path(path, session_id=session_id)
        return hide_image, gr.update(value=str(path), visible=True), f"已选择视频: {path.name}"

    return hide_image, hide_video, f"不支持的素材格式: {path.suffix}"


def _default_ai_choice() -> str:
    preferred = GUI_DEFAULTS["preferred_ai"]
    if any(value == preferred for _, value in _ui_choices):
        return preferred
    return _ui_choices[0][1] if _ui_choices else "traditional"


def _apply_recommended_preset() -> dict:
    return {
        "mode": GUI_DEFAULTS["mode"],
        "quality": GUI_DEFAULTS["quality"],
        "use_ai": _default_ai_choice(),
        "pure_color_mode": GUI_DEFAULTS["pure_color"],
        "use_guided_filter": GUI_DEFAULTS["use_filter"],
        "green_similarity": GUI_DEFAULTS["green_similarity"],
        "green_despill": GUI_DEFAULTS["green_despill"],
        "green_hair": GUI_DEFAULTS["green_hair"],
        "white_protect_thresh": GUI_DEFAULTS["white_protect_brightness"],
        "white_protect_sat": GUI_DEFAULTS["white_protect_saturation"],
        "edge_despill_factor": GUI_DEFAULTS["edge_despill_factor"],
        "screen_color": GUI_DEFAULTS["screen_color"],
        "key_strength": GUI_DEFAULTS["key_strength"],
        "clip_black": GUI_DEFAULTS["clip_black"],
        "clip_white": GUI_DEFAULTS["clip_white"],
        "shrink_grow": GUI_DEFAULTS["shrink_grow"],
        "edge_blur": GUI_DEFAULTS["edge_blur"],
        "despill_enable": GUI_DEFAULTS["despill_enable"],
        "despill_strength": GUI_DEFAULTS["despill_strength"],
        "despill_color": GUI_DEFAULTS["despill_color"],
        "despeckle_enable": GUI_DEFAULTS["despeckle_enable"],
        "despeckle_radius": GUI_DEFAULTS["despeckle_radius"],
        "despeckle_threshold": GUI_DEFAULTS["despeckle_threshold"],
        "transparency_preserve": GUI_DEFAULTS["transparency_preserve"],
        "gvm_max_internal_size": GUI_DEFAULTS["gvm_max_internal_size"],
        "auto_optimize": GUI_DEFAULTS["auto_optimize"],
        "generate_zip": GUI_DEFAULTS["generate_zip"],
        "output_fg": GUI_DEFAULTS["output_fg"],
        "output_matte": GUI_DEFAULTS["output_matte"],
        "output_comp": GUI_DEFAULTS["output_comp"],
        "output_processed": GUI_DEFAULTS["output_processed"],
    }


def _recommended_preset_updates():
    preset = _apply_recommended_preset()
    return tuple(gr.update(value=preset[key]) for key in RECOMMENDED_PRESET_OUTPUT_KEYS)


def _default_service_factory():
    return MatteFlowService(pipeline_factory=MattingPipeline)


def _new_session_id() -> str:
    return uuid4().hex


def _default_queue_factory(session_id: str | None = None):
    return _session_queue_registry.get_queue(session_id or _new_session_id())


def _resolve_queue(queue_factory=None, session_id: str | None = None):
    factory = queue_factory or _default_queue_factory
    if queue_factory is None:
        return factory(session_id)
    try:
        return factory(session_id=session_id)
    except TypeError:
        try:
            return factory(session_id)
        except TypeError:
            return factory()


def _default_worker_factory(queue, service):
    return JobWorker(queue, service)


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


def _build_process_job_params(video_path, output_dir, config):
    config_overrides = {
        "ai_enhance": config.ai_enhance,
        "pure_color_mode": config.pure_color_mode,
        "use_guided_filter": config.use_guided_filter,
        "green_similarity": config.green_similarity,
        "green_despill_strength": config.green_despill_strength,
        "green_hair_detail": config.green_hair_detail,
        "white_protect_brightness": config.white_protect_brightness,
        "white_protect_saturation": config.white_protect_saturation,
        "edge_despill_factor": config.edge_despill_factor,
        "black_threshold": config.black_threshold,
        "black_glow_preserve": config.black_glow_preserve,
        "black_particle_boost": config.black_particle_boost,
        "edge_softness": config.edge_softness,
        "temporal_strength": config.temporal_strength,
        "transparency_preserve": config.transparency_preserve,
        "gvm_max_internal_size": config.gvm_max_internal_size,
        "ai_enhance_gamma": config.ai_enhance_gamma,
        "ai_enhance_threshold": config.ai_enhance_threshold,
        "ai_enhance_gain": config.ai_enhance_gain,
        "ai_enhance_sharpen": config.ai_enhance_sharpen,
        "screen_color": config.screen_color,
        "key_strength": config.key_strength,
        "clip_black": config.clip_black,
        "clip_white": config.clip_white,
        "shrink_grow": config.shrink_grow,
        "edge_blur": config.edge_blur,
        "despill_enable": config.despill_enable,
        "despill_strength": config.despill_strength,
        "despill_color": config.despill_color,
        "despeckle_enable": config.despeckle_enable,
        "despeckle_radius": config.despeckle_radius,
        "despeckle_threshold": config.despeckle_threshold,
        "color_space": config.color_space,
        "output_fg": config.output_fg,
        "output_matte": config.output_matte,
        "output_comp": config.output_comp,
        "output_processed": config.output_processed,
        "generate_zip_by_default": config.generate_zip_by_default,
        "preview_quality_mode": config.preview_quality_mode,
    }
    return ProcessJobParams(
        input_path=video_path,
        output_dir=output_dir,
        background_mode=config.background_mode,
        quality_mode=config.quality_mode,
        use_ai=config.use_ai,
        ai_model=config.ai_model,
        config_overrides=config_overrides,
    )


def _build_process_job(video_path, params, job_id: str | None = None):
    return GPUJob(
        job_type=JobType.PROCESS_MEDIA,
        input_path=video_path,
        params=params,
        id=job_id or uuid4().hex,
    )


def _queued_position(queue, job):
    queued_jobs = list(queue.queued_snapshot)
    try:
        return queued_jobs.index(job) + 1
    except ValueError:
        return max(len(queued_jobs), 1)


def _queued_status(queue, job):
    return f"⏳ 已入队，第 {_queued_position(queue, job)} 位"


QUEUE_TABLE_HEADERS = ["job_id", "type", "status", "progress", "input", "message"]


def _format_job_progress(job):
    if job.total <= 0:
        return "-"
    percentage = int((job.current / job.total) * 100) if job.total else 0
    return f"{job.current}/{job.total} ({percentage}%)"


def _format_queue_rows(queue):
    rows = []
    jobs = []
    if queue.running_job is not None:
        jobs.append(queue.running_job)
    jobs.extend(queue.queued_snapshot)
    jobs.extend(queue.history_snapshot)

    for job in jobs:
        message = job.error_message or (job.stage if job.status == JobStatus.RUNNING else "")
        rows.append(
            [
                job.id[:8],
                job.job_type.value,
                job.status.value,
                _format_job_progress(job),
                Path(job.input_path).name,
                message,
            ]
        )
    return rows


def get_queue_panel_value(session_id=None, queue_factory=None):
    queue = _resolve_queue(queue_factory=queue_factory, session_id=session_id)
    return QUEUE_TABLE_HEADERS, _format_queue_rows(queue)


def _queue_panel_rows_only(session_id=None, queue_factory=None):
    _headers, rows = get_queue_panel_value(session_id=session_id, queue_factory=queue_factory)
    return rows


def _clear_terminal_history(queue):
    return queue.clear_history()


def _summarize_job_result(job):
    result = job.result
    frame_count = job.current or job.total
    processing_time = 0.0

    if result is not None:
        frame_count = result.frame_count or frame_count
        processing_time = result.processing_time

    fps = (frame_count / processing_time) if processing_time > 0 else 0.0
    return frame_count, processing_time, fps


def cancel_current_job(session_id=None, queue_factory=None):
    queue = _resolve_queue(queue_factory=queue_factory, session_id=session_id)
    running_job = queue.running_job
    if running_job is None:
        return "当前没有正在运行的任务"

    queue.cancel_job(running_job.id)
    return "⏹ 已请求取消当前任务"


def cancel_current_job_with_panel(session_id=None, queue_factory=None):
    status = cancel_current_job(session_id=session_id, queue_factory=queue_factory)
    return status, _queue_panel_rows_only(session_id=session_id, queue_factory=queue_factory)


def run_all_jobs(session_id=None, queue_factory=None, worker_factory=None, service_factory=None):
    queue = _resolve_queue(queue_factory=queue_factory, session_id=session_id)

    if queue.running_job is not None:
        headers, rows = get_queue_panel_value(session_id=session_id, queue_factory=lambda: queue)
        return "⏳ 当前已有任务在运行", headers, rows

    if queue.next_job() is None:
        headers, rows = get_queue_panel_value(session_id=session_id, queue_factory=lambda: queue)
        return "当前没有待运行任务", headers, rows

    service = (service_factory or _default_service_factory)()
    worker = (worker_factory or _default_worker_factory)(queue, service)
    while queue.running_job is None and queue.next_job() is not None:
        completed_job = worker.run_next_job()
        if completed_job is None:
            break

    headers, rows = get_queue_panel_value(session_id=session_id, queue_factory=lambda: queue)
    return "▶ 队列已全部运行完成", headers, rows


def run_all_jobs_for_ui(session_id=None, queue_factory=None, worker_factory=None, service_factory=None):
    status, _headers, rows = run_all_jobs(
        session_id=session_id,
        queue_factory=queue_factory,
        worker_factory=worker_factory,
        service_factory=service_factory,
    )
    return status, rows


def clear_completed_jobs(session_id=None, queue_factory=None):
    queue = _resolve_queue(queue_factory=queue_factory, session_id=session_id)
    removed_count = _clear_terminal_history(queue)
    headers, rows = get_queue_panel_value(session_id=session_id, queue_factory=lambda: queue)
    if removed_count == 0:
        return "当前没有可清理的完成任务", headers, rows
    return f"🧹 已清空 {removed_count} 个完成任务", headers, rows


def clear_completed_jobs_for_ui(session_id=None, queue_factory=None):
    status, _headers, rows = clear_completed_jobs(session_id=session_id, queue_factory=queue_factory)
    return status, rows


class _GuiProgressService:
    def __init__(self, base_service, gui_progress_callback):
        self._base_service = base_service
        self._gui_progress_callback = gui_progress_callback

    def process(self, params, progress_callback=None, cancel_check=None):
        def combined_progress(current, total, stage):
            if progress_callback is not None:
                progress_callback(current, total, stage)
            self._gui_progress_callback(current, total, stage)

        try:
            return self._base_service.process(
                params,
                progress_callback=combined_progress,
                cancel_check=cancel_check,
            )
        except Exception as exc:
            if cancel_check is None or "unexpected keyword argument 'cancel_check'" not in str(exc):
                raise
            return self._base_service.process(
                params,
                progress_callback=combined_progress,
            )


def process_video(
    video_path,
    mode,
    quality,
    use_ai,
    pure_color_mode,
    use_guided_filter,
    green_similarity,
    green_despill,
    green_hair,
    white_protect_thresh,
    white_protect_sat,
    edge_despill_factor,
    black_threshold,
    black_glow,
    black_particle,
    edge_softness,
    temporal_strength,
    transparency_preserve,
    gvm_max_internal_size,
    auto_optimize,
    # Chroma Key 参数
    screen_color,
    key_strength,
    clip_black,
    clip_white,
    shrink_grow,
    edge_blur,
    despill_enable,
    despill_strength,
    despill_color,
    despeckle_enable,
    despeckle_radius,
    despeckle_threshold,
    color_space,
    output_fg,
    output_matte,
    output_comp,
    output_processed,
    generate_zip,
    ai_gamma,
    ai_threshold,
    ai_gain,
    ai_sharpen,
    session_id=None,
    progress=gr.Progress(),
    service_factory=None,
    queue_factory=None,
    worker_factory=None,
):
    """处理视频 - 带实时预览"""
    if video_path is None:
        return None, None, "请先上传视频", None, None, 0, None
    try:
        validated_video_path = _validate_gui_input_path(
            video_path,
            session_id=session_id,
            require_registered=session_id is not None,
        )
    except ValueError as exc:
        return None, None, str(exc), None, None, 0, None
    if not validated_video_path.exists():
        return None, None, GUI_INPUT_PATH_MISSING_MESSAGE, None, None, 0, None
    video_path = str(validated_video_path)

    # 构建配置
    config = MattingConfig()

    if mode == "green":
        config.background_mode = BackgroundMode.GREEN_SCREEN
    elif mode == "black":
        config.background_mode = BackgroundMode.BLACK_BACKGROUND
    else:
        config.background_mode = BackgroundMode.AUTO

    if quality == "fast":
        config.quality_mode = QualityMode.FAST
    elif quality == "high":
        config.quality_mode = QualityMode.HIGH
    else:
        config.quality_mode = QualityMode.STANDARD

    # 解析抠图引擎选项
    if use_ai == "enhance":
        config.use_ai = True
        config.ai_enhance = True
        config.ai_model = "auto"
    elif use_ai == "gvm":
        config.use_ai = True
        config.ai_enhance = False
        config.ai_model = "gvm"
    elif use_ai == "matanyone2":
        config.use_ai = True
        config.ai_enhance = False
        config.ai_model = "matanyone2"
    elif use_ai in ("ai", "traditional"):
        config.use_ai = False
        config.ai_enhance = False
        config.ai_model = "auto"
    else:
        config.use_ai = False
        config.ai_enhance = False
        config.ai_model = "auto"

    config.pure_color_mode = pure_color_mode
    config.use_guided_filter = use_guided_filter

    config.green_similarity = green_similarity
    config.green_despill_strength = green_despill
    config.green_hair_detail = green_hair
    config.white_protect_brightness = white_protect_thresh
    config.white_protect_saturation = white_protect_sat
    config.edge_despill_factor = edge_despill_factor
    config.black_threshold = black_threshold
    config.black_glow_preserve = black_glow
    config.black_particle_boost = black_particle
    config.edge_softness = edge_softness
    config.temporal_strength = temporal_strength
    config.transparency_preserve = transparency_preserve
    config.gvm_max_internal_size = gvm_max_internal_size
    config.ai_enhance_gamma = ai_gamma
    config.ai_enhance_threshold = ai_threshold
    config.ai_enhance_gain = ai_gain
    config.ai_enhance_sharpen = ai_sharpen

    # Chroma Key 参数
    config.screen_color = screen_color
    config.key_strength = key_strength
    config.clip_black = clip_black
    config.clip_white = clip_white
    config.shrink_grow = shrink_grow
    config.edge_blur = edge_blur


    config.despill_enable = despill_enable
    config.despill_strength = despill_strength
    config.despill_color = despill_color
    config.despeckle_enable = despeckle_enable
    config.despeckle_radius = despeckle_radius
    config.despeckle_threshold = despeckle_threshold
    config.color_space = color_space
    config.output_fg = output_fg
    config.output_matte = output_matte
    config.output_comp = output_comp
    config.output_processed = output_processed
    config.generate_zip_by_default = generate_zip

    auto_summary = ""
    if auto_optimize:
        suggestion = suggest_input_params(video_path, config)
        apply_suggestion(config, suggestion)
        auto_summary = f"\n{suggestion.summary}{_format_actual_parameter_summary(config)}"
        logger.info("Applied auto optimization for %s: %s", video_path, suggestion)

    # 处理
    request_token = uuid4().hex
    output_dir = _resolve_request_output_dir(video_path, request_token)
    logger.info(
        "Starting GUI processing: video=%s output_dir=%s mode=%s quality=%s ai_model=%s",
        video_path,
        output_dir,
        mode,
        quality,
        use_ai,
    )
    
    logger.info(
        "GUI key config: screen_color=%s key_strength=%s clip_black=%s clip_white=%s "
        "shrink_grow=%s edge_blur=%s output_fg=%s output_matte=%s output_comp=%s output_processed=%s",
        config.screen_color,
        config.key_strength,
        config.clip_black,
        config.clip_white,
        config.shrink_grow,
        config.edge_blur,
        config.output_fg,
        config.output_matte,
        config.output_comp,
        config.output_processed,
    )
    
    try:
        def on_progress(current, total, stage):
            ratio = (current / total) if total > 0 else 0
            progress(ratio, desc=stage)

        params = _build_process_job_params(video_path, output_dir, config)
        queue = _resolve_queue(queue_factory=queue_factory, session_id=session_id)
        job = queue.submit(_build_process_job(video_path, params, job_id=request_token))

        if queue.running_job is not None and queue.running_job is not job:
            return None, None, _queued_status(queue, job), None, None, 0, None

        if job.status == JobStatus.RUNNING and queue.running_job is job:
            return None, None, "⏳ 当前任务处理中", None, None, 0, None

        if job.status == JobStatus.QUEUED:
            if queue.running_job is not None or queue.next_job() is not job:
                return None, None, _queued_status(queue, job), None, None, 0, None
            service = (service_factory or _default_service_factory)()
            worker = (worker_factory or _default_worker_factory)(queue, _GuiProgressService(service, on_progress))
            request_job = job
            completed_job = worker.run_next_job()
            if completed_job is None:
                return None, None, _queued_status(queue, job), None, None, 0, None
            job = request_job

        preview_input = None
        preview_output = None

        if job.status == JobStatus.FAILED:
            return None, None, f"❌ 处理失败: {job.error_message or 'processing failed'}", None, None, 0, None
        if job.status == JobStatus.CANCELLED:
            return None, None, "⏹ 已取消", None, None, 0, None

        job_output_dir = Path(job.result.output_dir if job.result is not None else job.params.output_dir)

        # 打包序列帧为 ZIP
        zip_path = job_output_dir / "frames.zip" if generate_zip else None
        if zip_path is not None:
            _create_zip(job_output_dir, zip_path)

        # 生成预览视频
        preview_path = job_output_dir / "preview.mp4"
        preview_quality_mode = str(job.params.config_overrides.get("preview_quality_mode", "fast"))
        preview_video_params = inspect.signature(_create_preview_video).parameters
        if "preview_quality_mode" in preview_video_params:
            _create_preview_video(
                job_output_dir,
                preview_path,
                preview_quality_mode=preview_quality_mode,
            )
        else:
            _create_preview_video(job_output_dir, preview_path)

        # 生成首帧预览图
        input_preview, output_preview = _create_preview_frames(job_output_dir, video_path)
        transparent_png_params = inspect.signature(_find_transparent_png_download).parameters
        if "session_id" in transparent_png_params:
            transparent_png = _find_transparent_png_download(job_output_dir, session_id=session_id)
        else:
            transparent_png = _find_transparent_png_download(job_output_dir)
        frame_count, processing_time, fps = _summarize_job_result(job)

        status = (
            f"✅ 完成!{frame_count}帧 | {fps:.1f} fps | 耗时 {processing_time:.1f}s{auto_summary}"
        )

        # 返回预览视频和首帧对比图
        # Gradio Video 组件需要字符串路径,且视频必须可被浏览器播放
        preview_video = None
        if preview_path.exists() and preview_path.stat().st_size > 0:
            # Gradio 需要将文件放在其工作目录下才能正确服务
            # 复制到 Gradio 的临时目录
            try:
                preview_video = _copy_preview_into_gradio_cache(
                    job,
                    preview_path,
                    job_output_dir,
                    session_id=session_id,
                )
                logger.info("Copied preview video to Gradio temp path: %s", preview_video)
            except Exception as e:
                # 如果复制失败,直接使用原路径
                preview_video = str(preview_path.resolve())
                logger.warning("Failed to copy preview to Gradio temp path, using original preview: %s", preview_video)
        else:
            logger.warning("Preview not found or empty: %s", preview_path)

        logger.info(
            "GUI processing completed: frames=%s elapsed=%.1fs fps=%.2f zip=%s png=%s preview=%s",
            frame_count,
            processing_time,
            fps,
            zip_path,
            transparent_png,
            preview_video,
        )

        return (
            preview_video,
            str(zip_path) if zip_path else None,
            status,
            input_preview,
            output_preview,
            frame_count,
            str(transparent_png) if transparent_png else None,
        )
    except Exception as e:
        logger.exception("GUI processing failed for video=%s", video_path)
        report = from_exception(e, context={"stage": "process_video", "input_path": str(video_path)})
        return None, None, _format_diagnostic_report(report), None, None, 0, None


def _format_actual_parameter_summary(config):
    """Return the effective runtime parameters after optional auto optimization."""
    return (
        "\n本次实际参数: "
        f"screen={config.screen_color}, "
        f"similarity={config.green_similarity:.2f}, "
        f"key={config.key_strength:.2f}, "
        f"preserve={config.transparency_preserve:.2f}, "
        f"despill={config.green_despill_strength:.2f}, "
        f"edge_despill={config.edge_despill_factor:.2f}, "
        f"clip={config.clip_black:.2f}/{config.clip_white:.2f}, "
        f"white_protect={config.white_protect_brightness:.0f}/{config.white_protect_saturation:.0f}, "
        f"shrink_grow={config.shrink_grow}, "
        f"edge_blur={config.edge_blur}, "
        f"gvm_size={config.gvm_max_internal_size}"
    )


def _load_input_preview_frame(input_path: Path):
    input_kind = detect_input_kind(input_path)

    if input_kind is InputKind.IMAGE:
        frames, _ = ImageDecoder().decode(input_path)
        return frames[0].astype(np.uint8)

    if input_kind is InputKind.SEQUENCE:
        frames, _ = SequenceDecoder().decode(input_path)
        return frames[0].astype(np.uint8)

    if input_kind is InputKind.VIDEO:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok or frame is None:
            raise ValueError(f"Cannot read first frame from video: {input_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

    raise ValueError(f"Unsupported preview input: {input_path}")


def _create_preview_frames(output_dir, input_path):
    """创建首帧预览对比图 - 支持子目录结构
    
    优先使用 Comp (预乘合成，背景黑色) 或 Processed (RGBA)
    避免使用 FG (未预乘，背景仍是绿色)
    """
    output_dir = Path(output_dir)
    input_preview = _load_input_preview_frame(Path(input_path))

    # 查找输出帧 - 优先顺序：Processed > Comp > FG > Matte
    frames = []
    for subdir in ["Processed", "Comp", "FG", "Matte"]:
        subdir_path = output_dir / subdir
        frames = _preview_frame_candidates(subdir_path)
        if frames:
            logger.info("Using %s frame set for preview image: %s", subdir, frames[0].name)
            break

    # 如果没有子目录,检查根目录
    if not frames:
        frames = _preview_frame_candidates(output_dir, prefix="frame_")

    if not frames:
        logger.warning("No preview frames found in %s", output_dir)
        return None, None

    # 取首帧
    img = _normalize_preview_frame(_load_preview_output_frame(frames[0]))
    h, w = img.shape[:2]

    # 创建棋盘格背景
    grid_size = 20
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            color = 180 if ((y // grid_size) + (x // grid_size)) % 2 == 0 else 120
            bg[y:y+grid_size, x:x+grid_size] = color

    # 判断使用的是哪个目录
    used_subdir = None
    for subdir in ["Processed", "Comp", "FG", "Matte"]:
        subdir_path = output_dir / subdir
        if _preview_frame_candidates(subdir_path):
            used_subdir = subdir
            break

    # 合成输出预览
    if used_subdir == "Comp":
        # Comp 是预乘合成，直接显示在棋盘格上（背景已经是黑色）
        rgb = img[:, :, :3].astype(np.float32)
        # 简单混合：如果像素接近黑色，显示棋盘格
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        mask = (gray > 10).astype(np.float32)[:, :, np.newaxis]  # 非黑区域
        output_preview = (rgb * mask + bg * (1 - mask)).astype(np.uint8)
    elif used_subdir == "Processed" and img.shape[2] == 4:
        # Processed 是 RGBA，使用 alpha 混合
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        output_preview = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    elif used_subdir == "FG":
        # FG 是未预乘的，背景仍是绿色，需要特殊处理
        rgb = img[:, :, :3].astype(np.float32)
        # 简单去绿：检测绿色背景并替换为棋盘格
        green_mask = (img[:, :, 1] > img[:, :, 0] * 1.2) & (img[:, :, 1] > img[:, :, 2] * 1.2)
        mask = (~green_mask).astype(np.float32)[:, :, np.newaxis]
        output_preview = (rgb * mask + bg * (1 - mask)).astype(np.uint8)
    else:
        # 默认：直接显示
        output_preview = img[:, :, :3].astype(np.uint8)

    return input_preview, output_preview


def _normalize_preview_frame(image_array):
    if image_array.ndim == 2:
        return np.stack([image_array] * 3, axis=-1)
    if image_array.ndim == 3 and image_array.shape[2] in (3, 4):
        return image_array
    raise ValueError(f"Unsupported preview frame shape: {image_array.shape}")


def _preview_frame_candidates(directory: Path, prefix: str | None = None) -> list[Path]:
    if not directory.exists():
        return []
    candidates = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in PREVIEW_OUTPUT_EXTENSIONS
    ]
    if prefix is not None:
        candidates = [path for path in candidates if path.name.startswith(prefix)]
    return sorted(candidates)


def _load_preview_output_frame(frame_path: Path) -> np.ndarray:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        try:
            frame = np.array(Image.open(frame_path))
        except Exception as exc:
            raise ValueError(f"Cannot open preview frame: {frame_path}") from exc
        return frame

    if frame.dtype.kind == "f":
        frame = np.clip(frame, 0.0, 1.0)
        frame = (frame * 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported preview frame shape: {frame.shape}")


def _convert_preview_frame_to_png(frame_path: Path, session_id: str | None = None) -> Path | None:
    try:
        frame = _load_preview_output_frame(frame_path)
    except ValueError:
        logger.warning("Failed to convert preview frame to PNG: %s", frame_path, exc_info=True)
        return None

    download_dir = _gradio_preview_root() / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    output_path = download_dir / f"{frame_path.stem}.png"
    Image.fromarray(frame).save(output_path)
    _session_preview_artifact_registry.register_path(session_id, output_path)
    return output_path


def _select_preview_frame_paths(frames: list[Path], preview_quality_mode: str) -> list[Path]:
    if preview_quality_mode != "fast" or len(frames) <= PREVIEW_FAST_FRAME_LIMIT:
        return frames

    sampled_indices = np.linspace(0, len(frames) - 1, num=PREVIEW_FAST_FRAME_LIMIT, dtype=int)
    return [frames[index] for index in sampled_indices]


def _find_transparent_png_download(output_dir, session_id=None):
    """Return the first RGBA processed PNG for direct single-frame download."""
    processed_dir = Path(output_dir) / "Processed"
    if not processed_dir.exists():
        logger.info("Processed output directory does not exist for PNG download: %s", processed_dir)
        return None

    png_frames = sorted(processed_dir.glob("*.png"))
    if png_frames:
        return png_frames[0]

    frames = _preview_frame_candidates(processed_dir)
    if not frames:
        logger.info("No processed preview frames available for download in %s", processed_dir)
        return None

    return _convert_preview_frame_to_png(frames[0], session_id=session_id)


def _create_preview_video(output_dir, preview_path, preview_quality_mode="fast"):
    """创建带棋盘格背景的预览视频 - 使用 imageio 确保浏览器兼容
    
    优先使用 Comp (预乘合成，背景黑色) 或 Processed (RGBA)
    避免使用 FG (未预乘，背景仍是绿色)
    """

    # 查找输出帧 - 优先顺序：Processed > Comp > FG > Matte
    frames = []
    for subdir in ["Processed", "Comp", "FG", "Matte"]:
        subdir_path = output_dir / subdir
        frames = _preview_frame_candidates(subdir_path)
        if frames:
            logger.info("Using %s frame set for preview video", subdir)
            break

    # 如果没有子目录,检查根目录
    if not frames:
        frames = _preview_frame_candidates(output_dir, prefix="frame_")

    if not frames:
        logger.warning("No frames found for preview video in %s", output_dir)
        return

    frames = _select_preview_frame_paths(frames, preview_quality_mode)

    try:
        import imageio
    except ImportError:
        logger.warning("imageio not available, falling back to cv2 preview writer")
        _create_preview_video_cv2(output_dir, preview_path, preview_quality_mode=preview_quality_mode)
        return

    first = _normalize_preview_frame(_load_preview_output_frame(frames[0]))
    h, w = first.shape[:2]

    # 创建棋盘格背景
    grid_size = 20
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            color = 180 if ((y // grid_size) + (x // grid_size)) % 2 == 0 else 120
            bg[y:y+grid_size, x:x+grid_size] = color

    # 使用 imageio 写入 MP4 (H.264)
    preview_path = preview_path.with_suffix('.mp4')

    try:
        # 尝试使用 imageio-ffmpeg 插件,使用 H.264 编码确保浏览器兼容
        writer = imageio.get_writer(
            str(preview_path),
            fps=30,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p',  # 确保浏览器兼容
            macro_block_size=1      # 避免自动调整尺寸
        )
    except Exception as e:
        logger.warning("imageio H.264 writer failed, trying default writer: %s", e)
        try:
            writer = imageio.get_writer(str(preview_path), fps=30)
        except Exception as e2:
            logger.warning("imageio default writer failed, falling back to cv2: %s", e2)
            _create_preview_video_cv2(output_dir, preview_path, preview_quality_mode=preview_quality_mode)
            return

    frame_count = 0
    for frame_path in frames:
        rgba = _normalize_preview_frame(_load_preview_output_frame(frame_path))

        # 处理 RGBA 或 RGB
        if rgba.shape[2] == 4:
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            rgb = rgba[:, :, :3]
        else:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            rgb = rgba

        # 合成到棋盘格背景
        composed = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        writer.append_data(composed)
        frame_count += 1

    writer.close()
    logger.info("Created preview video: %s (%s frames)", preview_path, frame_count)


def _create_preview_video_cv2(output_dir, preview_path, preview_quality_mode="fast"):
    """备用:使用 OpenCV 创建预览视频"""

    # 查找输出帧
    frames = []
    for subdir in ["Processed", "Comp", "FG", "Matte"]:
        subdir_path = output_dir / subdir
        frames = _preview_frame_candidates(subdir_path)
        if frames:
            break

    if not frames:
        frames = _preview_frame_candidates(output_dir, prefix="frame_")

    if not frames:
        return

    frames = _select_preview_frame_paths(frames, preview_quality_mode)

    first = _normalize_preview_frame(_load_preview_output_frame(frames[0]))
    h, w = first.shape[:2]

    # 使用 mp4v 编码
    preview_path = preview_path.with_suffix('.mp4')
    fourcc = video_writer_fourcc("mp4v", cv2)
    writer = cv2.VideoWriter(str(preview_path), fourcc, 30.0, (w, h))

    if not writer.isOpened():
        logger.warning("CV2 preview writer failed to open for %s", preview_path)
        return

    # 创建棋盘格背景
    grid_size = 20
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            color = 180 if ((y // grid_size) + (x // grid_size)) % 2 == 0 else 120
            bg[y:y+grid_size, x:x+grid_size] = color

    for frame_path in frames:
        rgba = _normalize_preview_frame(_load_preview_output_frame(frame_path))
        if rgba.shape[2] == 4:
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            rgb = rgba[:, :, :3]
        else:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            rgb = rgba

        composed = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        composed_bgr = cv2.cvtColor(composed, cv2.COLOR_RGB2BGR)
        writer.write(composed_bgr)

    writer.release()
    logger.info("Created preview video with cv2 fallback: %s", preview_path)


def _create_zip(output_dir, zip_path):
    """打包输出序列帧为 ZIP。"""
    # 检查所有可能的输出目录
    all_frames = []
    for subdir in ["FG", "Matte", "Comp", "Processed"]:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            frames = _preview_frame_candidates(subdir_path)
            all_frames.extend(frames)

    # 如果没有子目录,检查根目录
    if not all_frames:
        all_frames = _preview_frame_candidates(output_dir, prefix="frame_")

    if not all_frames:
        logger.warning("No frames found to pack into zip under %s", output_dir)
        # 创建一个空 ZIP
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            pass
        return

    logger.info("Packing %s frames into zip: %s", len(all_frames), zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for frame_path in all_frames:
            # 使用相对路径保持目录结构
            arcname = frame_path.relative_to(output_dir)
            zf.write(frame_path, arcname)


# CSS 样式
custom_css = """
    .tab-button { font-size: 14px; }
    .param-group { border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; margin: 5px 0; }
    .preview-box { background: #f5f5f5; border-radius: 8px; padding: 10px; }
"""

def create_ui():
    """创建 Gradio UI"""

    with gr.Blocks(title="MatteFlow - 专业视频/序列帧/图片抠图") as app:
        session_state = gr.State()

        # 顶部标题栏
        with gr.Row():
            gr.Markdown("""
            # 🎬 MatteFlow
            **CorridorKey 物理分离 | BiRefNet AI | 传统色度键**
            """)

        # 主工作区:左右布局
        with gr.Row():

            # 左侧:输入与参数面板
            with gr.Column(scale=1, min_width=350):

                # 文件导入区
                with gr.Group():
                    gr.Markdown("### 📁 导入")
                    video_input = gr.File(
                        label="拖放视频或图片文件",
                        file_types=SUPPORTED_UPLOAD_EXTENSIONS,
                        type="filepath",
                        height=120,
                    )
                    upload_image_preview = gr.Image(
                        label="素材预览",
                        interactive=False,
                        visible=False,
                        height=220,
                    )
                    upload_video_preview = gr.Video(
                        label="素材预览",
                        interactive=False,
                        visible=False,
                        height=220,
                    )
                    upload_status = gr.Markdown("未选择素材")

                # 引擎选择
                with gr.Group():
                    gr.Markdown("### ⚙️ Alpha 生成器")

                    mode_select = gr.Radio(
                        choices=[("🟢 绿幕", "green"), ("⚫ 黑底", "black"), ("🔍 自动识别", "auto")],
                        value=GUI_DEFAULTS["mode"],
                        label="背景模式"
                    )

                    # 动态生成模型选项
                    ai_select = gr.Radio(
                        choices=_ui_choices if _ui_choices else [("📐 传统算法", "traditional")],
                        value=_default_ai_choice(),
                        label="Alpha 生成器 (✅=可用, ❌=未安装)"
                    )

                    quality_select = gr.Radio(
                        choices=[("⚡ 快速", "fast"), ("✨ 标准", "standard"), ("🎨 高质量", "high")],
                        value=GUI_DEFAULTS["quality"],
                        label="质量模式"
                    )


                # Chroma Key 参数
                with gr.Group(visible=True) as green_params:
                    gr.Markdown("### 🟢 推荐核心参数")

                    with gr.Group():
                        key_strength = gr.Slider(
                            0.6, 1.4, value=GUI_DEFAULTS["key_strength"], step=0.05,
                            label="Key Strength",
                            info="抠像强度。过高容易抠透明，过低容易留绿"
                        )
                        transparency_preserve = gr.Slider(
                            0.0, 1.0,
                            value=GUI_DEFAULTS["transparency_preserve"],
                            step=0.05,
                            label="半透明保留",
                            info="提高会保留爱心辉光/特效，过高可能保留背景雾边"
                        )
                        green_despill = gr.Slider(
                            0.0, 1.0,
                            value=GUI_DEFAULTS["green_despill"],
                            step=0.05,
                            label="去绿边强度"
                        )
                        edge_despill_factor = gr.Slider(
                            0.5, 2.0,
                            value=GUI_DEFAULTS["edge_despill_factor"],
                            step=0.1,
                            label="去绿系数"
                        )

                    with gr.Row():
                        shrink_grow = gr.Slider(
                            -5, 5, value=GUI_DEFAULTS["shrink_grow"], step=1,
                            label="Shrink/Grow",
                            info="负值收边，正值补边"
                        )
                        edge_blur = gr.Slider(
                            0, 5, value=GUI_DEFAULTS["edge_blur"], step=1,
                            label="Edge Blur",
                            info="边缘柔化半径"
                        )

                    gvm_max_internal_size = gr.Radio(
                        choices=[("512 快速", 512), ("768 推荐", 768), ("1024 高质量", 1024)],
                        value=GUI_DEFAULTS["gvm_max_internal_size"],
                        label="GVM 推理尺寸"
                    )
                    auto_optimize = gr.Checkbox(
                        value=GUI_DEFAULTS["auto_optimize"],
                        label="自动优化参数",
                        info="图片直接分析当前图；视频/序列帧默认抽中间帧，本次处理临时优化参数"
                    )

                    with gr.Accordion("高级绿幕参数", open=False):
                        screen_color = gr.Radio(
                            choices=[("🟢 绿色", "green"), ("🔵 蓝色", "blue"), ("🔍 自动检测", "auto")],
                            value=GUI_DEFAULTS["screen_color"],
                            label="屏幕颜色"
                        )
                        green_sim = gr.Slider(0.1, 1.0, value=GUI_DEFAULTS["green_similarity"], step=0.05, label="颜色相似度")
                        green_hair = gr.Slider(
                            0.0, 1.0,
                            value=GUI_DEFAULTS["green_hair"],
                            step=0.05,
                            label="毛发保护",
                            visible=False,
                        )
                        white_protect_thresh = gr.Slider(150, 255, value=GUI_DEFAULTS["white_protect_brightness"], step=5, label="白色保护亮度")
                        white_protect_sat = gr.Slider(10, 60, value=GUI_DEFAULTS["white_protect_saturation"], step=1, label="白色保护饱和度")
                        with gr.Row():
                            clip_black = gr.Slider(
                                0.0, 1.0, value=GUI_DEFAULTS["clip_black"], step=0.01,
                                label="Clip Black",
                                info="危险参数：提高会吃掉辉光/耳朵边缘"
                            )
                            clip_white = gr.Slider(
                                0.0, 1.0, value=GUI_DEFAULTS["clip_white"], step=0.01,
                                label="Clip White",
                                info="降低会拉实主体，也可能加重雾边"
                            )

                    pure_color = gr.Checkbox(value=GUI_FIXED_PARAMETER_DEFAULTS["pure_color"], visible=False)
                    use_filter = gr.Checkbox(value=GUI_FIXED_PARAMETER_DEFAULTS["use_filter"], visible=False)

                # 黑底参数
                with gr.Group(visible=False) as black_params:
                    gr.Markdown("### ⚫ 黑底参数")

                    black_thresh = gr.Slider(0.0, 0.2, value=0.03, step=0.01, label="黑场阈值")
                    black_glow = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="辉光保留")
                    black_particle = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="粒子增强")

                # 低频通用参数固定为推荐值，避免普通用户误调坏效果。
                edge_soft = gr.Slider(0.0, 1.0, value=GUI_FIXED_PARAMETER_DEFAULTS["edge_softness"], visible=False)
                temporal_str = gr.Slider(0.0, 1.0, value=GUI_FIXED_PARAMETER_DEFAULTS["temporal_strength"], visible=False)

                # 推理控制
                with gr.Group():
                    gr.Markdown("### 🎛️ 高级与输出")

                    despill_enable = gr.Checkbox(value=GUI_FIXED_PARAMETER_DEFAULTS["despill_enable"], visible=False)
                    despill_strength = gr.Slider(
                        0.0, 1.0,
                        value=GUI_DEFAULTS["despill_strength"],
                        step=0.05,
                        label="去溢色强度",
                        visible=False
                    )
                    despill_color = gr.Radio(
                        choices=[("绿色", "green"), ("蓝色", "blue"), ("自动", "auto")],
                        value=GUI_FIXED_PARAMETER_DEFAULTS["despill_color"],
                        visible=False
                    )

                    with gr.Accordion("去噪点", open=False):
                        despeckle_enable = gr.Checkbox(value=GUI_DEFAULTS["despeckle_enable"], label="启用去噪点")
                        despeckle_radius = gr.Slider(1, 5, value=GUI_DEFAULTS["despeckle_radius"], step=1, label="去噪点半径")
                        despeckle_threshold = gr.Slider(
                            0.0, 1.0,
                            value=GUI_DEFAULTS["despeckle_threshold"],
                            step=0.05,
                            label="去噪点阈值",
                            info="提高会删除低透明噪点，也可能削掉辉光/半透明细节"
                        )

                    color_space = gr.Radio(
                        choices=[("sRGB", "sRGB"), ("Rec.709", "Rec709"), ("Linear", "Linear"), ("ACES", "ACES")],
                        value=GUI_FIXED_PARAMETER_DEFAULTS["color_space"],
                        visible=False
                    )

                    with gr.Accordion("输出设置", open=False):
                        with gr.Row():
                            output_fg = gr.Checkbox(value=GUI_DEFAULTS["output_fg"], label="FG (直接前景)")
                            output_matte = gr.Checkbox(value=GUI_DEFAULTS["output_matte"], label="Matte (Alpha)")
                        with gr.Row():
                            output_comp = gr.Checkbox(value=GUI_DEFAULTS["output_comp"], label="Comp (预乘)")
                            output_processed = gr.Checkbox(value=GUI_DEFAULTS["output_processed"], label="Processed (RGBA)")
                        generate_zip = gr.Checkbox(value=GUI_DEFAULTS["generate_zip"], label="打包 ZIP 下载")

                # AI 增强参数
                with gr.Group(visible=False) as ai_params:
                    gr.Markdown("### 🤖 AI 参数")
                    ai_gamma = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Gamma")
                    ai_threshold = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="泄漏阈值")
                    ai_gain = gr.Slider(1.0, 3.0, value=1.2, step=0.1, label="增益")
                    ai_sharpen = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="锐化")

                # 处理按钮
                with gr.Row():
                    preset_btn = gr.Button("↩ 恢复推荐参数", variant="secondary")
                    process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                    run_all_btn = gr.Button("▶ 运行全部", variant="secondary")
                    cancel_btn = gr.Button("⏹ 取消当前任务", variant="secondary")
                    clear_completed_btn = gr.Button("🧹 清空完成任务", variant="secondary")

                # 状态栏
                status_text = gr.Textbox(
                    label="状态",
                    value="就绪",
                    interactive=False,
                    lines=2
                )

                with gr.Group():
                    gr.Markdown("### 队列面板")
                    queue_table = gr.Dataframe(
                        headers=QUEUE_TABLE_HEADERS,
                        value=_queue_panel_rows_only(),
                        interactive=False,
                        wrap=True,
                    )

            # 右侧:预览与输出面板
            with gr.Column(scale=2):

                # 视频预览
                with gr.Group():
                    gr.Markdown("### 🎬 结果预览")
                    result_preview = gr.Video(
                        label="棋盘格背景预览",
                        interactive=False,
                        height=400
                    )

                    # 首帧对比图
                    with gr.Row():
                        input_preview = gr.Image(
                            label="输入帧",
                            interactive=False,
                            height=200
                        )
                        output_preview = gr.Image(
                            label="输出帧",
                            interactive=False,
                            height=200
                        )

                # 输出下载
                with gr.Group():
                    gr.Markdown("### 💾 输出")
                    with gr.Row():
                        frames_zip = gr.File(
                            label="PNG 序列帧 (ZIP)",
                            interactive=False
                        )
                        transparent_png = gr.File(
                            label="单帧透明 PNG",
                            interactive=False
                        )

                        frame_count = gr.Number(
                            label="处理帧数",
                            value=0,
                            interactive=False
                        )

        # 模式切换显示/隐藏参数
        def toggle_mode(mode):
            return {
                green_params: gr.update(visible=(mode == "green" or mode == "auto")),
                black_params: gr.update(visible=(mode == "black"))
            }

        mode_select.change(
            fn=toggle_mode,
            inputs=[mode_select],
            outputs=[green_params, black_params]
        )

        # AI 参数显示/隐藏
        def toggle_ai_params(choice):
            return gr.update(visible=(choice == "enhance"))

        ai_select.change(
            fn=toggle_ai_params,
            inputs=[ai_select],
            outputs=[ai_params]
        )

        video_input.change(
            fn=_create_upload_preview,
            inputs=[video_input, session_state],
            outputs=[upload_image_preview, upload_video_preview, upload_status],
        )

        preset_outputs = [
            mode_select,
            quality_select,
            ai_select,
            pure_color,
            use_filter,
            green_sim,
            green_despill,
            green_hair,
            white_protect_thresh,
            white_protect_sat,
            edge_despill_factor,
            screen_color,
            key_strength,
            clip_black,
            clip_white,
            shrink_grow,
            edge_blur,
            despill_enable,
            despill_strength,
            despill_color,
            despeckle_enable,
            despeckle_radius,
            despeckle_threshold,
            transparency_preserve,
            gvm_max_internal_size,
            auto_optimize,
            generate_zip,
            output_fg,
            output_matte,
            output_comp,
            output_processed,
        ]

        app.load(fn=_new_session_id, outputs=[session_state])
        app.load(fn=_recommended_preset_updates, outputs=preset_outputs)

        preset_btn.click(
            fn=_recommended_preset_updates,
            outputs=preset_outputs,
        )

        # 绑定处理事件
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                mode_select,
                quality_select,
                ai_select,
                pure_color,
                use_filter,
                green_sim,
                green_despill,
                green_hair,
                white_protect_thresh,
                white_protect_sat,
                edge_despill_factor,
                black_thresh,
                black_glow,
                black_particle,
                edge_soft,
                temporal_str,
                transparency_preserve,
                gvm_max_internal_size,
                auto_optimize,
                # Chroma Key 参数
                screen_color,
                key_strength,
                clip_black,
                clip_white,
                shrink_grow,
                edge_blur,
                despill_enable,
                despill_strength,
                despill_color,
                despeckle_enable,
                despeckle_radius,
                despeckle_threshold,
                color_space,
                output_fg,
                output_matte,
                output_comp,
                output_processed,
                generate_zip,
                ai_gamma,
                ai_threshold,
                ai_gain,
                ai_sharpen,
                session_state,
            ],
            outputs=[
                result_preview,
                frames_zip,
                status_text,
                input_preview,
                output_preview,
                frame_count,
                transparent_png,
            ]
        ).then(
            fn=_queue_panel_rows_only,
            inputs=[session_state],
            outputs=[queue_table],
        )

        run_all_btn.click(
            fn=run_all_jobs_for_ui,
            inputs=[session_state],
            outputs=[status_text, queue_table],
        )

        cancel_btn.click(
            fn=cancel_current_job_with_panel,
            inputs=[session_state],
            outputs=[status_text, queue_table],
        )

        clear_completed_btn.click(
            fn=clear_completed_jobs_for_ui,
            inputs=[session_state],
            outputs=[status_text, queue_table],
        )

        # 底部说明
        gr.Markdown("""
        ---
        ### 📖 快速指南

        **推荐流程:**
        1. 上传视频或图片(视频: MP4/MOV/AVI/MKV/WEBM; 图片: PNG/JPG/JPEG/WEBP/BMP/TIF/TIFF/EXR)
        2. 选择背景模式(绿幕/黑底/自动)
        3. 选择算法(CorridorKey 推荐用于绿幕)
        4. 调整参数(通常默认即可)
        5. 点击「开始处理」

        **算法选择:**
        - 🏆 **CorridorKey**:绿幕最佳,无需背景图,保留毛发/半透明
        - 🤖 **AI 增强**:传统 + AI 边缘细化,平衡速度质量
        - 📐 **传统算法**:快速,适合简单场景

        **快捷键:**
        - Ctrl+Enter:开始处理
        - R:重置参数
        """)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    _configure_logging(args.debug)
    logger.info("Available models: %s", _available_models)
    if args.debug:
        logger.info("Debug mode enabled")

    app = create_ui()
    # 禁用 SSR 模式,避免启动额外进程
    import os
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = str(args.port)
    logger.info("Launching Gradio UI on port=%s share=%s", args.port, args.share)

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        prevent_thread_lock=False,
        quiet=True,
        css=custom_css,
        ssr_mode=False
    )


if __name__ == "__main__":
    main()
