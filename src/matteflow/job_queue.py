"""GPU job queue primitives for serial MatteFlow workloads."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Deque, Optional
from uuid import uuid4

from .service import ProcessJobParams, ProcessResult


class JobType(str, Enum):
    """Types of GPU-heavy jobs that can be queued."""

    PROCESS_MEDIA = "process_media"
    GVM_ALPHA = "gvm_alpha"
    PREVIEW_REPROCESS = "preview_reprocess"


class JobStatus(str, Enum):
    """Lifecycle state for a queued GPU job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class GPUJob:
    """A single immutable-identity job with mutable runtime status."""

    job_type: JobType
    input_path: str | Path
    params: ProcessJobParams
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.QUEUED
    error_message: Optional[str] = None
    current: int = 0
    total: int = 0
    stage: str = "queued"
    result: Optional[ProcessResult] = None
    _cancel_requested: bool = False

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        if not isinstance(self.job_type, JobType):
            self.job_type = JobType(self.job_type)
        if not isinstance(self.status, JobStatus):
            self.status = JobStatus(self.status)

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested or finalized."""
        return self._cancel_requested or self.status == JobStatus.CANCELLED

    def request_cancel(self) -> None:
        """Mark the job for cancellation."""
        self._cancel_requested = True

    def update_progress(self, current: int, total: int, stage: str) -> None:
        """Update progress values reported by a future worker/service layer."""
        self.current = current
        self.total = total
        self.stage = stage


class GPUJobQueue:
    """In-memory FIFO queue for one-at-a-time GPU processing."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._queued: Deque[GPUJob] = deque()
        self._history: list[GPUJob] = []
        self._running: Optional[GPUJob] = None

    @property
    def queued_snapshot(self) -> tuple[GPUJob, ...]:
        with self._lock:
            return tuple(self._queued)

    @property
    def history_snapshot(self) -> tuple[GPUJob, ...]:
        with self._lock:
            return tuple(self._history)

    @property
    def running_job(self) -> Optional[GPUJob]:
        with self._lock:
            return self._running

    def submit(self, job: GPUJob) -> GPUJob:
        """Submit a job, deduplicating normal work and replacing previews."""
        with self._lock:
            if job.job_type == JobType.PREVIEW_REPROCESS:
                self._cancel_queued_previews(job.input_path)
            elif job.job_type != JobType.PROCESS_MEDIA:
                existing = self._find_active_duplicate(job)
                if existing is not None:
                    return existing

            self._reset_job_for_submit(job)
            self._queued.append(job)
            return job

    def next_job(self) -> Optional[GPUJob]:
        """Peek at the next queued job without changing its state."""
        with self._lock:
            if not self._queued:
                return None
            return self._queued[0]

    def claim_next_job(self) -> Optional[GPUJob]:
        """Atomically move the next queued job into running state."""
        with self._lock:
            if self._running is not None or not self._queued:
                return None
            job = self._queued.popleft()
            job.status = JobStatus.RUNNING
            job.stage = "running"
            self._running = job
            return job

    def start_job(self, job: GPUJob) -> None:
        """Move a queued job into running state."""
        with self._lock:
            if self._running is not None and self._running is not job:
                raise RuntimeError("A job is already running")
            try:
                self._queued.remove(job)
            except ValueError as exc:
                raise ValueError("Cannot start job that is not queued") from exc
            job.status = JobStatus.RUNNING
            job.stage = "running"
            self._running = job

    def complete_job(self, job: GPUJob) -> None:
        """Mark a running job as completed and archive it."""
        with self._lock:
            job.status = JobStatus.COMPLETED
            if job.total == 0:
                job.total = job.current
            else:
                job.current = job.total
            job.stage = "completed"
            self._finish(job)

    def fail_job(self, job: GPUJob, error: Exception | str) -> None:
        """Mark a job as failed and archive it."""
        with self._lock:
            job.status = JobStatus.FAILED
            job.error_message = str(error)
            job.stage = "failed"
            self._finish(job)

    def cancel_job(self, job_or_id: GPUJob | str) -> bool:
        """Cancel a queued job or request cancellation for a running job."""
        with self._lock:
            job = self._resolve_job(job_or_id)
            if job is None:
                return False

            job.request_cancel()
            if job.status == JobStatus.QUEUED:
                self._remove_queued(job)
                self._archive_cancelled(job)
            elif job.status == JobStatus.RUNNING and job_or_id is job:
                self._archive_cancelled(job)
            return True

    def cancel_all(self) -> None:
        """Cancel all queued jobs and request cancellation for the running job."""
        with self._lock:
            if self._running is not None:
                self._running.request_cancel()

            while self._queued:
                job = self._queued.popleft()
                job.request_cancel()
                self._archive_cancelled(job)

    def clear_history(self, statuses: Optional[set[JobStatus]] = None) -> int:
        """Remove archived jobs matching terminal statuses and return count."""
        with self._lock:
            target_statuses = statuses or {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            remaining = [job for job in self._history if job.status not in target_statuses]
            removed = len(self._history) - len(remaining)
            self._history = remaining
            return removed

    def _find_active_duplicate(self, job: GPUJob) -> Optional[GPUJob]:
        for candidate in self._queued:
            if self._same_identity(candidate, job):
                return candidate
        if (
            self._running is not None
            and not self._running.is_cancelled
            and self._same_identity(self._running, job)
        ):
            return self._running
        return None

    def _cancel_queued_previews(self, input_path: Path) -> None:
        for job in list(self._queued):
            if job.job_type == JobType.PREVIEW_REPROCESS and job.input_path == input_path:
                self._remove_queued(job)
                job.request_cancel()
                self._archive_cancelled(job)

    @staticmethod
    def _same_identity(left: GPUJob, right: GPUJob) -> bool:
        left_params = left.params
        right_params = right.params
        return (
            left.job_type == right.job_type
            and left.input_path == right.input_path
            and left_params.input_path == right_params.input_path
            and left_params.background_mode == right_params.background_mode
            and left_params.quality_mode == right_params.quality_mode
            and left_params.use_ai == right_params.use_ai
            and left_params.ai_model == right_params.ai_model
            and dict(left_params.config_overrides) == dict(right_params.config_overrides)
        )

    def _resolve_job(self, job_or_id: GPUJob | str) -> Optional[GPUJob]:
        if isinstance(job_or_id, GPUJob):
            return job_or_id
        if self._running is not None and self._running.id == job_or_id:
            return self._running
        for job in self._queued:
            if job.id == job_or_id:
                return job
        return None

    def _remove_queued(self, job: GPUJob) -> None:
        try:
            self._queued.remove(job)
        except ValueError:
            pass

    def _archive_cancelled(self, job: GPUJob) -> None:
        job.status = JobStatus.CANCELLED
        job.stage = "cancelled"
        self._finish(job)

    @staticmethod
    def _snapshot_job(job: GPUJob) -> GPUJob:
        result = job.result
        result_snapshot = None
        if isinstance(result, ProcessResult):
            result_snapshot = ProcessResult(
                success=result.success,
                input_path=result.input_path,
                output_dir=result.output_dir,
                background_mode=result.background_mode,
                frame_count=result.frame_count,
                processing_time=result.processing_time,
                timings=dict(result.timings),
                error_message=result.error_message,
            )
        elif result is not None:
            result_snapshot = result
        return GPUJob(
            job_type=job.job_type,
            input_path=job.input_path,
            params=ProcessJobParams(
                input_path=job.params.input_path,
                output_dir=job.params.output_dir,
                background_mode=job.params.background_mode,
                quality_mode=job.params.quality_mode,
                use_ai=job.params.use_ai,
                ai_model=job.params.ai_model,
                quality_selection_enable=job.params.quality_selection_enable,
                quality_birefnet_auto_load=job.params.quality_birefnet_auto_load,
                config_overrides=dict(job.params.config_overrides),
            ),
            id=job.id,
            status=job.status,
            error_message=job.error_message,
            current=job.current,
            total=job.total,
            stage=job.stage,
            result=result_snapshot,
            _cancel_requested=job._cancel_requested,
        )

    @staticmethod
    def _reset_job_for_submit(job: GPUJob) -> None:
        job.status = JobStatus.QUEUED
        job.error_message = None
        job.current = 0
        job.total = 0
        job.stage = "queued"
        job.result = None
        job._cancel_requested = False

    def _finish(self, job: GPUJob) -> None:
        if self._running is job:
            self._running = None
        self._history.append(self._snapshot_job(job))
