"""GPU job queue primitives for serial MatteFlow workloads."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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
        self._queued: Deque[GPUJob] = deque()
        self._history: list[GPUJob] = []
        self._running: Optional[GPUJob] = None

    @property
    def queued_snapshot(self) -> tuple[GPUJob, ...]:
        return tuple(self._queued)

    @property
    def history_snapshot(self) -> tuple[GPUJob, ...]:
        return tuple(self._history)

    @property
    def running_job(self) -> Optional[GPUJob]:
        return self._running

    def submit(self, job: GPUJob) -> GPUJob:
        """Submit a job, deduplicating normal work and replacing previews."""
        if job.job_type == JobType.PREVIEW_REPROCESS:
            self._cancel_queued_previews(job.input_path)
        else:
            existing = self._find_active_duplicate(job)
            if existing is not None:
                return existing

        job.status = JobStatus.QUEUED
        job.stage = "queued"
        self._queued.append(job)
        return job

    def next_job(self) -> Optional[GPUJob]:
        """Peek at the next queued job without changing its state."""
        if not self._queued:
            return None
        return self._queued[0]

    def start_job(self, job: GPUJob) -> None:
        """Move a queued job into running state."""
        try:
            self._queued.remove(job)
        except ValueError:
            pass
        job.status = JobStatus.RUNNING
        job.stage = "running"
        self._running = job

    def complete_job(self, job: GPUJob) -> None:
        """Mark a running job as completed and archive it."""
        job.status = JobStatus.COMPLETED
        if job.total == 0:
            job.total = job.current
        else:
            job.current = job.total
        job.stage = "completed"
        self._finish(job)

    def fail_job(self, job: GPUJob, error: Exception | str) -> None:
        """Mark a job as failed and archive it."""
        job.status = JobStatus.FAILED
        job.error_message = str(error)
        job.stage = "failed"
        self._finish(job)

    def cancel_job(self, job_or_id: GPUJob | str) -> bool:
        """Cancel a queued job or request cancellation for a running job."""
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
        if self._running is not None:
            self._running.request_cancel()

        while self._queued:
            job = self._queued.popleft()
            job.request_cancel()
            self._archive_cancelled(job)

    def clear_history(self, statuses: Optional[set[JobStatus]] = None) -> int:
        """Remove archived jobs matching terminal statuses and return count."""
        target_statuses = statuses or {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
        remaining = [job for job in self._history if job.status not in target_statuses]
        removed = len(self._history) - len(remaining)
        self._history = remaining
        return removed

    def _find_active_duplicate(self, job: GPUJob) -> Optional[GPUJob]:
        for candidate in self._queued:
            if self._same_identity(candidate, job):
                return candidate
        if self._running is not None and self._same_identity(self._running, job):
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
        return left.input_path == right.input_path and left.job_type == right.job_type

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

    def _finish(self, job: GPUJob) -> None:
        if self._running is job:
            self._running = None
        if job not in self._history:
            self._history.append(job)
