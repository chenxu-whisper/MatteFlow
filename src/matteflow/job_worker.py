"""Minimal serial worker for queued MatteFlow jobs."""

from __future__ import annotations

from typing import Any

from .errors import JobCancelledError, ProcessingError
from .job_queue import GPUJob, GPUJobQueue, JobType
from .service import ProcessResult


class JobWorker:
    """Consumes one queued job at a time via the service layer."""

    def __init__(self, queue: GPUJobQueue, service: Any) -> None:
        self._queue = queue
        self._service = service

    def run_next_job(self) -> GPUJob | None:
        """Run the next queued job and persist its final state."""
        job = self._queue.claim_next_job()
        if job is None:
            return None

        try:
            self._run_job(job)
        except JobCancelledError:
            self._queue.cancel_job(job)
        except Exception as exc:
            self._queue.fail_job(job, exc)
        else:
            self._queue.complete_job(job)

        return job

    def _run_job(self, job: GPUJob) -> None:
        if job.job_type != JobType.PROCESS_MEDIA:
            raise NotImplementedError(f"Unsupported job type: {job.job_type}")

        job.result = self._service.process(
            job.params,
            progress_callback=lambda current, total, stage: job.update_progress(current, total, stage),
            cancel_check=lambda: job.is_cancelled,
        )
        if isinstance(job.result, ProcessResult) and not job.result.success:
            raise ProcessingError(job.result.error_message or "processing failed")
