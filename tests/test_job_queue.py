import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.job_queue import GPUJob, GPUJobQueue, JobStatus, JobType
from matteflow.service import ProcessJobParams


def make_job(
    tmp_path,
    *,
    job_type=JobType.PROCESS_MEDIA,
    input_name="input.png",
    output_name="out",
):
    return GPUJob(
        job_type=job_type,
        input_path=tmp_path / input_name,
        params=ProcessJobParams(
            input_path=tmp_path / input_name,
            output_dir=tmp_path / output_name,
        ),
    )


def test_job_queue_runs_lifecycle_from_queued_to_completed(tmp_path):
    queue = GPUJobQueue()
    job = make_job(tmp_path)

    submitted = queue.submit(job)
    next_job = queue.next_job()

    assert submitted is job
    assert next_job is job
    assert job.status == JobStatus.QUEUED

    queue.start_job(job)
    job.update_progress(2, 5, "matting")
    queue.complete_job(job)

    assert job.status == JobStatus.COMPLETED
    assert job.current == 5
    assert job.total == 5
    assert job.stage == "completed"
    assert queue.history_snapshot == (job,)
    assert queue.next_job() is None


def test_job_queue_marks_running_job_failed(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))

    queue.start_job(job)
    queue.fail_job(job, RuntimeError("boom"))

    assert job.status == JobStatus.FAILED
    assert job.error_message == "boom"
    assert queue.history_snapshot == (job,)


def test_job_queue_can_cancel_queued_job(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))

    assert queue.cancel_job(job.id) is True

    assert job.status == JobStatus.CANCELLED
    assert job.is_cancelled is True
    assert queue.next_job() is None
    assert queue.history_snapshot == (job,)


def test_job_queue_requests_cancel_for_running_job(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))
    queue.start_job(job)

    assert queue.cancel_job(job.id) is True

    assert job.status == JobStatus.RUNNING
    assert job.is_cancelled is True

    queue.cancel_job(job)
    assert job.status == JobStatus.CANCELLED
    assert queue.history_snapshot == (job,)


def test_job_queue_deduplicates_same_input_and_job_type(tmp_path):
    queue = GPUJobQueue()
    first = queue.submit(make_job(tmp_path, input_name="same.png"))
    duplicate = make_job(tmp_path, input_name="same.png", output_name="out2")

    second = queue.submit(duplicate)

    assert second is first
    assert queue.queued_snapshot == (first,)


def test_job_queue_allows_same_input_with_different_job_type(tmp_path):
    queue = GPUJobQueue()
    process_job = queue.submit(make_job(tmp_path, input_name="same.png"))
    alpha_job = queue.submit(
        make_job(
            tmp_path,
            job_type=JobType.GVM_ALPHA,
            input_name="same.png",
            output_name="alpha",
        )
    )

    assert alpha_job is not process_job
    assert queue.queued_snapshot == (process_job, alpha_job)


def test_preview_reprocess_jobs_keep_only_latest(tmp_path):
    queue = GPUJobQueue()
    first = queue.submit(
        make_job(
            tmp_path,
            job_type=JobType.PREVIEW_REPROCESS,
            input_name="preview.png",
            output_name="preview1",
        )
    )
    latest = queue.submit(
        make_job(
            tmp_path,
            job_type=JobType.PREVIEW_REPROCESS,
            input_name="preview.png",
            output_name="preview2",
        )
    )

    assert first.status == JobStatus.CANCELLED
    assert latest is not first
    assert queue.queued_snapshot == (latest,)
    assert queue.history_snapshot == (first,)


def test_cancel_all_cancels_queued_and_requests_running_cancel(tmp_path):
    queue = GPUJobQueue()
    running = queue.submit(make_job(tmp_path, input_name="running.png"))
    queued = queue.submit(make_job(tmp_path, input_name="queued.png"))
    queue.start_job(running)

    queue.cancel_all()

    assert running.status == JobStatus.RUNNING
    assert running.is_cancelled is True
    assert queued.status == JobStatus.CANCELLED
    assert queue.queued_snapshot == ()
    assert queued in queue.history_snapshot
