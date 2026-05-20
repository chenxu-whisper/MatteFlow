import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.errors import JobCancelledError
from matteflow.job_queue import GPUJob, GPUJobQueue, JobStatus, JobType
from matteflow.service import ProcessJobParams
from matteflow.job_worker import JobWorker


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


def test_job_worker_completes_process_media_job(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))

    class FakeService:
        def process(self, params, progress_callback=None, cancel_check=None):
            progress_callback(1, 3, "matting")
            progress_callback(3, 3, "encoding")
            return object()

    worker = JobWorker(queue, FakeService())

    completed = worker.run_next_job()

    assert completed is job
    assert job.status == JobStatus.COMPLETED
    assert job.current == 3
    assert job.total == 3
    assert job.stage == "completed"
    assert queue.history_snapshot == (job,)


def test_job_worker_marks_job_failed_when_service_errors(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))

    class FailingService:
        def process(self, params, progress_callback=None, cancel_check=None):
            raise RuntimeError("boom")

    worker = JobWorker(queue, FailingService())

    worker.run_next_job()

    assert job.status == JobStatus.FAILED
    assert job.error_message == "boom"
    assert queue.history_snapshot == (job,)


def test_job_worker_marks_job_cancelled_when_service_raises_job_cancelled(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))

    class CancelledService:
        def process(self, params, progress_callback=None, cancel_check=None):
            raise JobCancelledError("cancelled")

    worker = JobWorker(queue, CancelledService())

    worker.run_next_job()

    assert job.status == JobStatus.CANCELLED
    assert job.is_cancelled is True
    assert queue.history_snapshot == (job,)


def test_job_worker_cancels_running_job_via_queue_job_id(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))
    captured = {}

    class CancelDuringRunService:
        def process(self, params, progress_callback=None, cancel_check=None):
            progress_callback(1, 4, "matting")
            assert queue.running_job is job

            cancelled = queue.cancel_job(job.id)
            captured["cancelled"] = cancelled
            captured["cancel_requested"] = cancel_check()

            raise JobCancelledError("cancelled during processing")

    worker = JobWorker(queue, CancelDuringRunService())

    worker.run_next_job()

    assert captured["cancelled"] is True
    assert captured["cancel_requested"] is True
    assert job.status == JobStatus.CANCELLED
    assert job.is_cancelled is True
    assert job.current == 1
    assert job.total == 4
    assert job.stage == "cancelled"
    assert queue.running_job is None
    assert queue.history_snapshot == (job,)


def test_job_worker_keeps_running_job_until_cancelled_job_is_finalized(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))
    captured = {}

    class CancelDuringRunService:
        def process(self, params, progress_callback=None, cancel_check=None):
            progress_callback(1, 4, "matting")
            captured["running_before_cancel"] = queue.running_job
            captured["history_before_cancel"] = queue.history_snapshot

            queue.cancel_job(job.id)

            captured["running_after_cancel_request"] = queue.running_job
            captured["history_after_cancel_request"] = queue.history_snapshot
            captured["cancel_requested"] = cancel_check()

            raise JobCancelledError("cancelled during processing")

    worker = JobWorker(queue, CancelDuringRunService())

    worker.run_next_job()

    assert captured["running_before_cancel"] is job
    assert captured["history_before_cancel"] == ()
    assert captured["running_after_cancel_request"] is job
    assert captured["history_after_cancel_request"] == ()
    assert captured["cancel_requested"] is True
    assert queue.running_job is None
    assert queue.history_snapshot == (job,)
    assert job.status == JobStatus.CANCELLED


def test_job_worker_cancel_all_preserves_running_job_until_worker_finalizes_it(tmp_path):
    queue = GPUJobQueue()
    running_job = queue.submit(make_job(tmp_path, input_name="input-running.png", output_name="out-running"))
    queued_job = queue.submit(make_job(tmp_path, input_name="input-queued.png", output_name="out-queued"))
    captured = {}

    class CancelAllDuringRunService:
        def process(self, params, progress_callback=None, cancel_check=None):
            progress_callback(1, 4, "matting")
            captured["running_before_cancel_all"] = queue.running_job
            captured["queued_before_cancel_all"] = queue.queued_snapshot
            captured["history_before_cancel_all"] = queue.history_snapshot

            queue.cancel_all()

            captured["running_after_cancel_all"] = queue.running_job
            captured["queued_after_cancel_all"] = queue.queued_snapshot
            captured["history_after_cancel_all"] = queue.history_snapshot
            captured["cancel_requested"] = cancel_check()

            raise JobCancelledError("cancelled by cancel_all")

    worker = JobWorker(queue, CancelAllDuringRunService())

    worker.run_next_job()

    assert captured["running_before_cancel_all"] is running_job
    assert captured["queued_before_cancel_all"] == (queued_job,)
    assert captured["history_before_cancel_all"] == ()
    assert captured["running_after_cancel_all"] is running_job
    assert captured["queued_after_cancel_all"] == ()
    assert captured["history_after_cancel_all"] == (queued_job,)
    assert captured["cancel_requested"] is True

    assert running_job.status == JobStatus.CANCELLED
    assert queued_job.status == JobStatus.CANCELLED
    assert queue.running_job is None
    assert queue.queued_snapshot == ()
    assert queue.history_snapshot == (queued_job, running_job)


def test_job_worker_writes_progress_back_to_job(tmp_path):
    queue = GPUJobQueue()
    job = queue.submit(make_job(tmp_path))
    captured = {}

    class ProgressService:
        def process(self, params, progress_callback=None, cancel_check=None):
            progress_callback(2, 5, "matting")
            captured["cancel_requested"] = cancel_check()
            progress_callback(5, 5, "encoding")
            return object()

    worker = JobWorker(queue, ProgressService())

    worker.run_next_job()

    assert captured["cancel_requested"] is False
    assert job.current == 5
    assert job.total == 5
    assert job.stage == "completed"


def test_job_worker_noops_when_queue_empty():
    queue = GPUJobQueue()

    class FakeService:
        def process(self, params, progress_callback=None, cancel_check=None):
            raise AssertionError("service should not be called")

    worker = JobWorker(queue, FakeService())

    assert worker.run_next_job() is None
    assert queue.history_snapshot == ()
