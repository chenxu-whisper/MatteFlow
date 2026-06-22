import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.job_queue import GPUJob, GPUJobQueue, JobStatus, JobType
from matteflow.job_worker import JobWorker
from matteflow.service import ProcessJobParams, ProcessResult
from scripts import web_gui


class FakeProgress:
    def __init__(self):
        self.calls = []

    def __call__(self, value, desc=None):
        self.calls.append((value, desc))


def _job(tmp_path, name, *, job_type=JobType.PROCESS_MEDIA):
    input_path = tmp_path / name
    return GPUJob(
        job_type=job_type,
        input_path=input_path,
        params=ProcessJobParams(
            input_path=input_path,
            output_dir=tmp_path / f"{input_path.stem}-out",
        ),
    )


def _base_process_kwargs(tmp_path, progress):
    return dict(
        video_path=str(tmp_path / "input.png"),
        mode="green",
        quality="standard",
        use_ai="gvm",
        pure_color_mode=True,
        use_guided_filter=False,
        green_similarity=0.4,
        green_despill=0.7,
        green_hair=0.8,
        white_protect_thresh=180,
        white_protect_sat=25,
        white_ring_cleanup_strength=1.0,
        glow_feather_strength=1.0,
        edge_despill_factor=1.2,
        black_threshold=0.03,
        black_glow=0.9,
        black_particle=0.7,
        edge_softness=0.0,
        temporal_strength=0.5,
        transparency_preserve=0.7,
        gvm_max_internal_size=768,
        auto_optimize=False,
        screen_color="auto",
        key_strength=1.0,
        clip_black=0.0,
        clip_white=1.0,
        shrink_grow=0,
        edge_blur=0,
        despill_enable=True,
        despill_strength=0.7,
        despill_color="green",
        despeckle_enable=True,
        despeckle_radius=2,
        despeckle_threshold=0.0,
        color_space="sRGB",
        output_fg=False,
        output_matte=True,
        output_comp=False,
        output_processed=True,
        generate_zip=False,
        ai_gamma=0.8,
        ai_threshold=0.1,
        ai_gain=1.2,
        ai_sharpen=0.0,
        progress=progress,
    )


def test_process_video_returns_queued_status_when_another_job_is_running(monkeypatch, tmp_path):
    progress = FakeProgress()
    queue = GPUJobQueue()
    output_dir = tmp_path / "out"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(tmp_path / "input.png")
    running_job = queue.submit(
        GPUJob(
            job_type=JobType.PROCESS_MEDIA,
            input_path=tmp_path / "already-running.png",
            params=ProcessJobParams(
                input_path=tmp_path / "already-running.png",
                output_dir=tmp_path / "running-out",
            ),
        )
    )
    queue.start_job(running_job)

    class FailingWorker:
        def __init__(self, queue, service):
            raise AssertionError("worker should not be created when job is only queued")

    monkeypatch.setattr(web_gui, "_resolve_gui_output_dir", lambda video_path: output_dir)

    result = web_gui.process_video(
        **_base_process_kwargs(tmp_path, progress),
        queue_factory=lambda: queue,
        worker_factory=lambda q, service: FailingWorker(q, service),
        service_factory=lambda: object(),
    )

    assert queue.running_job is running_job
    assert len(queue.queued_snapshot) == 1
    assert queue.queued_snapshot[0].status == JobStatus.QUEUED
    assert result == (None, None, "⏳ 已入队，第 1 位", None, None, 0, None)
    assert progress.calls == []


def test_process_video_failed_job_reports_real_error_message(monkeypatch, tmp_path):
    progress = FakeProgress()
    queue = GPUJobQueue()
    output_dir = tmp_path / "out"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(tmp_path / "input.png")

    class FailingService:
        def process(self, params, progress_callback=None, cancel_check=None):
            raise RuntimeError("真实处理错误")

    monkeypatch.setattr(web_gui, "_resolve_gui_output_dir", lambda video_path: output_dir)

    result = web_gui.process_video(
        **_base_process_kwargs(tmp_path, progress),
        queue_factory=lambda: queue,
        worker_factory=lambda q, service: JobWorker(q, service),
        service_factory=lambda: FailingService(),
    )

    assert queue.history_snapshot[0].status == JobStatus.FAILED
    assert queue.history_snapshot[0].error_message == "真实处理错误"
    assert "真实处理错误" in result[2]
    assert "GPUJob" not in result[2]


def test_cancel_current_job_requests_cancellation_for_running_job(tmp_path):
    queue = GPUJobQueue()
    running_job = queue.submit(
        GPUJob(
            job_type=JobType.PROCESS_MEDIA,
            input_path=tmp_path / "running.png",
            params=ProcessJobParams(
                input_path=tmp_path / "running.png",
                output_dir=tmp_path / "running-out",
            ),
        )
    )
    queue.start_job(running_job)

    status = web_gui.cancel_current_job(queue_factory=lambda: queue)

    assert status == "⏹ 已请求取消当前任务"
    assert queue.running_job is running_job
    assert running_job.status == JobStatus.RUNNING
    assert running_job.is_cancelled is True
    assert queue.history_snapshot == ()


def test_format_queue_rows_includes_running_queued_and_history_jobs(tmp_path):
    queue = GPUJobQueue()
    running_job = _job(tmp_path, "running.png")
    queued_job = _job(tmp_path, "queued.png")
    completed_job = _job(tmp_path, "done.png")
    failed_job = _job(tmp_path, "failed.png")

    queue.submit(running_job)
    queue.submit(queued_job)
    queue.start_job(running_job)
    running_job.update_progress(3, 5, "matting")
    completed_job.status = JobStatus.COMPLETED
    completed_job.stage = "completed"
    completed_job.current = 2
    completed_job.total = 2
    failed_job.status = JobStatus.FAILED
    failed_job.stage = "failed"
    failed_job.error_message = "boom"
    queue._history.extend([completed_job, failed_job])

    rows = web_gui._format_queue_rows(queue)

    assert rows[0][1:] == [
        "process_media",
        "running",
        "3/5 (60%)",
        "running.png",
        "matting",
    ]
    assert rows[1][2] == "queued"
    assert rows[1][4] == "queued.png"
    assert any(row[2] == "completed" and row[4] == "done.png" for row in rows)
    assert any(row[2] == "failed" and row[5] == "boom" for row in rows)


def test_get_queue_panel_value_returns_headers_and_rows(tmp_path):
    queue = GPUJobQueue()
    queue.submit(_job(tmp_path, "queued.png"))

    headers, rows = web_gui.get_queue_panel_value(queue_factory=lambda: queue)

    assert headers == web_gui.QUEUE_TABLE_HEADERS
    assert len(rows) == 1
    assert rows[0][2] == "queued"


def test_run_all_jobs_consumes_all_queued_jobs(tmp_path):
    queue = GPUJobQueue()
    processed = []

    class FakeService:
        def process(self, params, progress_callback=None, cancel_check=None):
            del cancel_check
            processed.append(params.input_path.name)
            if progress_callback is not None:
                progress_callback(1, 1, "encoding")
            return ProcessResult(
                success=True,
                input_path=params.input_path,
                output_dir=params.output_dir,
                background_mode="green_screen",
                frame_count=1,
                processing_time=0.1,
                timings={},
            )

    first = queue.submit(_job(tmp_path, "first.png"))
    second = queue.submit(_job(tmp_path, "second.png"))

    status, headers, rows = web_gui.run_all_jobs(
        queue_factory=lambda: queue,
        worker_factory=lambda q, service: JobWorker(q, service),
        service_factory=lambda: FakeService(),
    )

    assert status == "▶ 队列已全部运行完成"
    assert headers == web_gui.QUEUE_TABLE_HEADERS
    assert processed == ["first.png", "second.png"]
    assert queue.running_job is None
    assert queue.queued_snapshot == ()
    assert [job.status for job in queue.history_snapshot] == [JobStatus.COMPLETED, JobStatus.COMPLETED]
    assert {row[4] for row in rows} == {"first.png", "second.png"}
    assert first.current == 1
    assert second.current == 1


def test_clear_completed_jobs_only_removes_terminal_history(tmp_path):
    queue = GPUJobQueue()
    queued_job = queue.submit(_job(tmp_path, "queued.png"))
    completed_job = _job(tmp_path, "done.png")
    cancelled_job = _job(tmp_path, "cancelled.png")
    running_job = _job(tmp_path, "running.png")

    queue.submit(running_job)
    queue.start_job(running_job)
    completed_job.status = JobStatus.COMPLETED
    completed_job.stage = "completed"
    cancelled_job.status = JobStatus.CANCELLED
    cancelled_job.stage = "cancelled"
    queue._history.extend([completed_job, cancelled_job])

    status, headers, rows = web_gui.clear_completed_jobs(queue_factory=lambda: queue)

    assert status == "🧹 已清空 2 个完成任务"
    assert headers == web_gui.QUEUE_TABLE_HEADERS
    assert queue.running_job is running_job
    assert queue.queued_snapshot == (queued_job,)
    assert queue.history_snapshot == ()
    assert [row[2] for row in rows] == ["running", "queued"]


def test_process_video_drains_next_queued_jobs_after_current_job_finishes(monkeypatch, tmp_path):
    progress = FakeProgress()
    queue = GPUJobQueue()
    output_dir = tmp_path / "out"
    transparent_png = output_dir / "Processed" / "processed_000000.png"
    captured = {"processed_inputs": []}
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(tmp_path / "input.png")

    class FakeService:
        def process(self, params, progress_callback=None, cancel_check=None):
            captured["processed_inputs"].append(params.input_path.name)
            params.output_dir.mkdir(parents=True, exist_ok=True)
            transparent_png.parent.mkdir(parents=True, exist_ok=True)
            transparent_png.write_bytes(b"png")

            if params.input_path.name == "input.png":
                queued_job = GPUJob(
                    job_type=JobType.PROCESS_MEDIA,
                    input_path=tmp_path / "queued-next.png",
                    params=ProcessJobParams(
                        input_path=tmp_path / "queued-next.png",
                        output_dir=tmp_path / "queued-out",
                    ),
                )
                queue.submit(queued_job)
                progress_callback(4, 4, "encoding")
                return ProcessResult(
                    success=True,
                    input_path=params.input_path,
                    output_dir=params.output_dir,
                    background_mode="green_screen",
                    frame_count=4,
                    processing_time=1.5,
                    timings={"decode": 0.2},
                )

            return ProcessResult(
                success=True,
                input_path=params.input_path,
                output_dir=params.output_dir,
                background_mode="green_screen",
                frame_count=2,
                processing_time=0.5,
                timings={"decode": 0.1},
            )

    monkeypatch.setattr(web_gui, "_resolve_gui_output_dir", lambda video_path: output_dir)
    monkeypatch.setattr(web_gui, "_create_preview_video", lambda output_dir, preview_path: None)
    monkeypatch.setattr(
        web_gui,
        "_create_preview_frames",
        lambda output_dir, input_path: (
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.zeros((2, 2, 4), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(web_gui, "_find_transparent_png_download", lambda output_dir: transparent_png)

    result = web_gui.process_video(
        **_base_process_kwargs(tmp_path, progress),
        queue_factory=lambda: queue,
        worker_factory=lambda q, service: JobWorker(q, service),
        service_factory=lambda: FakeService(),
    )

    assert queue.running_job is None
    assert len(queue.queued_snapshot) == 1
    assert queue.queued_snapshot[0].input_path.name == "queued-next.png"
    assert len(queue.history_snapshot) == 1
    assert [job.status for job in queue.history_snapshot] == [JobStatus.COMPLETED]
    assert captured["processed_inputs"] == ["input.png"]
    assert progress.calls == [(1.0, "encoding")]
    assert "完成" in result[2]
    assert result[5] == 4
    assert result[6] == str(transparent_png)
