import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.errors import ProcessingError
from matteflow.ffmpeg_env import MediaToolDiscoveryResult


def test_from_media_tools_flags_missing_ffprobe():
    from matteflow.diagnostics import DiagnosticCode, DiagnosticSeverity, from_media_tools

    report = from_media_tools(
        MediaToolDiscoveryResult(
            ffmpeg_path="C:/ffmpeg/bin/ffmpeg.exe",
            ffprobe_path=None,
            bin_dir="C:/ffmpeg/bin",
            source="imageio_ffmpeg",
            complete=False,
            download_required=True,
        )
    )

    assert report.ok is False
    assert report.blocking_count == 1
    assert report.items[0].code is DiagnosticCode.FFPROBE_NOT_FOUND
    assert report.items[0].severity is DiagnosticSeverity.ERROR


def test_from_media_tools_flags_missing_toolchain():
    from matteflow.diagnostics import DiagnosticCode, from_media_tools

    report = from_media_tools(
        MediaToolDiscoveryResult(
            ffmpeg_path=None,
            ffprobe_path=None,
            bin_dir=None,
            source=None,
            complete=False,
            download_required=True,
        )
    )

    assert report.ok is False
    assert report.items[0].code is DiagnosticCode.FFMPEG_NOT_FOUND


def test_from_exception_maps_cuda_oom():
    from matteflow.diagnostics import DiagnosticCode, from_exception

    report = from_exception(RuntimeError("CUDA out of memory"))

    assert report.ok is False
    assert report.items[0].code is DiagnosticCode.GPU_OUT_OF_MEMORY
    assert report.items[0].blocking is True


def test_from_exception_maps_unknown_processing_error():
    from matteflow.diagnostics import DiagnosticCode, from_exception

    report = from_exception(ProcessingError("boom"))

    assert report.items[0].code is DiagnosticCode.UNKNOWN_PROCESSING_ERROR


def test_merge_reports_deduplicates_by_code_and_summary():
    from matteflow.diagnostics import from_exception, merge_reports

    report_a = from_exception(RuntimeError("CUDA out of memory"))
    report_b = from_exception(RuntimeError("CUDA out of memory"))

    merged = merge_reports(report_a, report_b)

    assert len(merged.items) == 1
    assert merged.blocking_count == 1


def test_from_media_tools_accepts_complete_toolchain():
    from matteflow.diagnostics import from_media_tools

    report = from_media_tools(
        MediaToolDiscoveryResult(
            ffmpeg_path="C:/ffmpeg/bin/ffmpeg.exe",
            ffprobe_path="C:/ffmpeg/bin/ffprobe.exe",
            bin_dir="C:/ffmpeg/bin",
            source="path",
            complete=True,
            download_required=False,
        )
    )

    assert report.ok is True
    assert report.items == ()


def test_from_model_status_maps_missing_model():
    from matteflow.diagnostics import DiagnosticCode, from_model_status

    report = from_model_status(
        {
            "corridorkey": {
                "display_name": "CorridorKey",
                "available": False,
                "path": "C:/models/corridorkey.pth",
                "reason": "需要手动下载",
                "auto_download": False,
            }
        }
    )

    assert report.ok is False
    assert report.items[0].code is DiagnosticCode.MODEL_MISSING


def test_from_model_status_maps_runtime_import_failure():
    from matteflow.diagnostics import DiagnosticCode, from_model_status

    report = from_model_status(
        {
            "gvm": {
                "display_name": "GVM",
                "available": False,
                "path": "C:/models/gvm",
                "reason": "GVM vendored runtime 不可导入",
                "auto_download": False,
            }
        }
    )

    assert report.items[0].code is DiagnosticCode.MODEL_RUNTIME_IMPORT_FAILED
