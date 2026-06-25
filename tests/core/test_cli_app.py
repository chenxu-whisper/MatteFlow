import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow import cli_app  # noqa: E402


def test_main_runs_verify_preview_cleanup_subcommand(monkeypatch):
    captured = {}

    def fake_verify_main():
        captured["called"] = True
        return 0

    monkeypatch.setattr(cli_app, "verify_preview_cleanup_main", fake_verify_main)

    assert cli_app.main(["verify-preview-cleanup"]) == 0
    assert captured["called"] is True


def test_build_parser_accepts_verify_preview_cleanup_subcommand():
    parser = cli_app.build_parser()

    args = parser.parse_args(["verify-preview-cleanup"])

    assert args.command == "verify-preview-cleanup"


def test_build_parser_accepts_quality_regression_subcommand():
    parser = cli_app.build_parser()

    args = parser.parse_args(["quality-regression", "--reports", "reports"])

    assert args.command == "quality-regression"
    assert args.reports == "reports"


def test_build_parser_keeps_legacy_top_level_process_arguments():
    parser = cli_app.build_parser()

    args = parser.parse_args(["--input", "sample.mp4"])

    assert args.command == "process"
    assert args.input == "sample.mp4"


def test_build_config_enables_quality_selection_from_cli_flag():
    parser = cli_app.build_parser()

    args = parser.parse_args(["--input", "sample.mp4", "--quality-selection"])
    config = cli_app._build_config(args)

    assert config.quality_selection_enable is True


def test_build_config_sets_quality_birefnet_auto_load():
    parser = cli_app.build_parser()

    args = parser.parse_args(
        ["--input", "sample.mp4", "--quality-selection", "--quality-birefnet-auto-load"]
    )
    config = cli_app._build_config(args)

    assert config.quality_selection_enable is True
    assert config.quality_birefnet_auto_load is True


def test_main_keeps_legacy_top_level_process_arguments(monkeypatch):
    captured = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def process(self, input_path, output_dir, on_progress):
            captured["input_path"] = input_path
            captured["output_dir"] = output_dir
            captured["on_progress"] = on_progress
            return {
                "frames_processed": 1,
                "elapsed_time": 0.1,
                "fps": 10.0,
                "output_dir": str(output_dir),
            }

    monkeypatch.setattr(cli_app, "MattingPipeline", FakePipeline)
    monkeypatch.setattr(cli_app, "_configure_logging", lambda debug: None)

    assert cli_app.main(["--input", "sample.mp4"]) == 0
    assert captured["input_path"] == "sample.mp4"


def test_main_runs_quality_regression_subcommand(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports" / "sample"
    report_dir.mkdir(parents=True)
    (report_dir / "processing_report.json").write_text(
        json.dumps(
            {
                "job": {"input_path": "sample.png"},
                "quality": {
                    "overall_score": 0.70,
                    "mean_edge_uncertainty": 0.10,
                    "hole_pixels": 30,
                    "background_residue": 0.02,
                    "temporal_flicker": 0.01,
                },
                "regions": {},
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "quality_regression.json"
    output_md = tmp_path / "quality_regression.md"
    monkeypatch.setattr(cli_app, "_configure_logging", lambda debug: None)

    exit_code = cli_app.main(
        [
            "quality-regression",
            "--reports",
            str(tmp_path / "reports"),
            "--min-overall-score",
            "0.80",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert exit_code == 1
    assert json.loads(output_json.read_text(encoding="utf-8"))["summary"]["failed_count"] == 1
    assert "sample" in output_md.read_text(encoding="utf-8")


def test_module_entrypoint_runs_verify_preview_cleanup_subcommand():
    env = dict(os.environ)
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

    result = subprocess.run(
        [sys.executable, "-m", "matteflow", "verify-preview-cleanup"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )

    assert "Before cleanup:" in result.stdout
    assert "After cleanup:" in result.stdout
    assert "File Not Found" not in result.stdout
    assert "File Not Found" not in result.stderr
