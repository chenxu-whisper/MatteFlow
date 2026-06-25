"""Shared MatteFlow CLI implementation."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import BackgroundMode, MattingConfig, QualityMode
from .evaluation import (
    QualityRegressionBaseline,
    QualityRegressionEvaluator,
    QualityRegressionThresholds,
)
from .input.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .pipeline import MattingPipeline
from .utils.output_paths import resolve_project_output_dir
from .verify_preview_cleanup import main as verify_preview_cleanup_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _resolve_output_dir(
    input_path: Path,
    output_arg: str | None,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    if output_arg:
        return Path(output_arg)
    return resolve_project_output_dir(input_path, project_root=project_root)


def _configure_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _add_process_arguments(parser: argparse.ArgumentParser) -> None:
    supported_video = ", ".join(sorted(VIDEO_EXTENSIONS))
    supported_image = ", ".join(sorted(IMAGE_EXTENSIONS))
    parser.add_argument(
        "--input",
        "-i",
        help=f"输入视频、单张图片或序列帧目录。视频: {supported_video}; 图片: {supported_image}",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="输出目录，默认使用 temp/output/<输入文件名>/",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "green", "black"],
        default="auto",
        help="背景模式 (default: auto)",
    )
    parser.add_argument(
        "--quality",
        choices=["fast", "standard", "high"],
        default="standard",
        help="质量模式 (default: standard)",
    )
    parser.add_argument("--mask", action="store_true", help="输出黑白遮罩")
    parser.add_argument("--debug", action="store_true", help="输出调试信息")
    parser.add_argument(
        "--quality-selection",
        action="store_true",
        help="启用实验性候选质量选择系统",
    )
    parser.set_defaults(command="process")


def _add_quality_regression_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reports", required=True, help="包含 processing_report.json 的目录或单个报告文件")
    parser.add_argument("--baseline", help="可选 baseline JSON 文件")
    parser.add_argument("--output-json", help="写入聚合评测 JSON")
    parser.add_argument("--output-md", help="写入 Markdown 评测报告")
    parser.add_argument("--min-overall-score", type=float, default=0.80)
    parser.add_argument("--max-mean-edge-uncertainty", type=float, default=0.08)
    parser.add_argument("--max-hole-pixels", type=int, default=100)
    parser.add_argument("--max-background-residue", type=float, default=0.02)
    parser.add_argument("--max-temporal-flicker", type=float, default=0.08)
    parser.add_argument("--max-score-drop", type=float, default=0.05)
    parser.set_defaults(command="quality-regression")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MatteFlow - High-quality video, sequence, and image matting"
    )

    subparsers = parser.add_subparsers(dest="command")
    verify_parser = subparsers.add_parser(
        "verify-preview-cleanup",
        help="验证 Gradio 预览缓存的纯时间驱动清理逻辑",
    )
    verify_parser.set_defaults(command="verify-preview-cleanup")
    quality_regression_parser = subparsers.add_parser(
        "quality-regression",
        help="聚合 processing_report.json 并执行质量回归检查",
    )
    _add_quality_regression_arguments(quality_regression_parser)

    _add_process_arguments(parser)
    return parser


def _build_config(args: argparse.Namespace) -> MattingConfig:
    config = MattingConfig()

    if args.mode == "green":
        config.background_mode = BackgroundMode.GREEN_SCREEN
    elif args.mode == "black":
        config.background_mode = BackgroundMode.BLACK_BACKGROUND
    else:
        config.background_mode = BackgroundMode.AUTO

    if args.quality == "fast":
        config.quality_mode = QualityMode.FAST
    elif args.quality == "high":
        config.quality_mode = QualityMode.HIGH
    else:
        config.quality_mode = QualityMode.STANDARD

    config.output_mask = args.mask
    config.output_debug = args.debug
    config.quality_selection_enable = bool(getattr(args, "quality_selection", False))
    return config


def _run_quality_regression(args: argparse.Namespace) -> int:
    baseline = QualityRegressionBaseline.from_path(args.baseline) if args.baseline else None
    thresholds = QualityRegressionThresholds(
        min_overall_score=args.min_overall_score,
        max_mean_edge_uncertainty=args.max_mean_edge_uncertainty,
        max_hole_pixels=args.max_hole_pixels,
        max_background_residue=args.max_background_residue,
        max_temporal_flicker=args.max_temporal_flicker,
        max_score_drop=args.max_score_drop,
    )
    result = QualityRegressionEvaluator(thresholds=thresholds, baseline=baseline).evaluate_root(
        args.reports
    )

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(result.to_markdown(), encoding="utf-8")

    print(result.to_markdown(), end="")
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "verify-preview-cleanup":
        return verify_preview_cleanup_main()
    if args.command == "quality-regression":
        _configure_logging(getattr(args, "debug", False))
        return _run_quality_regression(args)

    if not args.input:
        parser.error("the following arguments are required: --input/-i")

    _configure_logging(args.debug)

    config = _build_config(args)
    pipeline = MattingPipeline(config)

    def on_progress(current, total, stage):
        pct = int(current / total * 100) if total else 100
        print(f"\r[MatteFlow] {stage}: {pct}%", end="", flush=True)

    output_dir = _resolve_output_dir(Path(args.input), args.output)

    logger.info(
        "Starting CLI matting: input=%s output=%s mode=%s quality=%s mask=%s debug=%s",
        args.input,
        output_dir,
        config.background_mode.value,
        config.quality_mode.value,
        args.mask,
        args.debug,
    )

    try:
        result = pipeline.process(args.input, str(output_dir), on_progress)
        print()
        logger.info(
            "CLI matting completed: frames=%s time=%.1fs fps=%.2f output=%s",
            result["frames_processed"],
            result["elapsed_time"],
            result["fps"],
            result["output_dir"],
        )
        return 0
    except Exception as exc:
        print()
        logger.exception("CLI matting failed: %s", exc)
        return 1
