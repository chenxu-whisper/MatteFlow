import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.__main__ import build_parser


def test_cli_help_mentions_video_image_and_sequence_support():
    parser = build_parser()

    assert "支持视频 / 图片 / 序列帧" in parser.description
    input_action = next(action for action in parser._actions if action.dest == "input")
    assert "输入媒体路径" in input_action.help


def test_cli_rejects_conflicting_ai_flags():
    parser = build_parser()

    try:
        parser.parse_args(["--input", "in.png", "--output", "out", "--ai", "--no-ai"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected conflicting --ai/--no-ai flags to be rejected")
