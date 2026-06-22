import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.errors import InputValidationError
from matteflow.input.decoder import SequenceDecoder


def test_sequence_decoder_source_files_only_include_loaded_frames(tmp_path):
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(tmp_path / "001.png")
    (tmp_path / "002.png").write_bytes(b"not a valid png")
    Image.fromarray(np.ones((2, 2, 3), dtype=np.uint8) * 255, mode="RGB").save(
        tmp_path / "003.png"
    )

    frames, meta = SequenceDecoder().decode(tmp_path)

    assert len(frames) == 2
    assert meta["source_files"] == ["001.png", "003.png"]


def test_sequence_decoder_rejects_more_than_max_frames_while_loading(tmp_path):
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(tmp_path / "001.png")
    Image.fromarray(np.ones((2, 2, 3), dtype=np.uint8) * 255, mode="RGB").save(
        tmp_path / "002.png"
    )

    with pytest.raises(InputValidationError, match="exceeds configured max_input_frames=1"):
        SequenceDecoder(max_frames=1).decode(tmp_path)
