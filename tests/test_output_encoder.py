import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.output.encoder import RGBAEncoder


def test_rgba_encoder_writes_float_rgba_png(tmp_path):
    out_path = tmp_path / "rgba.png"
    rgba = np.zeros((2, 2, 4), dtype=np.float32)
    rgba[..., 0] = 0.5
    rgba[..., 3] = 0.75

    RGBAEncoder().encode_image(rgba, out_path)

    loaded = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
    assert loaded is not None
    assert loaded.shape == (2, 2, 4)
    assert loaded[0, 0, 2] in (127, 128)
    assert loaded[0, 0, 3] in (191, 192)


def test_rgba_encoder_writes_grayscale_png(tmp_path):
    out_path = tmp_path / "mask.png"
    matte = np.array([[0, 127], [191, 255]], dtype=np.uint8)

    RGBAEncoder().encode_grayscale(matte, out_path)

    loaded = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
    assert loaded is not None
    assert loaded.shape == (2, 2)
    assert int(loaded[1, 0]) == 191
