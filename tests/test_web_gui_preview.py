import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import web_gui


def test_create_preview_frames_prefers_processed_rgba_over_comp(tmp_path):
    output_dir = tmp_path / "out"
    processed_dir = output_dir / "Processed"
    comp_dir = output_dir / "Comp"
    processed_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:, :, :3] = [255, 200, 200]
    rgba[:, :, 3] = 255
    rgba[0, 0, :3] = [0, 0, 255]
    rgba[0, 0, 3] = 8
    Image.fromarray(rgba, mode="RGBA").save(processed_dir / "processed_000000.png")

    comp = np.zeros((4, 4, 3), dtype=np.uint8)
    comp[:, :, :] = [255, 200, 200]
    comp[0, 0, :] = [0, 0, 8]
    Image.fromarray(comp, mode="RGB").save(comp_dir / "comp_000000.png")

    _, output_preview = web_gui._create_preview_frames(output_dir)

    assert output_preview is not None
    assert output_preview[0, 0, 2] > output_preview[0, 0, 0]
    assert output_preview[0, 0, 2] > output_preview[0, 0, 1]
