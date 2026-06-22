import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.matte.birefnet_matte import BiRefNetMatte


def test_birefnet_postprocess_accepts_list_outputs():
    matte = BiRefNetMatte.__new__(BiRefNetMatte)
    pred = [torch.full((1, 1, 2, 3), 0.75, dtype=torch.float32)]

    alpha = matte._postprocess(pred, (4, 6))

    assert alpha.shape == (4, 6)
    assert alpha.dtype == np.float32
    assert float(alpha.mean()) == 0.75
