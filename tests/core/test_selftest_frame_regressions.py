import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from matteflow.config import BackgroundMode, MattingConfig
from matteflow.input.decoder import ImageDecoder
from matteflow.matte.green_screen_matte import GreenScreenMatte
from matteflow.matte.hybrid_matte import HybridMatte
from matteflow.pipeline import MattingPipeline
from matteflow.refine.color_decontaminate import ColorDecontaminate
from matteflow.refine.despeckle import Despeckle
from matteflow.refine.edge_refiner import EdgeRefiner
from matteflow.refine.effect_prop_repair import EffectPropRepair


def test_frame_2_green_screen_background_residue_is_cleared():
    frames, _ = ImageDecoder().decode(PROJECT_ROOT / "assets" / "frame" / "test_frame_2.png")

    alpha = GreenScreenMatte(MattingConfig()).generate(frames[0])

    assert float((alpha > 0.01).mean()) < 0.80
    assert float(alpha.min()) == 0.0


def test_unknown_background_without_ai_uses_traditional_fallback_for_frame_3():
    frames, _ = ImageDecoder().decode(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg")
    matte = HybridMatte(MattingConfig(use_ai=False))

    alphas = matte.generate_sequence(frames, BackgroundMode.UNKNOWN)

    assert len(alphas) == 1
    assert matte.last_active_ai_model == "traditional_green_fallback"
    assert float((alphas[0] > 0.01).mean()) < 0.90


def test_green_screen_keeps_warm_luminous_star_core():
    frame = np.full((40, 40, 3), [0, 210, 40], dtype="uint8")
    frame[14:26, 14:26] = [170, 220, 80]
    frame[17:23, 17:23] = [245, 205, 95]

    alpha = GreenScreenMatte(MattingConfig()).generate(frame)

    assert float(alpha[14:26, 14:26].mean()) >= 0.95
    assert float(alpha[17:23, 17:23].min()) >= 0.95
    assert float(alpha[:8, :8].mean()) <= 0.05


def test_frame_3_yellow_star_solid_spill_is_decontaminated():
    frames, _ = ImageDecoder().decode(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg")
    frame = frames[0]
    alpha = GreenScreenMatte(MattingConfig()).generate(frame)

    processed = ColorDecontaminate(MattingConfig()).process([frame], [alpha], BackgroundMode.GREEN_SCREEN)[0]

    r, g, b = frame[:, :, 0].astype("float32"), frame[:, :, 1].astype("float32"), frame[:, :, 2].astype("float32")
    warm_luminous = (r > 170) & (g > 115) & (b < 175) & ((r - b) > 45) & ((g - b) > 8)
    import cv2

    warm_reach = cv2.dilate(
        warm_luminous.astype("uint8"),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ).astype(bool)
    spill = warm_reach & (alpha > 0.20) & (g > r + 12) & (g > b + 6) & (r < 170) & (g > 95)

    processed_f = processed.astype("float32")
    assert int(spill.sum()) >= 100
    assert float((processed_f[:, :, 1] - processed_f[:, :, 0])[spill].mean()) < float((g - r)[spill].mean()) - 10.0


def test_frame_3_unknown_green_fallback_runs_green_decontamination(tmp_path):
    source_path = PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"
    result = MattingPipeline(
        MattingConfig(background_mode=BackgroundMode.AUTO, use_ai=False, output_mask=True)
    ).process(source_path, tmp_path)

    import cv2

    frame = cv2.cvtColor(cv2.imread(str(source_path)), cv2.COLOR_BGR2RGB).astype("float32")
    processed = cv2.cvtColor(
        cv2.imread(str(tmp_path / "Processed" / "processed_000000.png")),
        cv2.COLOR_BGR2RGB,
    ).astype("float32")
    alpha = cv2.imread(str(tmp_path / "mask" / "mask_000000.png"), cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0

    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    warm_luminous = (r > 170) & (g > 115) & (b < 175) & ((r - b) > 45) & ((g - b) > 8)
    warm_reach = cv2.dilate(
        warm_luminous.astype("uint8"),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
        iterations=1,
    ).astype(bool)
    spill = warm_reach & (alpha > 0.20) & (g > r + 12) & (g > b + 6) & (r < 170) & (g > 95)

    assert result["status"] == "success"
    assert int(spill.sum()) >= 100
    assert float((processed[:, :, 1] - processed[:, :, 0])[spill].mean()) < 5.0


def test_frame_3_gvm_fusion_keeps_warm_luminous_props_from_base_alpha():
    frames, _ = ImageDecoder().decode(PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg")
    frame = frames[0]
    base_alpha = GreenScreenMatte(MattingConfig()).generate(frame)
    ai_alpha = np.full_like(base_alpha, 0.50, dtype=np.float32)
    matte = HybridMatte(MattingConfig(use_ai=False, transparency_preserve=0.7))
    matte.last_active_ai_model = "gvm"

    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    r, g, b = frame[:, :, 0].astype("float32"), frame[:, :, 1].astype("float32"), frame[:, :, 2].astype("float32")
    warm_luminous = (r > 170) & (g > 115) & (b < 175) & ((r - b) > 45) & ((g - b) > 8)

    assert int(warm_luminous.sum()) >= 1_000
    assert float((merged[warm_luminous] < 0.95).mean()) <= 0.02

    refined = EdgeRefiner(MattingConfig()).refine([frame], [merged])[0]
    despeckled = Despeckle(MattingConfig()).process(
        [refined],
        frames=[frame],
        context={"active_ai_model": "gvm"},
    )[0]

    assert float((despeckled[warm_luminous] < 0.95).mean()) <= 0.02


def test_effect_prop_repair_restores_bright_core_inside_warm_luminous_prop():
    frame = np.full((80, 80, 3), [0, 220, 40], dtype=np.uint8)
    frame[24:56, 20:60] = [230, 170, 80]
    frame[34:46, 34:46] = [245, 235, 190]
    base_alpha = np.zeros((80, 80), dtype=np.float32)
    base_alpha[24:56, 20:60] = 1.0
    damaged_alpha = base_alpha.copy()
    damaged_alpha[34:46, 34:46] = 0.0

    repaired = EffectPropRepair(MattingConfig())._repair_single(frame, damaged_alpha, base_alpha)

    assert float(repaired[34:46, 34:46].mean()) >= 0.98


def test_effect_prop_repair_fills_internal_hole_from_prop_topology():
    frame = np.full((90, 90, 3), [0, 220, 40], dtype=np.uint8)
    cv2.circle(frame, (45, 45), 22, [235, 170, 75], thickness=8)
    cv2.circle(frame, (45, 45), 8, [20, 210, 45], thickness=-1)
    base_alpha = np.zeros((90, 90), dtype=np.float32)
    cv2.circle(base_alpha, (45, 45), 24, 1.0, thickness=-1)
    damaged_alpha = base_alpha.copy()
    cv2.circle(damaged_alpha, (45, 45), 8, 0.0, thickness=-1)

    repaired = EffectPropRepair(MattingConfig())._repair_single(frame, damaged_alpha, base_alpha)

    center = np.zeros((90, 90), dtype=np.uint8)
    cv2.circle(center, (45, 45), 8, 1, thickness=-1)
    center = center.astype(bool)
    assert float(repaired[center].mean()) >= 0.98


def test_green_screen_gvm_suppresses_large_document_background():
    frame = np.full((220, 260, 3), [0, 220, 40], dtype=np.uint8)
    frame[10:115, 20:240] = [236, 236, 232]
    frame[38:48, 45:190] = [35, 35, 35]
    frame[130:190, 80:170] = [210, 160, 235]
    base_alpha = np.zeros((220, 260), dtype=np.float32)
    base_alpha[10:115, 20:240] = 1.0
    base_alpha[130:190, 80:170] = 1.0
    ai_alpha = base_alpha.copy()

    matte = HybridMatte(MattingConfig(use_ai=False, transparency_preserve=0.7))
    matte.last_active_ai_model = "gvm"
    merged = matte._merge_green_screen_effects([base_alpha], [ai_alpha], [frame])[0]

    assert float(merged[20:105, 30:230].mean()) <= 0.05
    assert float(merged[140:180, 95:155].mean()) >= 0.85


def test_frame_3_green_gvm_pipeline_restores_warm_prop_cores(tmp_path):
    source_path = PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"
    result = MattingPipeline(
        MattingConfig(
            background_mode=BackgroundMode.GREEN_SCREEN,
            use_ai=True,
            ai_model="gvm",
            ai_enhance=False,
            output_matte=True,
            output_processed=True,
        )
    ).process(source_path, tmp_path)

    import cv2

    frame = cv2.cvtColor(cv2.imread(str(source_path)), cv2.COLOR_BGR2RGB).astype("float32")
    alpha = cv2.imread(str(tmp_path / "Matte" / "matte_000000.png"), cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    warm = (r > 170) & (g > 115) & (b < 175) & ((r - b) > 45) & ((g - b) > 8)
    bright_core = (
        (r > 190)
        & (g > 145)
        & (b > 70)
        & (((r + g + b) / 3.0) > 165)
    )
    warm_reach = cv2.dilate(
        warm.astype("uint8"),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        iterations=1,
    ).astype(bool)
    prop_core = warm_reach & bright_core
    processed = cv2.cvtColor(
        cv2.imread(str(tmp_path / "Processed" / "processed_000000.png"), cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGRA2RGBA,
    ).astype("float32")
    processed_rgb = processed[:, :, :3]
    green_edge = (
        (processed_rgb[:, :, 1] > processed_rgb[:, :, 0] + 10)
        & (processed_rgb[:, :, 1] > processed_rgb[:, :, 2] + 5)
        & (alpha > 0.10)
    )

    assert result["status"] == "success"
    assert int(prop_core.sum()) >= 500
    assert float((alpha[prop_core] < 0.95).mean()) <= 0.02
    assert int((green_edge & warm_reach).sum()) <= 30


def test_frame_3_green_gvm_restores_line_connected_luminous_comets(tmp_path):
    source_path = PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"
    result = MattingPipeline(
        MattingConfig(
            background_mode=BackgroundMode.GREEN_SCREEN,
            use_ai=True,
            ai_model="gvm",
            ai_enhance=False,
            output_matte=True,
            output_processed=True,
        )
    ).process(source_path, tmp_path)

    frame = cv2.cvtColor(cv2.imread(str(source_path)), cv2.COLOR_BGR2RGB).astype("float32")
    alpha = cv2.imread(str(tmp_path / "Matte" / "matte_000000.png"), cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
    x, y, w, h = 1309, 623, 149, 107
    roi = frame[y : y + h, x : x + w]
    roi_alpha = alpha[y : y + h, x : x + w]
    r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    luminous_core = (r > 185) & (g > 130) & (b > 50) & (((r + g + b) / 3.0) > 130)

    assert result["status"] == "success"
    assert int(luminous_core.sum()) >= 400
    assert float((roi_alpha[luminous_core] < 0.80).mean()) <= 0.05


def test_frame_3_green_gvm_removes_solid_green_spill_on_warm_star_edges(tmp_path):
    source_path = PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"
    result = MattingPipeline(
        MattingConfig(
            background_mode=BackgroundMode.GREEN_SCREEN,
            use_ai=True,
            ai_model="gvm",
            ai_enhance=False,
            output_matte=True,
            output_processed=True,
        )
    ).process(source_path, tmp_path)

    processed = cv2.cvtColor(
        cv2.imread(str(tmp_path / "Processed" / "processed_000000.png"), cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGRA2RGBA,
    ).astype("float32")
    alpha = processed[:, :, 3] / 255.0
    rgb = processed[:, :, :3]
    x, y, w, h = 498, 33, 280, 240
    roi_rgb = rgb[y : y + h, x : x + w]
    roi_alpha = alpha[y : y + h, x : x + w]
    green_edge = (
        (roi_rgb[:, :, 1] > roi_rgb[:, :, 0] + 8)
        & (roi_rgb[:, :, 1] > roi_rgb[:, :, 2] + 4)
        & (roi_alpha > 0.10)
    )

    assert result["status"] == "success"
    assert int(green_edge.sum()) <= 40


def test_frame_3_green_gvm_removes_subtle_yellow_green_cast_on_star_edges(tmp_path):
    source_path = PROJECT_ROOT / "assets" / "frame" / "test_frame_3.jpg"
    result = MattingPipeline(
        MattingConfig(
            background_mode=BackgroundMode.GREEN_SCREEN,
            use_ai=True,
            ai_model="gvm",
            ai_enhance=False,
            output_processed=True,
        )
    ).process(source_path, tmp_path)

    processed = cv2.cvtColor(
        cv2.imread(str(tmp_path / "Processed" / "processed_000000.png"), cv2.IMREAD_UNCHANGED),
        cv2.COLOR_BGRA2RGBA,
    ).astype("float32")
    alpha = processed[:, :, 3] / 255.0
    rgb = processed[:, :, :3]
    x, y, w, h = 498, 33, 280, 240
    roi_rgb = rgb[y : y + h, x : x + w]
    roi_alpha = alpha[y : y + h, x : x + w]
    subtle_yellow_green_edge = (
        (roi_rgb[:, :, 1] > roi_rgb[:, :, 0] + 6)
        & (roi_rgb[:, :, 1] > roi_rgb[:, :, 2] + 3)
        & (roi_alpha > 0.08)
    )

    assert result["status"] == "success"
    assert int(subtle_yellow_green_edge.sum()) <= 40
