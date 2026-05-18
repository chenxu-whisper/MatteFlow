"""
用法:
    python scripts/web_gui.py
    python scripts/web_gui.py --port 7860
    python scripts/web_gui.py --port 7862 --debug
    python scripts/web_gui.py --share
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
import logging
import tempfile
import zipfile
import cv2
import numpy as np
import gradio as gr
from PIL import Image

from matteflow import MattingPipeline, MattingConfig, QualityMode, BackgroundMode
from matteflow.auto_params import apply_suggestion, suggest_input_params
from matteflow.input.formats import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from matteflow.utils.cv_compat import video_writer_fourcc
from matteflow.utils.model_checker import ModelChecker
from matteflow.utils.output_paths import (
    resolve_project_output_dir,
    sanitize_output_name,
)

logger = logging.getLogger(__name__)
SUPPORTED_UPLOAD_EXTENSIONS = sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)
GUI_DEFAULTS = {
    "mode": "green",
    "quality": "standard",
    "preferred_ai": "gvm",
    "pure_color": True,
    "use_filter": False,
    "green_similarity": 0.4,
    "green_despill": 0.7,
    "green_hair": 0.8,
    "white_protect_brightness": 180,
    "white_protect_saturation": 25,
    "edge_despill_factor": 1.2,
    "screen_color": "auto",
    "key_strength": 1.0,
    "clip_black": 0.0,
    "clip_white": 1.0,
    "shrink_grow": 0,
    "edge_blur": 0,
    "despill_enable": True,
    "despill_strength": 0.7,
    "despill_color": "green",
    "despeckle_enable": True,
    "despeckle_radius": 2,
    "despeckle_threshold": 0.0,
    "transparency_preserve": 0.7,
    "gvm_max_internal_size": 768,
    "auto_optimize": False,
    "generate_zip": False,
    "output_fg": False,
    "output_matte": True,
    "output_comp": False,
    "output_processed": True,
}

GUI_PRIMARY_CONTROL_KEYS = [
    "use_ai",
    "quality",
    "key_strength",
    "transparency_preserve",
    "green_despill",
    "edge_despill_factor",
    "shrink_grow",
    "edge_blur",
    "gvm_max_internal_size",
]

GUI_FIXED_PARAMETER_DEFAULTS = {
    "pure_color": GUI_DEFAULTS["pure_color"],
    "use_filter": GUI_DEFAULTS["use_filter"],
    "edge_softness": 0.0,
    "temporal_strength": 0.5,
    "color_space": "sRGB",
    "despill_enable": GUI_DEFAULTS["despill_enable"],
    "despill_color": GUI_DEFAULTS["despill_color"],
}

RECOMMENDED_PRESET_OUTPUT_KEYS = [
    "mode",
    "quality",
    "use_ai",
    "pure_color_mode",
    "use_guided_filter",
    "green_similarity",
    "green_despill",
    "green_hair",
    "white_protect_thresh",
    "white_protect_sat",
    "edge_despill_factor",
    "screen_color",
    "key_strength",
    "clip_black",
    "clip_white",
    "shrink_grow",
    "edge_blur",
    "despill_enable",
    "despill_strength",
    "despill_color",
    "despeckle_enable",
    "despeckle_radius",
    "despeckle_threshold",
    "transparency_preserve",
    "gvm_max_internal_size",
    "auto_optimize",
    "generate_zip",
    "output_fg",
    "output_matte",
    "output_comp",
    "output_processed",
]

# 全局状态
_output_dir = None
_current_preview_index = 0

# 检查可用模型
_model_checker = ModelChecker()
_available_models = _model_checker.get_available_models()
_ui_choices = _model_checker.get_ui_choices()


def _configure_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _sanitize_output_name(name: str) -> str:
    return sanitize_output_name(name)


def _resolve_gui_output_dir(video_path, output_root: Path | None = None) -> Path:
    return resolve_project_output_dir(
        Path(video_path),
        project_root=project_root,
        output_root=output_root,
    )


def _create_upload_preview(file_path):
    hide_image = gr.update(value=None, visible=False)
    hide_video = gr.update(value=None, visible=False)
    if not file_path:
        return hide_image, hide_video, "未选择素材"

    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        if not path.exists():
            return hide_image, hide_video, f"图片文件不存在: {path.name}"
        return gr.update(value=str(path), visible=True), hide_video, f"已选择图片: {path.name}"

    if suffix in VIDEO_EXTENSIONS:
        if not path.exists():
            return hide_image, hide_video, f"视频文件不存在: {path.name}"
        return hide_image, gr.update(value=str(path), visible=True), f"已选择视频: {path.name}"

    return hide_image, hide_video, f"不支持的素材格式: {path.suffix}"


def _default_ai_choice() -> str:
    preferred = GUI_DEFAULTS["preferred_ai"]
    if any(value == preferred for _, value in _ui_choices):
        return preferred
    return _ui_choices[0][1] if _ui_choices else "traditional"


def _apply_recommended_preset() -> dict:
    return {
        "mode": GUI_DEFAULTS["mode"],
        "quality": GUI_DEFAULTS["quality"],
        "use_ai": _default_ai_choice(),
        "pure_color_mode": GUI_DEFAULTS["pure_color"],
        "use_guided_filter": GUI_DEFAULTS["use_filter"],
        "green_similarity": GUI_DEFAULTS["green_similarity"],
        "green_despill": GUI_DEFAULTS["green_despill"],
        "green_hair": GUI_DEFAULTS["green_hair"],
        "white_protect_thresh": GUI_DEFAULTS["white_protect_brightness"],
        "white_protect_sat": GUI_DEFAULTS["white_protect_saturation"],
        "edge_despill_factor": GUI_DEFAULTS["edge_despill_factor"],
        "screen_color": GUI_DEFAULTS["screen_color"],
        "key_strength": GUI_DEFAULTS["key_strength"],
        "clip_black": GUI_DEFAULTS["clip_black"],
        "clip_white": GUI_DEFAULTS["clip_white"],
        "shrink_grow": GUI_DEFAULTS["shrink_grow"],
        "edge_blur": GUI_DEFAULTS["edge_blur"],
        "despill_enable": GUI_DEFAULTS["despill_enable"],
        "despill_strength": GUI_DEFAULTS["despill_strength"],
        "despill_color": GUI_DEFAULTS["despill_color"],
        "despeckle_enable": GUI_DEFAULTS["despeckle_enable"],
        "despeckle_radius": GUI_DEFAULTS["despeckle_radius"],
        "despeckle_threshold": GUI_DEFAULTS["despeckle_threshold"],
        "transparency_preserve": GUI_DEFAULTS["transparency_preserve"],
        "gvm_max_internal_size": GUI_DEFAULTS["gvm_max_internal_size"],
        "auto_optimize": GUI_DEFAULTS["auto_optimize"],
        "generate_zip": GUI_DEFAULTS["generate_zip"],
        "output_fg": GUI_DEFAULTS["output_fg"],
        "output_matte": GUI_DEFAULTS["output_matte"],
        "output_comp": GUI_DEFAULTS["output_comp"],
        "output_processed": GUI_DEFAULTS["output_processed"],
    }


def _recommended_preset_updates():
    preset = _apply_recommended_preset()
    return tuple(gr.update(value=preset[key]) for key in RECOMMENDED_PRESET_OUTPUT_KEYS)


def process_video(
    video_path,
    mode,
    quality,
    use_ai,
    pure_color_mode,
    use_guided_filter,
    green_similarity,
    green_despill,
    green_hair,
    white_protect_thresh,
    white_protect_sat,
    edge_despill_factor,
    black_threshold,
    black_glow,
    black_particle,
    edge_softness,
    temporal_strength,
    transparency_preserve,
    gvm_max_internal_size,
    auto_optimize,
    # Chroma Key 参数
    screen_color,
    key_strength,
    clip_black,
    clip_white,
    shrink_grow,
    edge_blur,
    despill_enable,
    despill_strength,
    despill_color,
    despeckle_enable,
    despeckle_radius,
    despeckle_threshold,
    color_space,
    output_fg,
    output_matte,
    output_comp,
    output_processed,
    generate_zip,
    ai_gamma,
    ai_threshold,
    ai_gain,
    ai_sharpen,
    progress=gr.Progress()
):
    """处理视频 - 带实时预览"""
    global _output_dir

    if video_path is None:
        return None, None, "请先上传视频", None, None, 0, None

    # 构建配置
    config = MattingConfig()

    if mode == "green":
        config.background_mode = BackgroundMode.GREEN_SCREEN
    elif mode == "black":
        config.background_mode = BackgroundMode.BLACK_BACKGROUND
    else:
        config.background_mode = BackgroundMode.AUTO

    if quality == "fast":
        config.quality_mode = QualityMode.FAST
    elif quality == "high":
        config.quality_mode = QualityMode.HIGH
    else:
        config.quality_mode = QualityMode.STANDARD

    # 解析抠图引擎选项
    if use_ai == "enhance":
        config.use_ai = True
        config.ai_enhance = True
        config.ai_model = "auto"
    elif use_ai == "gvm":
        config.use_ai = True
        config.ai_enhance = False
        config.ai_model = "gvm"
    elif use_ai == "matanyone2":
        config.use_ai = True
        config.ai_enhance = False
        config.ai_model = "matanyone2"
    elif use_ai in ("ai", "traditional"):
        config.use_ai = False
        config.ai_enhance = False
        config.ai_model = "auto"
    else:
        config.use_ai = False
        config.ai_enhance = False
        config.ai_model = "auto"

    config.pure_color_mode = pure_color_mode
    config.use_guided_filter = use_guided_filter

    config.green_similarity = green_similarity
    config.green_despill_strength = green_despill
    config.green_hair_detail = green_hair
    config.white_protect_brightness = white_protect_thresh
    config.white_protect_saturation = white_protect_sat
    config.edge_despill_factor = edge_despill_factor
    config.black_threshold = black_threshold
    config.black_glow_preserve = black_glow
    config.black_particle_boost = black_particle
    config.edge_softness = edge_softness
    config.temporal_strength = temporal_strength
    config.transparency_preserve = transparency_preserve
    config.gvm_max_internal_size = gvm_max_internal_size
    config.ai_enhance_gamma = ai_gamma
    config.ai_enhance_threshold = ai_threshold
    config.ai_enhance_gain = ai_gain
    config.ai_enhance_sharpen = ai_sharpen

    # Chroma Key 参数
    config.screen_color = screen_color
    config.key_strength = key_strength
    config.clip_black = clip_black
    config.clip_white = clip_white
    config.shrink_grow = shrink_grow
    config.edge_blur = edge_blur


    config.despill_enable = despill_enable
    config.despill_strength = despill_strength
    config.despill_color = despill_color
    config.despeckle_enable = despeckle_enable
    config.despeckle_radius = despeckle_radius
    config.despeckle_threshold = despeckle_threshold
    config.color_space = color_space
    config.output_fg = output_fg
    config.output_matte = output_matte
    config.output_comp = output_comp
    config.output_processed = output_processed
    config.generate_zip_by_default = generate_zip

    auto_summary = ""
    if auto_optimize:
        suggestion = suggest_input_params(video_path, config)
        apply_suggestion(config, suggestion)
        auto_summary = f"\n{suggestion.summary}{_format_actual_parameter_summary(config)}"
        logger.info("Applied auto optimization for %s: %s", video_path, suggestion)

    # 处理
    _output_dir = _resolve_gui_output_dir(video_path)
    logger.info(
        "Starting GUI processing: video=%s output_dir=%s mode=%s quality=%s ai_model=%s",
        video_path,
        _output_dir,
        mode,
        quality,
        use_ai,
    )
    
    logger.info(
        "GUI key config: screen_color=%s key_strength=%s clip_black=%s clip_white=%s "
        "shrink_grow=%s edge_blur=%s output_fg=%s output_matte=%s output_comp=%s output_processed=%s",
        config.screen_color,
        config.key_strength,
        config.clip_black,
        config.clip_white,
        config.shrink_grow,
        config.edge_blur,
        config.output_fg,
        config.output_matte,
        config.output_comp,
        config.output_processed,
    )
    
    try:
        pipeline = MattingPipeline(config)

        # 用于实时预览的变量
        preview_input = None
        preview_output = None

        def on_progress(current, total, stage):
            progress(current / total, desc=stage)
            # 可以在这里更新预览,但 Gradio 的进度回调不支持 yield

        result = pipeline.process(video_path, _output_dir, on_progress)

        # 打包序列帧为 ZIP
        zip_path = _output_dir / "frames.zip" if generate_zip else None
        if zip_path is not None:
            _create_zip(_output_dir, zip_path)

        # 生成预览视频
        preview_path = _output_dir / "preview.mp4"
        _create_preview_video(_output_dir, preview_path)

        # 生成首帧预览图
        input_preview, output_preview = _create_preview_frames(_output_dir)
        transparent_png = _find_transparent_png_download(_output_dir)

        status = (
            f"✅ 完成!{result['frames_processed']}帧 | {result['fps']:.1f} fps | "
            f"耗时 {result['elapsed_time']:.1f}s{auto_summary}"
        )

        # 返回预览视频和首帧对比图
        # Gradio Video 组件需要字符串路径,且视频必须可被浏览器播放
        preview_video = None
        if preview_path.exists() and preview_path.stat().st_size > 0:
            # Gradio 需要将文件放在其工作目录下才能正确服务
            # 复制到 Gradio 的临时目录
            import shutil
            gradio_temp = Path(tempfile.gettempdir()) / "gradio" / "matteflow_previews"
            gradio_temp.mkdir(parents=True, exist_ok=True)

            # 使用唯一文件名避免冲突
            preview_name = f"preview_{_output_dir.name}.mp4"
            gradio_preview_path = gradio_temp / preview_name

            try:
                shutil.copy2(str(preview_path), str(gradio_preview_path))
                preview_video = str(gradio_preview_path.resolve())
                logger.info("Copied preview video to Gradio temp path: %s", preview_video)
            except Exception as e:
                # 如果复制失败,直接使用原路径
                preview_video = str(preview_path.resolve())
                logger.warning("Failed to copy preview to Gradio temp path, using original preview: %s", preview_video)
        else:
            logger.warning("Preview not found or empty: %s", preview_path)

        logger.info(
            "GUI processing completed: frames=%s elapsed=%.1fs fps=%.2f zip=%s png=%s preview=%s",
            result["frames_processed"],
            result["elapsed_time"],
            result["fps"],
            zip_path,
            transparent_png,
            preview_video,
        )

        return (
            preview_video,
            str(zip_path) if zip_path else None,
            status,
            input_preview,
            output_preview,
            result['frames_processed'],
            str(transparent_png) if transparent_png else None,
        )
    except Exception as e:
        logger.exception("GUI processing failed for video=%s", video_path)
        return None, None, f"❌ 错误: {str(e)}", None, None, 0, None


def _format_actual_parameter_summary(config):
    """Return the effective runtime parameters after optional auto optimization."""
    return (
        "\n本次实际参数: "
        f"screen={config.screen_color}, "
        f"similarity={config.green_similarity:.2f}, "
        f"key={config.key_strength:.2f}, "
        f"preserve={config.transparency_preserve:.2f}, "
        f"despill={config.green_despill_strength:.2f}, "
        f"edge_despill={config.edge_despill_factor:.2f}, "
        f"clip={config.clip_black:.2f}/{config.clip_white:.2f}, "
        f"white_protect={config.white_protect_brightness:.0f}/{config.white_protect_saturation:.0f}, "
        f"shrink_grow={config.shrink_grow}, "
        f"edge_blur={config.edge_blur}, "
        f"gvm_size={config.gvm_max_internal_size}"
    )


def _create_preview_frames(output_dir):
    """创建首帧预览对比图 - 支持子目录结构
    
    优先使用 Comp (预乘合成，背景黑色) 或 Processed (RGBA)
    避免使用 FG (未预乘，背景仍是绿色)
    """
    # 查找输出帧 - 优先顺序：Processed > Comp > FG > Matte
    frames = []
    for subdir in ["Processed", "Comp", "FG", "Matte"]:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            frames = sorted(subdir_path.glob("*.png"))
            if frames:
                logger.info("Using %s frame set for preview image: %s", subdir, frames[0].name)
                break

    # 如果没有子目录,检查根目录
    if not frames:
        frames = sorted(output_dir.glob("frame_*.png"))

    if not frames:
        logger.warning("No preview frames found in %s", output_dir)
        return None, None

    # 取首帧
    img = np.array(Image.open(frames[0]))
    h, w = img.shape[:2]

    # 创建棋盘格背景
    grid_size = 20
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            color = 180 if ((y // grid_size) + (x // grid_size)) % 2 == 0 else 120
            bg[y:y+grid_size, x:x+grid_size] = color

    # 判断使用的是哪个目录
    used_subdir = None
    for subdir in ["Processed", "Comp", "FG", "Matte"]:
        subdir_path = output_dir / subdir
        if subdir_path.exists() and any(subdir_path.glob("*.png")):
            used_subdir = subdir
            break

    # 合成输出预览
    if used_subdir == "Comp":
        # Comp 是预乘合成，直接显示在棋盘格上（背景已经是黑色）
        rgb = img[:, :, :3].astype(np.float32)
        # 简单混合：如果像素接近黑色，显示棋盘格
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
        mask = (gray > 10).astype(np.float32)[:, :, np.newaxis]  # 非黑区域
        output_preview = (rgb * mask + bg * (1 - mask)).astype(np.uint8)
    elif used_subdir == "Processed" and img.shape[2] == 4:
        # Processed 是 RGBA，使用 alpha 混合
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        output_preview = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
    elif used_subdir == "FG":
        # FG 是未预乘的，背景仍是绿色，需要特殊处理
        rgb = img[:, :, :3].astype(np.float32)
        # 简单去绿：检测绿色背景并替换为棋盘格
        green_mask = (img[:, :, 1] > img[:, :, 0] * 1.2) & (img[:, :, 1] > img[:, :, 2] * 1.2)
        mask = (~green_mask).astype(np.float32)[:, :, np.newaxis]
        output_preview = (rgb * mask + bg * (1 - mask)).astype(np.uint8)
    else:
        # 默认：直接显示
        output_preview = img[:, :, :3].astype(np.uint8)

    # 输入预览(RGB 原图,不带 alpha)
    input_preview = img[:, :, :3].astype(np.uint8)

    return input_preview, output_preview


def _find_transparent_png_download(output_dir):
    """Return the first RGBA processed PNG for direct single-frame download."""
    processed_dir = Path(output_dir) / "Processed"
    if not processed_dir.exists():
        logger.info("Processed output directory does not exist for PNG download: %s", processed_dir)
        return None

    frames = sorted(processed_dir.glob("*.png"))
    if not frames:
        logger.info("No processed PNG frames available for download in %s", processed_dir)
        return None

    return frames[0]


def _create_preview_video(output_dir, preview_path):
    """创建带棋盘格背景的预览视频 - 使用 imageio 确保浏览器兼容
    
    优先使用 Comp (预乘合成，背景黑色) 或 Processed (RGBA)
    避免使用 FG (未预乘，背景仍是绿色)
    """

    # 查找输出帧 - 优先顺序：Processed > Comp > FG > Matte
    frames = []
    for subdir in ["Processed", "Comp", "FG", "Matte"]:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            frames = sorted(subdir_path.glob("*.png"))
            if frames:
                logger.info("Using %s frame set for preview video", subdir)
                break

    # 如果没有子目录,检查根目录
    if not frames:
        frames = sorted(output_dir.glob("frame_*.png"))

    if not frames:
        logger.warning("No frames found for preview video in %s", output_dir)
        return

    try:
        import imageio
    except ImportError:
        logger.warning("imageio not available, falling back to cv2 preview writer")
        _create_preview_video_cv2(output_dir, preview_path)
        return

    first = np.array(Image.open(frames[0]))
    h, w = first.shape[:2]

    # 创建棋盘格背景
    grid_size = 20
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            color = 180 if ((y // grid_size) + (x // grid_size)) % 2 == 0 else 120
            bg[y:y+grid_size, x:x+grid_size] = color

    # 使用 imageio 写入 MP4 (H.264)
    preview_path = preview_path.with_suffix('.mp4')

    try:
        # 尝试使用 imageio-ffmpeg 插件,使用 H.264 编码确保浏览器兼容
        writer = imageio.get_writer(
            str(preview_path),
            fps=30,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p',  # 确保浏览器兼容
            macro_block_size=1      # 避免自动调整尺寸
        )
    except Exception as e:
        logger.warning("imageio H.264 writer failed, trying default writer: %s", e)
        try:
            writer = imageio.get_writer(str(preview_path), fps=30)
        except Exception as e2:
            logger.warning("imageio default writer failed, falling back to cv2: %s", e2)
            _create_preview_video_cv2(output_dir, preview_path)
            return

    frame_count = 0
    for frame_path in frames:
        rgba = np.array(Image.open(frame_path))

        # 处理 RGBA 或 RGB
        if rgba.shape[2] == 4:
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            rgb = rgba[:, :, :3]
        else:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            rgb = rgba

        # 合成到棋盘格背景
        composed = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        writer.append_data(composed)
        frame_count += 1

    writer.close()
    logger.info("Created preview video: %s (%s frames)", preview_path, frame_count)


def _create_preview_video_cv2(output_dir, preview_path):
    """备用:使用 OpenCV 创建预览视频"""

    # 查找输出帧
    frames = []
    for subdir in ["Processed", "Comp", "FG", "Matte"]:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            frames = sorted(subdir_path.glob("*.png"))
            if frames:
                break

    if not frames:
        frames = sorted(output_dir.glob("frame_*.png"))

    if not frames:
        return

    first = np.array(Image.open(frames[0]))
    h, w = first.shape[:2]

    # 使用 mp4v 编码
    preview_path = preview_path.with_suffix('.mp4')
    fourcc = video_writer_fourcc("mp4v", cv2)
    writer = cv2.VideoWriter(str(preview_path), fourcc, 30.0, (w, h))

    if not writer.isOpened():
        logger.warning("CV2 preview writer failed to open for %s", preview_path)
        return

    # 创建棋盘格背景
    grid_size = 20
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            color = 180 if ((y // grid_size) + (x // grid_size)) % 2 == 0 else 120
            bg[y:y+grid_size, x:x+grid_size] = color

    for frame_path in frames:
        rgba = np.array(Image.open(frame_path))
        if rgba.shape[2] == 4:
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            rgb = rgba[:, :, :3]
        else:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            rgb = rgba

        composed = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
        composed_bgr = cv2.cvtColor(composed, cv2.COLOR_RGB2BGR)
        writer.write(composed_bgr)

    writer.release()
    logger.info("Created preview video with cv2 fallback: %s", preview_path)


def _create_zip(output_dir, zip_path):
    """打包 PNG 序列帧为 ZIP"""
    # 检查所有可能的输出目录
    all_frames = []
    for subdir in ["FG", "Matte", "Comp", "Processed"]:
        subdir_path = output_dir / subdir
        if subdir_path.exists():
            frames = sorted(subdir_path.glob("*.png"))
            all_frames.extend(frames)

    # 如果没有子目录,检查根目录
    if not all_frames:
        all_frames = sorted(output_dir.glob("frame_*.png"))

    if not all_frames:
        logger.warning("No frames found to pack into zip under %s", output_dir)
        # 创建一个空 ZIP
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            pass
        return

    logger.info("Packing %s frames into zip: %s", len(all_frames), zip_path)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for frame_path in all_frames:
            # 使用相对路径保持目录结构
            arcname = frame_path.relative_to(output_dir)
            zf.write(frame_path, arcname)


# CSS 样式
custom_css = """
    .tab-button { font-size: 14px; }
    .param-group { border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; margin: 5px 0; }
    .preview-box { background: #f5f5f5; border-radius: 8px; padding: 10px; }
"""

def create_ui():
    """创建 Gradio UI"""

    with gr.Blocks(title="MatteFlow - 专业视频/序列帧/图片抠图") as app:

        # 顶部标题栏
        with gr.Row():
            gr.Markdown("""
            # 🎬 MatteFlow
            **CorridorKey 物理分离 | BiRefNet AI | 传统色度键**
            """)

        # 主工作区:左右布局
        with gr.Row():

            # 左侧:输入与参数面板
            with gr.Column(scale=1, min_width=350):

                # 文件导入区
                with gr.Group():
                    gr.Markdown("### 📁 导入")
                    video_input = gr.File(
                        label="拖放视频或图片文件",
                        file_types=SUPPORTED_UPLOAD_EXTENSIONS,
                        type="filepath",
                        height=120,
                    )
                    upload_image_preview = gr.Image(
                        label="素材预览",
                        interactive=False,
                        visible=False,
                        height=220,
                    )
                    upload_video_preview = gr.Video(
                        label="素材预览",
                        interactive=False,
                        visible=False,
                        height=220,
                    )
                    upload_status = gr.Markdown("未选择素材")

                # 引擎选择
                with gr.Group():
                    gr.Markdown("### ⚙️ Alpha 生成器")

                    mode_select = gr.Radio(
                        choices=[("🟢 绿幕", "green"), ("⚫ 黑底", "black"), ("🔍 自动识别", "auto")],
                        value=GUI_DEFAULTS["mode"],
                        label="背景模式"
                    )

                    # 动态生成模型选项
                    ai_select = gr.Radio(
                        choices=_ui_choices if _ui_choices else [("📐 传统算法", "traditional")],
                        value=_default_ai_choice(),
                        label="Alpha 生成器 (✅=可用, ❌=未安装)"
                    )

                    quality_select = gr.Radio(
                        choices=[("⚡ 快速", "fast"), ("✨ 标准", "standard"), ("🎨 高质量", "high")],
                        value=GUI_DEFAULTS["quality"],
                        label="质量模式"
                    )


                # Chroma Key 参数
                with gr.Group(visible=True) as green_params:
                    gr.Markdown("### 🟢 推荐核心参数")

                    with gr.Group():
                        key_strength = gr.Slider(
                            0.6, 1.4, value=GUI_DEFAULTS["key_strength"], step=0.05,
                            label="Key Strength",
                            info="抠像强度。过高容易抠透明，过低容易留绿"
                        )
                        transparency_preserve = gr.Slider(
                            0.0, 1.0,
                            value=GUI_DEFAULTS["transparency_preserve"],
                            step=0.05,
                            label="半透明保留",
                            info="提高会保留爱心辉光/特效，过高可能保留背景雾边"
                        )
                        green_despill = gr.Slider(
                            0.0, 1.0,
                            value=GUI_DEFAULTS["green_despill"],
                            step=0.05,
                            label="去绿边强度"
                        )
                        edge_despill_factor = gr.Slider(
                            0.5, 2.0,
                            value=GUI_DEFAULTS["edge_despill_factor"],
                            step=0.1,
                            label="去绿系数"
                        )

                    with gr.Row():
                        shrink_grow = gr.Slider(
                            -5, 5, value=GUI_DEFAULTS["shrink_grow"], step=1,
                            label="Shrink/Grow",
                            info="负值收边，正值补边"
                        )
                        edge_blur = gr.Slider(
                            0, 5, value=GUI_DEFAULTS["edge_blur"], step=1,
                            label="Edge Blur",
                            info="边缘柔化半径"
                        )

                    gvm_max_internal_size = gr.Radio(
                        choices=[("512 快速", 512), ("768 推荐", 768), ("1024 高质量", 1024)],
                        value=GUI_DEFAULTS["gvm_max_internal_size"],
                        label="GVM 推理尺寸"
                    )
                    auto_optimize = gr.Checkbox(
                        value=GUI_DEFAULTS["auto_optimize"],
                        label="自动优化参数",
                        info="图片直接分析当前图；视频/序列帧默认抽中间帧，本次处理临时优化参数"
                    )

                    with gr.Accordion("高级绿幕参数", open=False):
                        screen_color = gr.Radio(
                            choices=[("🟢 绿色", "green"), ("🔵 蓝色", "blue"), ("🔍 自动检测", "auto")],
                            value=GUI_DEFAULTS["screen_color"],
                            label="屏幕颜色"
                        )
                        green_sim = gr.Slider(0.1, 1.0, value=GUI_DEFAULTS["green_similarity"], step=0.05, label="颜色相似度")
                        green_hair = gr.Slider(
                            0.0, 1.0,
                            value=GUI_DEFAULTS["green_hair"],
                            step=0.05,
                            label="毛发保护",
                            visible=False,
                        )
                        white_protect_thresh = gr.Slider(150, 255, value=GUI_DEFAULTS["white_protect_brightness"], step=5, label="白色保护亮度")
                        white_protect_sat = gr.Slider(10, 60, value=GUI_DEFAULTS["white_protect_saturation"], step=1, label="白色保护饱和度")
                        with gr.Row():
                            clip_black = gr.Slider(
                                0.0, 1.0, value=GUI_DEFAULTS["clip_black"], step=0.01,
                                label="Clip Black",
                                info="危险参数：提高会吃掉辉光/耳朵边缘"
                            )
                            clip_white = gr.Slider(
                                0.0, 1.0, value=GUI_DEFAULTS["clip_white"], step=0.01,
                                label="Clip White",
                                info="降低会拉实主体，也可能加重雾边"
                            )

                    pure_color = gr.Checkbox(value=GUI_FIXED_PARAMETER_DEFAULTS["pure_color"], visible=False)
                    use_filter = gr.Checkbox(value=GUI_FIXED_PARAMETER_DEFAULTS["use_filter"], visible=False)

                # 黑底参数
                with gr.Group(visible=False) as black_params:
                    gr.Markdown("### ⚫ 黑底参数")

                    black_thresh = gr.Slider(0.0, 0.2, value=0.03, step=0.01, label="黑场阈值")
                    black_glow = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="辉光保留")
                    black_particle = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="粒子增强")

                # 低频通用参数固定为推荐值，避免普通用户误调坏效果。
                edge_soft = gr.Slider(0.0, 1.0, value=GUI_FIXED_PARAMETER_DEFAULTS["edge_softness"], visible=False)
                temporal_str = gr.Slider(0.0, 1.0, value=GUI_FIXED_PARAMETER_DEFAULTS["temporal_strength"], visible=False)

                # 推理控制
                with gr.Group():
                    gr.Markdown("### 🎛️ 高级与输出")

                    despill_enable = gr.Checkbox(value=GUI_FIXED_PARAMETER_DEFAULTS["despill_enable"], visible=False)
                    despill_strength = gr.Slider(
                        0.0, 1.0,
                        value=GUI_DEFAULTS["despill_strength"],
                        step=0.05,
                        label="去溢色强度",
                        visible=False
                    )
                    despill_color = gr.Radio(
                        choices=[("绿色", "green"), ("蓝色", "blue"), ("自动", "auto")],
                        value=GUI_FIXED_PARAMETER_DEFAULTS["despill_color"],
                        visible=False
                    )

                    with gr.Accordion("去噪点", open=False):
                        despeckle_enable = gr.Checkbox(value=GUI_DEFAULTS["despeckle_enable"], label="启用去噪点")
                        despeckle_radius = gr.Slider(1, 5, value=GUI_DEFAULTS["despeckle_radius"], step=1, label="去噪点半径")
                        despeckle_threshold = gr.Slider(
                            0.0, 1.0,
                            value=GUI_DEFAULTS["despeckle_threshold"],
                            step=0.05,
                            label="去噪点阈值",
                            info="提高会删除低透明噪点，也可能削掉辉光/半透明细节"
                        )

                    color_space = gr.Radio(
                        choices=[("sRGB", "sRGB"), ("Rec.709", "Rec709"), ("Linear", "Linear"), ("ACES", "ACES")],
                        value=GUI_FIXED_PARAMETER_DEFAULTS["color_space"],
                        visible=False
                    )

                    with gr.Accordion("输出设置", open=False):
                        with gr.Row():
                            output_fg = gr.Checkbox(value=GUI_DEFAULTS["output_fg"], label="FG (直接前景)")
                            output_matte = gr.Checkbox(value=GUI_DEFAULTS["output_matte"], label="Matte (Alpha)")
                        with gr.Row():
                            output_comp = gr.Checkbox(value=GUI_DEFAULTS["output_comp"], label="Comp (预乘)")
                            output_processed = gr.Checkbox(value=GUI_DEFAULTS["output_processed"], label="Processed (RGBA)")
                        generate_zip = gr.Checkbox(value=GUI_DEFAULTS["generate_zip"], label="打包 ZIP 下载")

                # AI 增强参数
                with gr.Group(visible=False) as ai_params:
                    gr.Markdown("### 🤖 AI 参数")
                    ai_gamma = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Gamma")
                    ai_threshold = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="泄漏阈值")
                    ai_gain = gr.Slider(1.0, 3.0, value=1.2, step=0.1, label="增益")
                    ai_sharpen = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="锐化")

                # 处理按钮
                with gr.Row():
                    preset_btn = gr.Button("↩ 恢复推荐参数", variant="secondary")
                    process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")

                # 状态栏
                status_text = gr.Textbox(
                    label="状态",
                    value="就绪",
                    interactive=False,
                    lines=2
                )

            # 右侧:预览与输出面板
            with gr.Column(scale=2):

                # 视频预览
                with gr.Group():
                    gr.Markdown("### 🎬 结果预览")
                    result_preview = gr.Video(
                        label="棋盘格背景预览",
                        interactive=False,
                        height=400
                    )

                    # 首帧对比图
                    with gr.Row():
                        input_preview = gr.Image(
                            label="输入帧",
                            interactive=False,
                            height=200
                        )
                        output_preview = gr.Image(
                            label="输出帧",
                            interactive=False,
                            height=200
                        )

                # 输出下载
                with gr.Group():
                    gr.Markdown("### 💾 输出")
                    with gr.Row():
                        frames_zip = gr.File(
                            label="PNG 序列帧 (ZIP)",
                            interactive=False
                        )
                        transparent_png = gr.File(
                            label="单帧透明 PNG",
                            interactive=False
                        )

                        frame_count = gr.Number(
                            label="处理帧数",
                            value=0,
                            interactive=False
                        )

        # 模式切换显示/隐藏参数
        def toggle_mode(mode):
            return {
                green_params: gr.update(visible=(mode == "green" or mode == "auto")),
                black_params: gr.update(visible=(mode == "black"))
            }

        mode_select.change(
            fn=toggle_mode,
            inputs=[mode_select],
            outputs=[green_params, black_params]
        )

        # AI 参数显示/隐藏
        def toggle_ai_params(choice):
            return gr.update(visible=(choice == "enhance"))

        ai_select.change(
            fn=toggle_ai_params,
            inputs=[ai_select],
            outputs=[ai_params]
        )

        video_input.change(
            fn=_create_upload_preview,
            inputs=[video_input],
            outputs=[upload_image_preview, upload_video_preview, upload_status],
        )

        preset_outputs = [
            mode_select,
            quality_select,
            ai_select,
            pure_color,
            use_filter,
            green_sim,
            green_despill,
            green_hair,
            white_protect_thresh,
            white_protect_sat,
            edge_despill_factor,
            screen_color,
            key_strength,
            clip_black,
            clip_white,
            shrink_grow,
            edge_blur,
            despill_enable,
            despill_strength,
            despill_color,
            despeckle_enable,
            despeckle_radius,
            despeckle_threshold,
            transparency_preserve,
            gvm_max_internal_size,
            auto_optimize,
            generate_zip,
            output_fg,
            output_matte,
            output_comp,
            output_processed,
        ]

        app.load(
            fn=_recommended_preset_updates,
            outputs=preset_outputs,
        )

        preset_btn.click(
            fn=_recommended_preset_updates,
            outputs=preset_outputs,
        )

        # 绑定处理事件
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                mode_select,
                quality_select,
                ai_select,
                pure_color,
                use_filter,
                green_sim,
                green_despill,
                green_hair,
                white_protect_thresh,
                white_protect_sat,
                edge_despill_factor,
                black_thresh,
                black_glow,
                black_particle,
                edge_soft,
                temporal_str,
                transparency_preserve,
                gvm_max_internal_size,
                auto_optimize,
                # Chroma Key 参数
                screen_color,
                key_strength,
                clip_black,
                clip_white,
                shrink_grow,
                edge_blur,
                despill_enable,
                despill_strength,
                despill_color,
                despeckle_enable,
                despeckle_radius,
                despeckle_threshold,
                color_space,
                output_fg,
                output_matte,
                output_comp,
                output_processed,
                generate_zip,
                ai_gamma,
                ai_threshold,
                ai_gain,
                ai_sharpen,
            ],
            outputs=[
                result_preview,
                frames_zip,
                status_text,
                input_preview,
                output_preview,
                frame_count,
                transparent_png,
            ]
        )

        # 底部说明
        gr.Markdown("""
        ---
        ### 📖 快速指南

        **推荐流程:**
        1. 上传视频或图片(视频: MP4/MOV/AVI/MKV/WEBM; 图片: PNG/JPG/JPEG/WEBP/BMP/TIF/TIFF/EXR)
        2. 选择背景模式(绿幕/黑底/自动)
        3. 选择算法(CorridorKey 推荐用于绿幕)
        4. 调整参数(通常默认即可)
        5. 点击「开始处理」

        **算法选择:**
        - 🏆 **CorridorKey**:绿幕最佳,无需背景图,保留毛发/半透明
        - 🤖 **AI 增强**:传统 + AI 边缘细化,平衡速度质量
        - 📐 **传统算法**:快速,适合简单场景

        **快捷键:**
        - Ctrl+Enter:开始处理
        - R:重置参数
        """)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    args = parser.parse_args()
    _configure_logging(args.debug)
    logger.info("Available models: %s", _available_models)
    if args.debug:
        logger.info("Debug mode enabled")

    app = create_ui()
    # 禁用 SSR 模式,避免启动额外进程
    import os
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = str(args.port)
    logger.info("Launching Gradio UI on port=%s share=%s", args.port, args.share)

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        prevent_thread_lock=False,
        quiet=True,
        css=custom_css,
        ssr_mode=False
    )


if __name__ == "__main__":
    main()
