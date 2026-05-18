"""MatteFlow 主 Pipeline"""

import logging
import time
from pathlib import Path
from typing import Union, Optional
import numpy as np

from .config import MattingConfig, QualityMode, BackgroundMode
from .input.decoder import ImageDecoder, SequenceDecoder, VideoDecoder
from .input.formats import InputKind, detect_input_kind
from .analysis.background_analyzer import BackgroundAnalyzer
from .matte.matte_fusion import MatteFusion
from .refine.edge_refiner import EdgeRefiner
from .refine.color_decontaminate import ColorDecontaminate
from .refine.despeckle import Despeckle
from .temporal.temporal_stabilizer import TemporalStabilizer
from .output.encoder import RGBAEncoder

logger = logging.getLogger(__name__)


class MattingPipeline:
    """高质量抠图 Pipeline"""
    
    def __init__(self, config: Optional[MattingConfig] = None):
        self.config = config or MattingConfig()
        self.analyzer = BackgroundAnalyzer()
        self.fusion = MatteFusion()
        self.refiner = EdgeRefiner(self.config)
        self.decontaminate = ColorDecontaminate(self.config)
        self.despeckle = Despeckle(self.config)
        self.stabilizer = TemporalStabilizer(self.config)
        self.encoder = RGBAEncoder()
        
        # 初始化混合抠图引擎
        from .matte.hybrid_matte import HybridMatte
        self.hybrid_matte = HybridMatte(self.config)
        
        self._frames = []
        self._alphas = []
        self._processed = []

    def _log_alpha_stage_delta(self, stage: str, before, after) -> None:
        """Log how much a stage changed alpha so regressions are easy to localize."""
        if len(before) != len(after) or not before:
            return

        mean_abs_diffs = []
        changed_pixels = 0
        lifted_from_zero = 0
        suppressed_to_near_zero = 0

        for prev_alpha, next_alpha in zip(before, after):
            prev_f = np.clip(prev_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            next_f = np.clip(next_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            diff = np.abs(next_f - prev_f)
            mean_abs_diffs.append(float(diff.mean()))
            changed_pixels += int((diff > 1e-4).sum())
            lifted_from_zero += int(((prev_f <= 0.03) & (next_f > 0.03)).sum())
            suppressed_to_near_zero += int(((prev_f > 0.03) & (next_f <= 0.03)).sum())

        logger.info(
            "Alpha stage delta: stage=%s mean_abs_diff=%.6f changed_pixels=%s lifted_from_zero=%s suppressed_to_near_zero=%s",
            stage,
            float(np.mean(mean_abs_diffs)),
            changed_pixels,
            lifted_from_zero,
            suppressed_to_near_zero,
        )
    
    def process(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback=None
    ) -> dict:
        """
        处理视频、序列帧或单张图片
        
        Args:
            input_path: 输入文件或目录路径
            output_dir: 输出目录
            progress_callback: 进度回调函数 (current, total, stage)
        
        Returns:
            dict: 处理结果信息
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Starting process: input=%s output=%s quality=%s use_ai=%s ai_model=%s",
            input_path,
            output_dir,
            getattr(self.config.quality_mode, "value", self.config.quality_mode),
            getattr(self.config, "use_ai", False),
            getattr(self.config, "ai_model", "auto"),
        )
        
        start_time = time.time()
        timings = {}
        
        # 1. 输入解码
        self._notify(progress_callback, 0, 100, "decoding")
        stage_start = time.time()
        frames, meta = self._decode_input(input_path)
        timings["decode"] = time.time() - stage_start
        total_frames = len(frames)
        
        if total_frames == 0:
            raise ValueError(f"No frames found in {input_path}")
        
        logger.info(
            "Loaded %s frames, resolution=%sx%s",
            total_frames,
            meta["width"],
            meta["height"],
        )
        
        # 2. 背景模式识别
        self._notify(progress_callback, 5, 100, "analyzing")
        stage_start = time.time()
        if self.config.background_mode == BackgroundMode.AUTO:
            bg_mode = self.analyzer.analyze(frames)
            logger.info("Auto-detected background: %s", bg_mode.value)
        else:
            bg_mode = self.config.background_mode
            logger.info("Using configured background mode: %s", bg_mode.value)
        timings["analyze"] = time.time() - stage_start
        
        # 3. 生成 Matte
        self._notify(progress_callback, 10, 100, "matting")
        stage_start = time.time()
        alphas = self._generate_matte(frames, bg_mode, progress_callback)
        timings["matte"] = time.time() - stage_start
        
        # 4. 边缘细化
        timings["refine"] = 0.0
        if self.config.quality_mode in (QualityMode.STANDARD, QualityMode.HIGH):
            self._notify(progress_callback, 50, 100, "refining")
            stage_start = time.time()
            alphas_before_refine = [alpha.copy() for alpha in alphas]
            alphas = self.refiner.refine(frames, alphas)
            self._log_alpha_stage_delta("refine", alphas_before_refine, alphas)
            timings["refine"] = time.time() - stage_start
        else:
            logger.info("Skipping refine stage for quality mode: %s", self.config.quality_mode.value)
        
        # 5. 去噪点（EZ-CorridorKey 风格）
        timings["despeckle"] = 0.0
        if self.config.quality_mode in (QualityMode.STANDARD, QualityMode.HIGH):
            self._notify(progress_callback, 55, 100, "despeckling")
            stage_start = time.time()
            alphas_before_despeckle = [alpha.copy() for alpha in alphas]
            despeckle_context = {
                "active_ai_model": getattr(getattr(self, "hybrid_matte", None), "last_active_ai_model", None),
            }
            alphas = self.despeckle.process(alphas, frames=frames, context=despeckle_context)
            self._log_alpha_stage_delta("despeckle", alphas_before_despeckle, alphas)
            timings["despeckle"] = time.time() - stage_start
        else:
            logger.info("Skipping despeckle stage for quality mode: %s", self.config.quality_mode.value)
        
        # 6. 时序稳定（移到颜色去污染之前，避免绿色泄漏）
        timings["stabilize"] = 0.0
        if self.config.quality_mode in (QualityMode.STANDARD, QualityMode.HIGH) and total_frames > 1:
            self._notify(progress_callback, 60, 100, "stabilizing")
            stage_start = time.time()
            alphas = self.stabilizer.stabilize(alphas)
            timings["stabilize"] = time.time() - stage_start
        else:
            logger.info(
                "Skipping stabilize stage: quality=%s total_frames=%s",
                self.config.quality_mode.value,
                total_frames,
            )
        
        # 7. 颜色去污染（在时序稳定之后，基于稳定的 alpha）
        self._notify(progress_callback, 75, 100, "decontaminating")
        stage_start = time.time()
        frames = self.decontaminate.process(frames, alphas, bg_mode)
        timings["decontaminate"] = time.time() - stage_start
        
        # 7. RGBA 合成与输出
        self._notify(progress_callback, 85, 100, "encoding")
        stage_start = time.time()
        self._encode_output(frames, alphas, output_dir, meta)
        timings["encode"] = time.time() - stage_start
        
        elapsed = time.time() - start_time
        timings["total"] = elapsed
        fps = total_frames / elapsed if elapsed > 0 else 0
        
        result = {
            "status": "success",
            "frames_processed": total_frames,
            "output_dir": str(output_dir),
            "background_mode": bg_mode.value if hasattr(bg_mode, 'value') else str(bg_mode),
            "quality_mode": self.config.quality_mode.value if hasattr(self.config.quality_mode, 'value') else str(self.config.quality_mode),
            "elapsed_time": elapsed,
            "fps": fps,
            "timings": timings,
        }
        
        logger.info(
            "Stage timings summary: "
            "decode=%.2fs analyze=%.2fs matte=%.2fs refine=%.2fs "
            "despeckle=%.2fs stabilize=%.2fs decontaminate=%.2fs encode=%.2fs total=%.2fs",
            timings["decode"],
            timings["analyze"],
            timings["matte"],
            timings["refine"],
            timings["despeckle"],
            timings["stabilize"],
            timings["decontaminate"],
            timings["encode"],
            timings["total"],
        )
        logger.info(
            "Process completed: frames=%s elapsed=%.1fs fps=%.2f background=%s output=%s",
            total_frames,
            elapsed,
            fps,
            bg_mode.value if hasattr(bg_mode, "value") else str(bg_mode),
            output_dir,
        )
        return result
    
    def _decode_input(self, input_path: Path):
        """解码输入"""
        input_kind = detect_input_kind(input_path)
        if input_kind == InputKind.VIDEO:
            decoder = VideoDecoder()
            return decoder.decode(input_path)
        if input_kind == InputKind.IMAGE:
            decoder = ImageDecoder()
            return decoder.decode(input_path)
        if input_kind == InputKind.SEQUENCE:
            decoder = SequenceDecoder()
            return decoder.decode(input_path)
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    def _generate_matte(self, frames, bg_mode, progress_callback):
        """生成 matte"""
        total = len(frames)
        
        def on_progress(i, total):
            if progress_callback:
                progress = 10 + int((i / total) * 40)
                self._notify(progress_callback, progress, 100, "matting")
        
        # 使用混合抠图引擎
        alphas = self.hybrid_matte.generate_sequence(frames, bg_mode, on_progress)
        
        return alphas
    
    def _encode_output(self, frames, alphas, output_dir, meta):
        """编码输出"""
        output_dir = Path(output_dir)
        
        # RGBA 合成（统一为 float32 [0,1]）
        rgba_frames = []
        for frame, alpha in zip(frames, alphas):
            # frame 可能是 uint8 [0,255] 或 float [0,1]
            if frame.dtype == np.uint8:
                frame_f = frame.astype(np.float32) / 255.0
            else:
                frame_f = frame.astype(np.float32)
            
            rgba = np.dstack([frame_f, alpha])
            rgba_frames.append(rgba)
        
        # EZ-CorridorKey 风格输出
        # 1. FG (Straight foreground) - 直接前景，未预乘
        if self.config.output_fg:
            fg_dir = output_dir / "FG"
            fg_dir.mkdir(exist_ok=True)
            for i, rgba in enumerate(rgba_frames):
                fg = rgba[:, :, :3]  # RGB only
                out_path = fg_dir / f"fg_{i:06d}.png"
                self.encoder.encode_image(fg, out_path)
            logger.info("Saved %s FG frames to %s", len(rgba_frames), fg_dir)
        
        # 2. Matte (Linear alpha) - 线性 alpha 遮罩
        if self.config.output_matte:
            matte_dir = output_dir / "Matte"
            matte_dir.mkdir(exist_ok=True)
            for i, alpha in enumerate(alphas):
                matte = (alpha * 255).astype(np.uint8)
                out_path = matte_dir / f"matte_{i:06d}.png"
                self.encoder.encode_grayscale(matte, out_path)
            logger.info("Saved %s matte frames to %s", len(alphas), matte_dir)
        
        # 3. Comp (Premultiplied) - 预乘合成
        if self.config.output_comp:
            comp_dir = output_dir / "Comp"
            comp_dir.mkdir(exist_ok=True)
            for i, rgba in enumerate(rgba_frames):
                alpha = rgba[:, :, 3:4]
                rgb = rgba[:, :, :3]
                comp = rgb * alpha  # 预乘
                out_path = comp_dir / f"comp_{i:06d}.png"
                self.encoder.encode_image(comp, out_path)
            logger.info("Saved %s comp frames to %s", len(rgba_frames), comp_dir)
        
        # 4. Processed (RGBA) - 处理后完整 RGBA
        if self.config.output_processed:
            processed_dir = output_dir / "Processed"
            processed_dir.mkdir(exist_ok=True)
            for i, rgba in enumerate(rgba_frames):
                out_path = processed_dir / f"processed_{i:06d}.png"
                self.encoder.encode_image(rgba, out_path)
            logger.info("Saved %s processed RGBA frames to %s", len(rgba_frames), processed_dir)
        
        # 输出遮罩（可选）
        if self.config.output_mask:
            mask_dir = output_dir / "mask"
            mask_dir.mkdir(exist_ok=True)
            for i, alpha in enumerate(alphas):
                mask = (alpha * 255).astype(np.uint8)
                out_path = mask_dir / f"mask_{i:06d}.png"
                self.encoder.encode_grayscale(mask, out_path)
            logger.info("Saved %s mask frames to %s", len(alphas), mask_dir)
    
    def _notify(self, callback, current, total, stage):
        """通知进度"""
        if callback:
            try:
                callback(current, total, stage)
            except Exception:
                pass
