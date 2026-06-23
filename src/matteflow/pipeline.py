"""MatteFlow 主 Pipeline"""

import inspect
import logging
import time
from pathlib import Path
from typing import Union, Optional
import numpy as np

from .config import MattingConfig, QualityMode, BackgroundMode
from .errors import InputValidationError, JobCancelledError, ProgressCallbackError
from .input.decoder import ImageDecoder, SequenceDecoder, VideoDecoder
from .input.formats import InputKind, detect_input_kind
from .analysis.background_analyzer import BackgroundAnalyzer
from .analysis.alpha_quality import AlphaQualityAnalyzer
from .analysis.region_ownership import RegionOwnershipAnalyzer
from .matte.matte_fusion import MatteFusion
from .refine.edge_refiner import EdgeRefiner
from .refine.color_decontaminate import ColorDecontaminate
from .refine.despeckle import Despeckle
from .refine.effect_prop_repair import EffectPropRepair
from .temporal.temporal_stabilizer import TemporalStabilizer
from .output.encoder import RGBAEncoder

logger = logging.getLogger(__name__)


class MattingPipeline:
    """高质量抠图 Pipeline"""
    
    def __init__(self, config: Optional[MattingConfig] = None):
        self.config = config or MattingConfig()
        self.analyzer = BackgroundAnalyzer()
        self.quality_analyzer = AlphaQualityAnalyzer()
        self.region_analyzer = RegionOwnershipAnalyzer()
        self.fusion = MatteFusion()
        self.refiner = EdgeRefiner(self.config)
        self.decontaminate = ColorDecontaminate(self.config)
        self.despeckle = Despeckle(self.config)
        self.effect_prop_repair = EffectPropRepair(self.config)
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

    @staticmethod
    def _assert_sequence_length(stage: str, expected_count: int, actual_count: int, item_name: str) -> None:
        if expected_count != actual_count:
            raise RuntimeError(
                f"{item_name} count mismatch after {stage}: expected {expected_count}, got {actual_count}"
            )

    def _assert_frame_alpha_alignment(self, stage: str, frames, alphas) -> None:
        self._assert_sequence_length(stage, len(frames), len(alphas), "Alpha")
    
    def process(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        progress_callback=None,
        cancel_check=None,
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
        self._check_cancelled(cancel_check)
        stage_start = time.time()
        frames, meta = self._decode_input(input_path)
        timings["decode"] = time.time() - stage_start
        total_frames = len(frames)
        self._check_cancelled(cancel_check)
        
        if total_frames == 0:
            raise ValueError(f"No frames found in {input_path}")
        self._validate_frame_limit(total_frames, input_path)
        
        logger.info(
            "Loaded %s frames, resolution=%sx%s",
            total_frames,
            meta["width"],
            meta["height"],
        )
        
        # 2. 背景模式识别
        self._notify(progress_callback, 5, 100, "analyzing")
        self._check_cancelled(cancel_check)
        stage_start = time.time()
        if self.config.background_mode == BackgroundMode.AUTO:
            bg_mode = self.analyzer.analyze(frames)
            logger.info("Auto-detected background: %s", bg_mode.value)
        else:
            bg_mode = self.config.background_mode
            logger.info("Using configured background mode: %s", bg_mode.value)
        timings["analyze"] = time.time() - stage_start
        self._check_cancelled(cancel_check)
        
        # 3. 生成 Matte
        self._notify(progress_callback, 10, 100, "matting")
        self._check_cancelled(cancel_check)
        stage_start = time.time()
        generate_matte_params = inspect.signature(self._generate_matte).parameters
        if "cancel_check" in generate_matte_params:
            alphas = self._generate_matte(
                frames,
                bg_mode,
                progress_callback,
                cancel_check=cancel_check,
            )
        else:
            alphas = self._generate_matte(frames, bg_mode, progress_callback)
        self._assert_frame_alpha_alignment("matte", frames, alphas)
        timings["matte"] = time.time() - stage_start
        self._check_cancelled(cancel_check)
        
        # 4. 边缘细化
        timings["refine"] = 0.0
        if self.config.quality_mode in (QualityMode.STANDARD, QualityMode.HIGH):
            self._notify(progress_callback, 50, 100, "refining")
            self._check_cancelled(cancel_check)
            stage_start = time.time()
            alphas_before_refine = [alpha.copy() for alpha in alphas]
            region_context = self._build_region_context(frames, alphas)
            alphas = self._call_with_optional_context(
                self.refiner.refine,
                frames,
                alphas,
                context=region_context,
            )
            self._assert_frame_alpha_alignment("refine", frames, alphas)
            self._log_alpha_stage_delta("refine", alphas_before_refine, alphas)
            timings["refine"] = time.time() - stage_start
            self._check_cancelled(cancel_check)
        else:
            logger.info("Skipping refine stage for quality mode: %s", self.config.quality_mode.value)
        
        # 5. 去噪点（EZ-CorridorKey 风格）
        timings["despeckle"] = 0.0
        if self.config.quality_mode in (QualityMode.STANDARD, QualityMode.HIGH):
            self._notify(progress_callback, 55, 100, "despeckling")
            self._check_cancelled(cancel_check)
            stage_start = time.time()
            alphas_before_despeckle = [alpha.copy() for alpha in alphas]
            despeckle_context = {
                "active_ai_model": getattr(getattr(self, "hybrid_matte", None), "last_active_ai_model", None),
            }
            despeckle_context.update(self._build_region_context(frames, alphas))
            alphas = self._call_with_optional_context(
                self.despeckle.process,
                alphas,
                frames=frames,
                context=despeckle_context,
            )
            self._assert_frame_alpha_alignment("despeckle", frames, alphas)
            self._log_alpha_stage_delta("despeckle", alphas_before_despeckle, alphas)
            timings["despeckle"] = time.time() - stage_start
            self._check_cancelled(cancel_check)
        else:
            logger.info("Skipping despeckle stage for quality mode: %s", self.config.quality_mode.value)
        
        # 5.5. 特效道具完整性修复：用传统 key 的结构证据修复 AI 对小发光件的误抠
        timings["effect_prop_repair"] = 0.0
        if self.config.quality_mode in (QualityMode.STANDARD, QualityMode.HIGH):
            stage_start = time.time()
            alphas_before_effect_repair = [alpha.copy() for alpha in alphas]
            repair_bg_mode = self._effective_decontamination_mode(bg_mode)
            active_model = getattr(getattr(self, "hybrid_matte", None), "last_active_ai_model", None)
            repair_context = self._build_region_context(frames, alphas)
            alphas = self._call_with_optional_context(
                self.effect_prop_repair.process,
                frames,
                alphas,
                repair_bg_mode,
                active_model=active_model,
                context=repair_context,
            )
            self._assert_frame_alpha_alignment("effect_prop_repair", frames, alphas)
            self._log_alpha_stage_delta("effect_prop_repair", alphas_before_effect_repair, alphas)
            timings["effect_prop_repair"] = time.time() - stage_start
            self._check_cancelled(cancel_check)

        # 6. 时序稳定（移到颜色去污染之前，避免绿色泄漏）
        timings["stabilize"] = 0.0
        if self.config.quality_mode in (QualityMode.STANDARD, QualityMode.HIGH) and total_frames > 1:
            self._notify(progress_callback, 60, 100, "stabilizing")
            self._check_cancelled(cancel_check)
            stage_start = time.time()
            alphas = self._stabilize_alphas(frames, alphas)
            self._assert_frame_alpha_alignment("stabilize", frames, alphas)
            timings["stabilize"] = time.time() - stage_start
            self._check_cancelled(cancel_check)
        else:
            logger.info(
                "Skipping stabilize stage: quality=%s total_frames=%s",
                self.config.quality_mode.value,
                total_frames,
            )
        
        # 7. 质量诊断调试输出（基于稳定后的 alpha）
        timings["quality_debug"] = 0.0
        if getattr(self.config, "output_debug", False):
            stage_start = time.time()
            self._write_quality_debug_outputs(frames, alphas, output_dir)
            timings["quality_debug"] = time.time() - stage_start
            self._check_cancelled(cancel_check)

        # 8. 颜色去污染（在时序稳定之后，基于稳定的 alpha）
        self._notify(progress_callback, 75, 100, "decontaminating")
        self._check_cancelled(cancel_check)
        stage_start = time.time()
        decontaminate_bg_mode = self._effective_decontamination_mode(bg_mode)
        decontaminate_context = self._build_region_context(frames, alphas)
        decontaminate_context["active_ai_model"] = getattr(
            getattr(self, "hybrid_matte", None),
            "last_active_ai_model",
            None,
        )
        frames = self._call_with_optional_context(
            self.decontaminate.process,
            frames,
            alphas,
            decontaminate_bg_mode,
            context=decontaminate_context,
        )
        self._assert_sequence_length("decontaminate", total_frames, len(frames), "Frame")
        self._assert_frame_alpha_alignment("decontaminate", frames, alphas)
        timings["decontaminate"] = time.time() - stage_start
        self._check_cancelled(cancel_check)
        
        # 7. RGBA 合成与输出
        self._notify(progress_callback, 85, 100, "encoding")
        self._check_cancelled(cancel_check)
        stage_start = time.time()
        encode_output_params = inspect.signature(self._encode_output).parameters
        if "cancel_check" in encode_output_params:
            self._encode_output(frames, alphas, output_dir, meta, cancel_check=cancel_check)
        else:
            self._encode_output(frames, alphas, output_dir, meta)
        timings["encode"] = time.time() - stage_start
        self._check_cancelled(cancel_check)
        
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
            "despeckle=%.2fs effect_prop_repair=%.2fs stabilize=%.2fs "
            "decontaminate=%.2fs encode=%.2fs total=%.2fs",
            timings["decode"],
            timings["analyze"],
            timings["matte"],
            timings["refine"],
            timings["despeckle"],
            timings["effect_prop_repair"],
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

    @staticmethod
    def _check_cancelled(cancel_check) -> None:
        if cancel_check is not None and cancel_check():
            raise JobCancelledError("Processing cancelled by user")

    def _validate_frame_limit(self, total_frames: int, input_path: Path) -> None:
        max_input_frames = getattr(self.config, "max_input_frames", None)
        if max_input_frames is None or max_input_frames <= 0:
            return
        if total_frames > max_input_frames:
            raise InputValidationError(
                f"Input {input_path} has {total_frames} frames, "
                f"which exceeds configured max_input_frames={max_input_frames}. "
                "Increase max_input_frames or split the input before processing."
            )

    def _effective_decontamination_mode(self, bg_mode: BackgroundMode) -> BackgroundMode:
        if bg_mode != BackgroundMode.UNKNOWN:
            return bg_mode

        active_model = getattr(getattr(self, "hybrid_matte", None), "last_active_ai_model", None)
        if active_model in {"traditional_green_fallback", "green_screen_fallback"}:
            return BackgroundMode.GREEN_SCREEN
        if active_model in {"traditional_black_fallback", "black_background_fallback"}:
            return BackgroundMode.BLACK_BACKGROUND
        return bg_mode

    def _stabilize_alphas(self, frames, alphas):
        """Stabilize alpha mattes with frame context when the stabilizer supports it."""
        stabilize_params = inspect.signature(self.stabilizer.stabilize).parameters
        if "frames" in stabilize_params:
            return self.stabilizer.stabilize(alphas, frames=frames)
        return self.stabilizer.stabilize(alphas)

    @staticmethod
    def _call_with_optional_context(callable_obj, *args, context=None, **kwargs):
        """Pass shared region context only to stage implementations that accept it."""
        params = inspect.signature(callable_obj).parameters
        if context is not None and "context" in params:
            kwargs["context"] = context
        return callable_obj(*args, **kwargs)

    def _build_region_context(self, frames, alphas, base_alphas=None) -> dict:
        """Build shared per-frame region ownership for downstream repair stages."""
        ownerships = []
        for index, (frame, alpha) in enumerate(zip(frames, alphas)):
            base_alpha = base_alphas[index] if base_alphas is not None and index < len(base_alphas) else None
            ownerships.append(self.region_analyzer.analyze(frame, alpha, base_alpha))
        return {"region_ownership": ownerships}

    def _write_quality_debug_outputs(self, frames, alphas, output_dir: Path):
        """Write alpha quality overlays and a compact metric report when debug output is enabled."""
        if not getattr(self.config, "output_debug", False):
            return None

        output_dir = Path(output_dir)
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        report = self.quality_analyzer.analyze_sequence(frames, alphas)
        for i, (frame, alpha) in enumerate(zip(frames, alphas)):
            overlay = self.quality_analyzer.build_debug_overlay(frame, alpha)
            self.encoder.encode_image(overlay, debug_dir / f"quality_overlay_{i:06d}.png")
            region_overlay = self.region_analyzer.build_debug_overlay(frame, alpha)
            self.encoder.encode_image(region_overlay, debug_dir / f"region_ownership_{i:06d}.png")

        report_path = debug_dir / "quality_report.txt"
        report_path.write_text(
            "\n".join(
                [
                    f"frame_count={report.frame_count}",
                    f"overall_score={report.overall_score:.6f}",
                    f"mean_edge_uncertainty={report.mean_edge_uncertainty:.6f}",
                    f"speckle_pixels={report.speckle_pixels}",
                    f"hole_pixels={report.hole_pixels}",
                    f"background_residue={report.background_residue:.6f}",
                    f"temporal_flicker={report.temporal_flicker:.6f}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return report
    
    def _decode_input(self, input_path: Path):
        """解码输入"""
        input_kind = detect_input_kind(input_path)
        max_input_frames = getattr(self.config, "max_input_frames", None)
        if input_kind == InputKind.VIDEO:
            decoder = VideoDecoder(max_frames=max_input_frames)
            return decoder.decode(input_path)
        if input_kind == InputKind.IMAGE:
            decoder = ImageDecoder()
            return decoder.decode(input_path)
        if input_kind == InputKind.SEQUENCE:
            decoder = SequenceDecoder(max_frames=max_input_frames)
            return decoder.decode(input_path)
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    def _generate_matte(self, frames, bg_mode, progress_callback, cancel_check=None):
        """生成 matte"""
        total = len(frames)
        
        def on_progress(i, total):
            if progress_callback:
                progress = 10 + int((i / total) * 40)
                self._notify(progress_callback, progress, 100, "matting")
        
        # 使用混合抠图引擎
        alphas = self.hybrid_matte.generate_sequence(
            frames,
            bg_mode,
            progress_callback=on_progress,
            cancel_check=cancel_check,
        )
        
        return alphas
    
    def _encode_output(self, frames, alphas, output_dir, meta, cancel_check=None):
        """编码输出"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_ext = f".{getattr(self.config, 'output_format', 'png').lower().lstrip('.')}"
        premultiply_rgb = bool(getattr(self.config, "output_premultiplied", False))
        self._assert_frame_alpha_alignment("encode", frames, alphas)

        def _check_encode_cancelled() -> None:
            self._check_cancelled(cancel_check)
        
        def _frame_to_float(frame):
            if frame.dtype == np.uint8:
                return frame.astype(np.float32) / 255.0
            return frame.astype(np.float32)

        def _compose_rgba(frame, alpha, premultiplied: bool) -> np.ndarray:
            frame_f = _frame_to_float(frame)
            alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
            if premultiplied:
                frame_f = frame_f * alpha_f[..., np.newaxis]
            return np.dstack([frame_f, alpha_f])
        
        # 1. FG (Straight foreground) - 直接前景，未预乘
        if self.config.output_fg:
            fg_dir = output_dir / "FG"
            fg_dir.mkdir(exist_ok=True)
            for i, (frame, alpha) in enumerate(zip(frames, alphas)):
                _check_encode_cancelled()
                rgba = _compose_rgba(frame, alpha, premultiplied=False)
                fg = rgba[:, :, :3]  # RGB only
                out_path = fg_dir / f"fg_{i:06d}{output_ext}"
                self.encoder.encode_image(fg, out_path)
            logger.info("Saved %s FG frames to %s", len(frames), fg_dir)
        
        # 2. Matte (Linear alpha) - 线性 alpha 遮罩
        if self.config.output_matte:
            matte_dir = output_dir / "Matte"
            matte_dir.mkdir(exist_ok=True)
            for i, alpha in enumerate(alphas):
                _check_encode_cancelled()
                matte = (alpha * 255).astype(np.uint8)
                out_path = matte_dir / f"matte_{i:06d}.png"
                self.encoder.encode_grayscale(matte, out_path)
            logger.info("Saved %s matte frames to %s", len(alphas), matte_dir)
        
        # 3. Comp (Premultiplied) - 预乘合成
        if self.config.output_comp:
            comp_dir = output_dir / "Comp"
            comp_dir.mkdir(exist_ok=True)
            for i, (frame, alpha) in enumerate(zip(frames, alphas)):
                _check_encode_cancelled()
                rgba = _compose_rgba(frame, alpha, premultiply_rgb)
                comp = rgba[:, :, :3]
                out_path = comp_dir / f"comp_{i:06d}{output_ext}"
                self.encoder.encode_image(comp, out_path)
            logger.info("Saved %s comp frames to %s", len(frames), comp_dir)
        
        # 4. Processed (RGBA) - 处理后完整 RGBA
        if self.config.output_processed:
            processed_dir = output_dir / "Processed"
            processed_dir.mkdir(exist_ok=True)
            for i, (frame, alpha) in enumerate(zip(frames, alphas)):
                _check_encode_cancelled()
                rgba = _compose_rgba(frame, alpha, premultiply_rgb)
                out_path = processed_dir / f"processed_{i:06d}{output_ext}"
                self.encoder.encode_image(rgba, out_path)
            logger.info("Saved %s processed RGBA frames to %s", len(frames), processed_dir)
        
        # 输出遮罩（可选）
        if self.config.output_mask:
            mask_dir = output_dir / "mask"
            mask_dir.mkdir(exist_ok=True)
            for i, alpha in enumerate(alphas):
                _check_encode_cancelled()
                mask = (alpha * 255).astype(np.uint8)
                out_path = mask_dir / f"mask_{i:06d}.png"
                self.encoder.encode_grayscale(mask, out_path)
            logger.info("Saved %s mask frames to %s", len(alphas), mask_dir)
    
    def _notify(self, callback, current, total, stage):
        """通知进度"""
        if callback:
            try:
                callback(current, total, stage)
            except Exception as exc:
                logger.exception(
                    "Progress callback failed: current=%s total=%s stage=%s",
                    current,
                    total,
                    stage,
                )
                if isinstance(exc, ProgressCallbackError):
                    raise
                raise ProgressCallbackError(
                    f"Progress callback failed during {stage}: {exc}"
                ) from exc
