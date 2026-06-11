"""混合抠图引擎 - AI + 传统算法融合"""

import logging
import inspect
import numpy as np
import cv2
from typing import List

from ..config import MattingConfig, BackgroundMode

logger = logging.getLogger(__name__)


class HybridMatte:
    """
    混合抠图引擎
    - 绿幕：RVM AI 为主
    - 黑底：传统算法 + AI 辅助增强
    """
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self.last_active_ai_model = None
        self.last_fallback_quality_metrics = None
        self.last_green_screen_layer_debug = None
        self.rvm = None
        self.birefnet = None
        self.rmbg = None
        self.rembg = None
        self.gvm = None
        self.matanyone2 = None
        self.sam2 = None
        self.corridorkey = None
        self.green_matte = None
        self.black_matte = None
        
        # 尝试加载 AI 模型（仅在 use_ai=True 时）
        if config.use_ai:
            # 根据用户选择强制加载指定模型
            ai_model = getattr(config, 'ai_model', 'auto')
            logger.info("Requested AI model: %s", ai_model)
            
            if ai_model == "gvm":
                self._load_gvm(config)
            elif ai_model == "matanyone2":
                self._load_matanyone2(config)
            elif ai_model == "corridorkey":
                self._load_corridorkey(config)
            elif ai_model == "birefnet":
                self._load_birefnet(config)
            elif ai_model == "rvm":
                self._load_rvm(config)
            elif ai_model == "rembg":
                self._load_rembg(config)
            elif ai_model == "sam2":
                self._load_sam2(config)
            else:
                # auto 模式：按优先级自动加载
                self._load_auto(config)
        else:
            logger.info("AI disabled, using traditional algorithms only")
        
        # 传统算法备用
        from .green_screen_matte import GreenScreenMatte
        from .black_background_matte import BlackBackgroundMatte
        self.green_matte = GreenScreenMatte(config)
        self.black_matte = BlackBackgroundMatte(config)
    
    def _load_gvm(self, config):
        """加载 GVM"""
        try:
            from .gvm_matte import GVMMatte
            self.gvm = GVMMatte(config)
            if self.gvm.model is not None:
                logger.info("GVM loaded successfully")
            else:
                raise RuntimeError("GVM model is None")
        except Exception as e:
            logger.exception("GVM loading failed")
            raise RuntimeError(f"GVM loading failed: {e}")
    
    def _load_matanyone2(self, config):
        """加载 MatAnyone2"""
        try:
            from .matanyone2_matte import MatAnyone2Matte
            self.matanyone2 = MatAnyone2Matte(config)
            if self.matanyone2.model is not None:
                logger.info("MatAnyone2 loaded successfully")
            else:
                raise RuntimeError("MatAnyone2 model is None")
        except Exception as e:
            logger.exception("MatAnyone2 loading failed")
            raise RuntimeError(f"MatAnyone2 loading failed: {e}")
    
    def _load_corridorkey(self, config):
        """加载 CorridorKey"""
        try:
            from .corridorkey_matte import CorridorKeyMatte
            self.corridorkey = CorridorKeyMatte(config)
            if self.corridorkey.model is not None:
                logger.info("CorridorKey loaded successfully")
            else:
                raise RuntimeError("CorridorKey model is None")
        except Exception as e:
            logger.exception("CorridorKey loading failed")
            raise RuntimeError(f"CorridorKey loading failed: {e}")
    
    def _load_birefnet(self, config):
        """加载 BiRefNet"""
        try:
            from .birefnet_matte import BiRefNetMatte
            self.birefnet = BiRefNetMatte(config)
            if self.birefnet.model is not None:
                logger.info("BiRefNet loaded successfully")
            else:
                raise RuntimeError("BiRefNet model is None")
        except Exception as e:
            logger.exception("BiRefNet loading failed")
            raise RuntimeError(f"BiRefNet loading failed: {e}")
    
    def _load_rvm(self, config):
        """加载 RVM"""
        try:
            from .rvm_matte import RVMMatte
            self.rvm = RVMMatte(config)
            if self.rvm.model is not None:
                logger.info("RVM loaded successfully")
            else:
                raise RuntimeError("RVM model is None")
        except Exception as e:
            logger.exception("RVM loading failed")
            raise RuntimeError(f"RVM loading failed: {e}")
    
    def _load_rembg(self, config):
        """加载 rembg"""
        try:
            from .rembg_matte import RembgMatte
            self.rembg = RembgMatte(config)
            if self.rembg._available:
                logger.info("rembg loaded successfully")
            else:
                raise RuntimeError("rembg not available")
        except Exception as e:
            logger.exception("rembg loading failed")
            raise RuntimeError(f"rembg loading failed: {e}")
    
    def _load_sam2(self, config):
        """加载 SAM2"""
        try:
            from .sam2_matte import SAM2Matte
            self.sam2 = SAM2Matte(config)
            if self.sam2.predictor is not None or self.sam2.model is not None:
                logger.info("SAM2 loaded successfully")
            else:
                raise RuntimeError("SAM2 model is None")
        except Exception as e:
            logger.exception("SAM2 loading failed")
            raise RuntimeError(f"SAM2 loading failed: {e}")
    
    def _load_auto(self, config):
        """自动加载：按优先级尝试"""
        # 1. 优先加载 GVM
        try:
            from .gvm_matte import GVMMatte
            self.gvm = GVMMatte(config)
            if self.gvm.model is not None:
                logger.info("Auto-selected GVM")
                return
            else:
                self.gvm = None
        except Exception as e:
            logger.warning("GVM not available for auto selection: %s", e)
        
        # 2. 加载 MatAnyone2
        try:
            from .matanyone2_matte import MatAnyone2Matte
            self.matanyone2 = MatAnyone2Matte(config)
            if self.matanyone2.model is not None:
                logger.info("Auto-selected MatAnyone2")
                return
            else:
                self.matanyone2 = None
        except Exception as e:
            logger.warning("MatAnyone2 not available for auto selection: %s", e)
        
        # 3. 加载 SAM2
        try:
            from .sam2_matte import SAM2Matte
            self.sam2 = SAM2Matte(config)
            if self.sam2.predictor is not None or self.sam2.model is not None:
                logger.info("Auto-selected SAM2")
                return
            else:
                self.sam2 = None
        except Exception as e:
            logger.warning("SAM2 not available for auto selection: %s", e)
        
        # 4. 加载 CorridorKey
        try:
            from .corridorkey_matte import CorridorKeyMatte
            self.corridorkey = CorridorKeyMatte(config)
            if self.corridorkey.model is not None:
                logger.info("Auto-selected CorridorKey")
                return
            else:
                self.corridorkey = None
        except Exception as e:
            logger.warning("CorridorKey not available for auto selection: %s", e)
        
        # 5. 回退到 rembg
        try:
            from .rembg_matte import RembgMatte
            self.rembg = RembgMatte(config)
            if self.rembg._available:
                logger.info("Auto-selected rembg")
                return
            else:
                self.rembg = None
        except Exception as e:
            logger.warning("rembg not available for auto selection: %s", e)
        
        # 6. 回退到 BiRefNet
        try:
            from .birefnet_matte import BiRefNetMatte
            self.birefnet = BiRefNetMatte(config)
            if self.birefnet.model is not None:
                logger.info("Auto-selected BiRefNet")
                return
            else:
                self.birefnet = None
        except Exception as e:
            logger.warning("BiRefNet not available for auto selection: %s", e)
        
        # 7. 最后回退到 RVM
        try:
            from .rvm_matte import RVMMatte
            self.rvm = RVMMatte(config)
            if self.rvm.model is not None:
                logger.info("Auto-selected RVM")
                return
            else:
                self.rvm = None
        except Exception as e:
            logger.warning("RVM not available for auto selection: %s", e)
        
        # 传统算法备用（始终加载）
        from .green_screen_matte import GreenScreenMatte
        from .black_background_matte import BlackBackgroundMatte
        self.green_matte = GreenScreenMatte(config)
        self.black_matte = BlackBackgroundMatte(config)
    
    @staticmethod
    def _check_cancelled(cancel_check) -> None:
        if cancel_check is not None and cancel_check():
            from ..errors import JobCancelledError

            raise JobCancelledError("Matting cancelled by user")

    def _generate_with_engine(
        self,
        engine,
        frames: List[np.ndarray],
        progress_callback=None,
        cancel_check=None,
    ) -> List[np.ndarray]:
        self._check_cancelled(cancel_check)
        kwargs = {}
        params = inspect.signature(engine.generate_sequence).parameters
        if "progress_callback" in params:
            kwargs["progress_callback"] = progress_callback
        if "cancel_check" in params:
            kwargs["cancel_check"] = cancel_check
        return engine.generate_sequence(frames, **kwargs)

    def _generate_single_frame_with_engine(
        self,
        engine,
        frame: np.ndarray,
        cancel_check=None,
    ) -> np.ndarray:
        alphas = self._generate_with_engine(engine, [frame], cancel_check=cancel_check)
        if not alphas:
            raise RuntimeError("AI engine returned no alpha mattes for single-frame refinement")
        return alphas[0]

    def _select_sequence_ai_engine(self):
        ai_model = getattr(self.config, "ai_model", "auto")

        def is_ready(engine, attr="model"):
            if engine is None:
                return False
            return getattr(engine, attr, None) is not None

        explicit_candidates = (
            ("gvm", self.gvm, "model"),
            ("matanyone2", self.matanyone2, "model"),
            ("corridorkey", self.corridorkey, "model"),
            ("sam2", self.sam2, "predictor"),
            ("rembg", self.rembg, "_available"),
            ("birefnet", self.birefnet, "model"),
            ("rvm", self.rvm, "model"),
        )
        for name, engine, attr in explicit_candidates:
            if ai_model != name:
                continue
            if name == "sam2":
                if engine is not None and (
                    getattr(engine, "predictor", None) is not None or getattr(engine, "model", None) is not None
                ):
                    return name, engine, False
            elif name == "rembg":
                if engine is not None and getattr(engine, "_available", False):
                    return name, engine, False
            elif is_ready(engine, attr):
                return name, engine, False
            return None, None, False

        auto_candidates = (
            ("gvm", self.gvm),
            ("corridorkey", self.corridorkey),
            ("rembg", self.rembg),
            ("birefnet", self.birefnet),
            ("rvm", self.rvm),
        )
        for name, engine in auto_candidates:
            if name == "rembg":
                if engine is not None and getattr(engine, "_available", False):
                    return name, engine, True
                continue
            if engine is not None and getattr(engine, "model", None) is not None:
                return name, engine, True

        return None, None, ai_model == "auto"

    def _generate_unknown_background_matte(
        self,
        frames: List[np.ndarray],
        progress_callback=None,
        cancel_check=None,
    ) -> List[np.ndarray]:
        if not getattr(self.config, "use_ai", False):
            raise RuntimeError("Unable to determine background mode automatically. Please choose green screen or black background.")

        ai_name, ai_engine, auto_selected = self._select_sequence_ai_engine()
        if ai_engine is None:
            raise RuntimeError("Unable to determine background mode automatically. Please choose green screen or black background.")

        self.last_active_ai_model = ai_name
        suffix = " (auto)" if auto_selected else ""
        logger.info("Using %s for unknown background fallback%s", ai_name.upper(), suffix)
        return self._generate_with_engine(ai_engine, frames, progress_callback, cancel_check)

    def generate_sequence(
        self,
        frames: List[np.ndarray],
        bg_mode: BackgroundMode,
        progress_callback=None,
        cancel_check=None,
    ) -> List[np.ndarray]:
        """序列抠图"""
        self.last_active_ai_model = None
        self.last_green_screen_layer_debug = None
        if bg_mode == BackgroundMode.GREEN_SCREEN:
            return self._green_screen_matte(frames, progress_callback, cancel_check)
        elif bg_mode == BackgroundMode.BLACK_BACKGROUND:
            return self._black_background_matte(frames, progress_callback, cancel_check)
        elif bg_mode == BackgroundMode.UNKNOWN:
            return self._generate_unknown_background_matte(frames, progress_callback, cancel_check)
        else:
            # Auto - 默认用绿幕逻辑
            return self._green_screen_matte(frames, progress_callback, cancel_check)
    
    def _green_screen_matte(self, frames: List[np.ndarray], progress_callback, cancel_check=None) -> List[np.ndarray]:
        """绿幕抠图 - 修复：传统算法为主，AI 仅辅助边缘细化"""
        logger.info(
            "Starting green screen matting: frames=%s ai_enhance=%s ai_model=%s",
            len(frames),
            self.config.ai_enhance,
            getattr(self.config, "ai_model", "auto"),
        )
        
        # 绿幕场景：传统算法效果通常更好，AI 容易误抠白色/浅色区域
        # 策略：传统算法生成基础 alpha，AI 只用于边缘细化（如果启用 AI 增强）
        
        # 1. 传统算法生成基础 alpha（更可靠）
        logger.info("Using traditional green screen as base")
        base_alphas = []
        for i, frame in enumerate(frames):
            self._check_cancelled(cancel_check)
            alpha = self.green_matte.generate(frame)
            base_alphas.append(alpha)
            if progress_callback and i % max(1, len(frames) // 20) == 0:
                progress_callback(i, len(frames))
        
        # 2. AI 增强模式：仅用于边缘细化（不是替代传统算法）
        if self.config.ai_enhance:
            ai_engine = None
            
            # 优先使用 GVM (Generative Video Matting)
            if self.gvm is not None and self.gvm.model is not None:
                ai_engine = self.gvm
                self.last_active_ai_model = "gvm"
                logger.info("Applying GVM edge refinement")
            # 其次使用 MatAnyone2 (人物视频抠图)
            elif self.matanyone2 is not None and self.matanyone2.model is not None:
                ai_engine = self.matanyone2
                self.last_active_ai_model = "matanyone2"
                logger.info("Applying MatAnyone2 edge refinement")
            # 然后使用 CorridorKey
            elif self.corridorkey is not None and self.corridorkey.model is not None:
                ai_engine = self.corridorkey
                self.last_active_ai_model = "corridorkey"
                logger.info("Applying CorridorKey edge refinement")
            elif self.rembg is not None and self.rembg._available:
                ai_engine = self.rembg
                self.last_active_ai_model = "rembg"
                logger.info("Applying rembg edge refinement")
            elif self.rmbg is not None and self.rmbg.model is not None:
                ai_engine = self.rmbg
                self.last_active_ai_model = "rmbg"
                logger.info("Applying RMBG-2 edge refinement")
            elif self.birefnet is not None and self.birefnet.model is not None:
                ai_engine = self.birefnet
                self.last_active_ai_model = "birefnet"
                logger.info("Applying BiRefNet edge refinement")
            elif self.rvm is not None and self.rvm.model is not None:
                ai_engine = self.rvm
                self.last_active_ai_model = "rvm"
                logger.info("Applying RVM edge refinement")
            
            if ai_engine is not None:
                refined_alphas = []
                for i, (frame, base_alpha) in enumerate(zip(frames, base_alphas)):
                    self._check_cancelled(cancel_check)
                    # 只在边缘区域用 AI 辅助
                    edge_mask = (base_alpha > 0.1) & (base_alpha < 0.9)
                    
                    if np.any(edge_mask):
                        # 用 AI 生成边缘 alpha
                        ai_alpha = self._generate_single_frame_with_engine(
                            ai_engine,
                            frame,
                            cancel_check=cancel_check,
                        )
                        
                        # 融合策略：
                        # - 核心前景 (alpha > 0.8)：传统算法（保护白色/毛发）
                        # - 边缘区域：传统 70% + AI 30%（AI 辅助平滑边缘）
                        # - 背景 (alpha < 0.1)：透明
                        result = base_alpha.copy()
                        
                        # 只在边缘区域混合 AI
                        blend = 0.3  # AI 占比 30%，传统占 70%
                        result[edge_mask] = base_alpha[edge_mask] * (1 - blend) + ai_alpha[edge_mask] * blend
                        
                        # 保护高置信度区域不被 AI 破坏
                        fg_mask = base_alpha > 0.9
                        result[fg_mask] = base_alpha[fg_mask]  # 完全用传统算法
                        
                        refined_alphas.append(np.clip(result, 0, 1))
                    else:
                        refined_alphas.append(base_alpha)
                
                return refined_alphas
        
        # 纯 AI 模式：根据用户选择的模型优先使用
        ai_alphas = None
        if self.config.use_ai:
            ai_name, ai_engine, auto_selected = self._select_sequence_ai_engine()
            if ai_engine is not None:
                self.last_active_ai_model = ai_name
                suffix = " (auto)" if auto_selected else ""
                logger.info("Using %s for green screen%s", ai_name.upper(), suffix)
                ai_alphas = self._generate_with_engine(ai_engine, frames, progress_callback, cancel_check)
                ai_name, ai_alphas = self._maybe_fallback_degenerate_gvm_sequence(
                    ai_name,
                    ai_alphas,
                    base_alphas,
                    frames,
                    progress_callback,
                    cancel_check,
                )
                self.last_active_ai_model = ai_name
        
        if ai_alphas is not None:
            semantic_subject_alphas = self._generate_green_screen_semantic_subject_alphas(
                ai_name,
                frames,
                progress_callback,
                cancel_check,
            )
            ai_alphas = self._merge_green_screen_effects(
                base_alphas,
                ai_alphas,
                frames,
                semantic_subject_alphas=semantic_subject_alphas,
            )
            # 应用 Chroma Key 后处理
            from .chroma_key_postprocess import apply_chroma_key_postprocess
            return apply_chroma_key_postprocess(ai_alphas, self.config)
        
        return base_alphas

    def _generate_green_screen_semantic_subject_alphas(
        self,
        ai_name: str | None,
        frames: List[np.ndarray],
        progress_callback=None,
        cancel_check=None,
    ) -> List[np.ndarray] | None:
        if ai_name != "gvm" or not frames:
            return None

        semantic_engine = self.birefnet
        if semantic_engine is None or getattr(semantic_engine, "model", None) is None:
            semantic_engine = self._load_green_screen_fallback_engine("birefnet")
        if semantic_engine is None or getattr(semantic_engine, "model", None) is None:
            logger.info("Semantic subject trimap skipped: BiRefNet is unavailable")
            return None

        try:
            logger.info("Building semantic subject trimap with BiRefNet for GVM green screen")
            return self._generate_with_engine(semantic_engine, frames, progress_callback, cancel_check)
        except Exception as exc:
            logger.warning("Semantic subject trimap failed, continuing without it: %s", exc)
            return None

    def _merge_green_screen_effects(
        self,
        base_alphas: List[np.ndarray],
        ai_alphas: List[np.ndarray],
        frames: List[np.ndarray] | None = None,
        semantic_subject_alphas: List[np.ndarray] | None = None,
    ) -> List[np.ndarray]:
        """Preserve transparent screen-space effects that AI subject mattes often drop."""
        preserve = float(np.clip(getattr(self.config, "transparency_preserve", 0.7), 0.0, 1.0))
        self.last_green_screen_layer_debug = None
        if preserve <= 0.0:
            return ai_alphas

        use_competitive_composer = self.last_active_ai_model == "gvm"
        composer = None
        if use_competitive_composer:
            from .green_screen_layer_composer import GreenScreenCompetitiveLayerComposer, LayerCandidate

            composer = GreenScreenCompetitiveLayerComposer()

        merged: List[np.ndarray] = []
        frame_iter = frames if frames is not None else [None] * len(base_alphas)
        semantic_iter = (
            semantic_subject_alphas
            if semantic_subject_alphas is not None
            else [None] * len(base_alphas)
        )
        for base_alpha, ai_alpha, frame, semantic_subject_alpha in zip(base_alphas, ai_alphas, frame_iter, semantic_iter):
            base_alpha = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            ai_alpha = np.clip(ai_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            semantic_subject_alpha = (
                np.clip(semantic_subject_alpha.astype(np.float32, copy=False), 0.0, 1.0)
                if semantic_subject_alpha is not None
                else None
            )
            gvm_fallback_mask = None
            apply_gvm_subject_fallback = False
            if self.last_active_ai_model == "gvm":
                gvm_fallback_mask = self._green_screen_gvm_fallback_subject_mask(base_alpha, frame)
                apply_gvm_subject_fallback = self._should_apply_gvm_subject_fallback(ai_alpha, base_alpha, gvm_fallback_mask)
                ai_alpha = self._recover_degenerate_gvm_subject_alpha(ai_alpha, base_alpha, frame)
            subject_confidence = self._green_screen_subject_confidence(ai_alpha, base_alpha, frame)
            if semantic_subject_alpha is not None:
                subject_confidence = np.maximum(
                    subject_confidence,
                    self._smoothstep(semantic_subject_alpha, 0.25, 0.75),
                )
            if (
                self.last_active_ai_model == "gvm"
                and frame is not None
                and gvm_fallback_mask is not None
                and apply_gvm_subject_fallback
            ):
                base_support = self._smoothstep(base_alpha, 0.30, 0.72)
                subject_confidence = np.where(
                    gvm_fallback_mask,
                    np.maximum(subject_confidence, np.clip(0.25 + 0.90 * base_support, 0.0, 0.98)),
                    subject_confidence,
                )
            subject_gate = self._smoothstep(subject_confidence, 0.45, 0.80)
            solid_alpha = np.maximum(
                self._green_screen_ai_subject_layer(ai_alpha, base_alpha, frame, subject_gate),
                self._green_screen_solid_layer(base_alpha, frame),
            )
            solid_alpha = np.maximum(
                solid_alpha,
                self._green_screen_score_blocked_subject_layer(base_alpha, frame),
            )
            solid_alpha = np.maximum(
                solid_alpha,
                self._green_screen_subject_integrity_layer(ai_alpha, base_alpha, frame, subject_gate),
            )
            solid_alpha = np.maximum(
                solid_alpha,
                self._green_screen_semantic_subject_layer(semantic_subject_alpha, frame),
            )
            effect_alpha = self._green_screen_effect_layer(base_alpha, frame) * preserve
            effect_alpha = effect_alpha * (1.0 - 0.85 * subject_gate)
            effect_alpha = np.maximum(
                effect_alpha,
                self._green_screen_luminous_effect_reconstruction_layer(base_alpha, frame) * preserve,
            )
            if composer is not None:
                subject_competitive_confidence = np.maximum(
                    subject_gate,
                    self._smoothstep(solid_alpha, 0.01, 0.35),
                )
                effect_competitive_confidence = np.maximum(
                    effect_alpha,
                    self._smoothstep(effect_alpha, 0.0, 0.04),
                )
                composed = composer.compose(
                    subject=LayerCandidate(
                        alpha=solid_alpha,
                        confidence=subject_competitive_confidence,
                        evidence=subject_confidence,
                    ),
                    effect=LayerCandidate(
                        alpha=effect_alpha,
                        confidence=effect_competitive_confidence,
                        evidence=effect_alpha,
                    ),
                )
                fused_alpha = composed.final_alpha
                self.last_green_screen_layer_debug = composed.debug_layers
            else:
                fused_alpha = self._soft_fuse_layers(solid_alpha, effect_alpha)
            self._log_transparency_fusion_stats("green_screen", solid_alpha, effect_alpha, fused_alpha)
            merged.append(fused_alpha)
        return merged

    def _smoothstep(self, x: np.ndarray, low: float, high: float) -> np.ndarray:
        t = np.clip((x - low) / max(high - low, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _soft_fuse_layers(self, solid_alpha: np.ndarray, effect_alpha: np.ndarray) -> np.ndarray:
        solid = np.clip(solid_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        effect = np.clip(effect_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        return np.clip(solid + effect * (1.0 - solid), 0.0, 1.0)

    def _log_transparency_fusion_stats(
        self,
        mode: str,
        solid_alpha: np.ndarray,
        effect_alpha: np.ndarray,
        fused_alpha: np.ndarray,
    ) -> None:
        logger.info(
            "Transparency fusion stats: mode=%s solid_mean=%.4f effect_mean=%.4f fused_mean=%.4f",
            mode,
            float(solid_alpha.mean()),
            float(effect_alpha.mean()),
            float(fused_alpha.mean()),
        )

    def _green_screen_effect_layer(self, base_alpha: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
        background_floor = float(np.percentile(base_alpha, 10))
        effect_floor = min(background_floor + 0.02, 0.95)
        normalized = np.clip((base_alpha - effect_floor) / max(1.0 - effect_floor, 1e-6), 0.0, 1.0)
        effect_alpha = self._smoothstep(normalized, 0.08, 0.75)
        if frame is not None:
            effect_alpha = effect_alpha * self._green_screen_effect_haze_suppression(frame)
            effect_alpha = effect_alpha * self._green_screen_effect_color_weight(frame)
            effect_alpha = np.maximum(effect_alpha, self._green_screen_white_ring_boost(base_alpha, frame))
        return np.clip(effect_alpha, 0.0, 1.0)

    def _green_screen_luminous_effect_reconstruction_layer(
        self,
        base_alpha: np.ndarray,
        frame: np.ndarray | None,
    ) -> np.ndarray:
        """Reconstruct bright lightning-like effect structures independently from subject alpha."""
        if frame is None:
            return np.zeros_like(base_alpha, dtype=np.float32)

        frame_f = frame.astype(np.float32, copy=False)
        red = frame_f[:, :, 0]
        green = frame_f[:, :, 1]
        blue = frame_f[:, :, 2]
        brightness = (red + green + blue) / 3.0
        chroma = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])

        screen_green = (green > red + 30.0) & (green > blue + 20.0) & (green > 90.0)
        white_core = (brightness > 205.0) & (chroma < 70.0)
        blue_white_core = (brightness > 185.0) & (blue > 150.0) & (green > 140.0) & (red > 130.0) & (chroma < 95.0)
        yellow_white_core = (red > 200.0) & (green > 155.0) & (blue > 90.0) & (brightness > 165.0)
        core_mask = (white_core | blue_white_core | yellow_white_core) & (~screen_green)

        component_count, labels = cv2.connectedComponents(core_mask.astype(np.uint8), connectivity=8)
        if component_count <= 1:
            return np.zeros_like(base_alpha, dtype=np.float32)

        component_sizes = np.bincount(labels.ravel())
        keep_labels = np.where(component_sizes >= 12)[0]
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size == 0:
            return np.zeros_like(base_alpha, dtype=np.float32)

        kept_core = np.isin(labels, keep_labels).astype(np.uint8)
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kept_core = cv2.morphologyEx(kept_core, cv2.MORPH_CLOSE, bridge_kernel, iterations=1)

        halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        halo = cv2.dilate(kept_core, halo_kernel, iterations=1).astype(bool)
        halo_color = (
            ((brightness > 145.0) & (chroma < 120.0))
            | ((blue > 135.0) & (green > 105.0) & (red > 95.0))
            | ((red > 185.0) & (green > 120.0))
        )
        halo = halo & halo_color & (~screen_green)

        core_alpha = kept_core.astype(np.float32)
        halo_alpha = 0.34 * halo.astype(np.float32) * self._smoothstep(brightness / 255.0, 0.42, 0.82)
        cyan_halo_alpha = self._green_screen_cyan_halo_band_layer(
            kept_core.astype(bool),
            base_alpha,
            red,
            green,
            blue,
            brightness,
            screen_green,
        )
        return np.clip(np.maximum.reduce([core_alpha, halo_alpha, cyan_halo_alpha]), 0.0, 1.0).astype(np.float32, copy=False)

    def _green_screen_cyan_halo_band_layer(
        self,
        core_mask: np.ndarray,
        base_alpha: np.ndarray,
        red: np.ndarray,
        green: np.ndarray,
        blue: np.ndarray,
        brightness: np.ndarray,
        screen_green: np.ndarray,
    ) -> np.ndarray:
        if not np.any(core_mask):
            return np.zeros_like(base_alpha, dtype=np.float32)

        core_reach = cv2.dilate(
            core_mask.astype(np.uint8, copy=False),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)),
            iterations=1,
        ).astype(bool)
        cyan_halo_candidate = (
            (blue > 140.0)
            & (green > 120.0)
            & (red < 150.0)
            & (brightness > 120.0)
            & (base_alpha < 0.45)
            & (~screen_green)
        )
        near_core_halo = core_reach & (~core_mask) & cyan_halo_candidate
        if not np.any(near_core_halo):
            return np.zeros_like(base_alpha, dtype=np.float32)

        distance_to_core = cv2.distanceTransform((~core_mask).astype(np.uint8), cv2.DIST_L2, 3)
        distance_weight = np.clip((28.0 - distance_to_core) / 28.0, 0.0, 1.0)
        brightness_weight = self._smoothstep(brightness / 255.0, 0.42, 0.72)
        halo_alpha = (0.34 + 0.18 * distance_weight) * brightness_weight
        return np.where(near_core_halo, halo_alpha, 0.0).astype(np.float32, copy=False)

    def _green_screen_solid_layer(self, base_alpha: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
        if frame is None:
            return np.where(base_alpha >= 0.92, base_alpha, 0.0)

        non_screen_mask = self._green_screen_non_screen_mask(frame)
        solid_color_mask = self._green_screen_solid_foreground_mask(frame)
        soft_subject_mask = self._green_screen_soft_subject_mask(frame)
        solid_mask = (
            ((base_alpha >= 0.94) & non_screen_mask)
            | ((base_alpha >= 0.92) & solid_color_mask)
            | ((base_alpha >= 0.28) & soft_subject_mask)
        )
        return np.where(solid_mask, 1.0, 0.0)

    def _green_screen_score_blocked_subject_layer(
        self,
        base_alpha: np.ndarray,
        frame: np.ndarray | None,
    ) -> np.ndarray:
        """Conservatively rescue soft subjects when GVM stays primary only because fallback quality is too weak."""
        if frame is None or self.last_active_ai_model != "gvm":
            return np.zeros_like(base_alpha, dtype=np.float32)

        metrics = self.last_fallback_quality_metrics or {}
        if not metrics.get("score_blocked") or metrics.get("effect_damage_blocked"):
            return np.zeros_like(base_alpha, dtype=np.float32)

        non_screen_mask = self._green_screen_non_screen_mask(frame)
        soft_subject_mask = self._green_screen_soft_subject_mask(frame)
        effect_like = self._green_screen_effect_color_weight(frame)
        frame_f = frame.astype(np.float32, copy=False)
        brightness = frame_f.mean(axis=2)
        chroma = np.maximum.reduce(frame_f, axis=2) - np.minimum.reduce(frame_f, axis=2)
        blue = frame_f[:, :, 2]
        green = frame_f[:, :, 1]
        red = frame_f[:, :, 0]
        pale_halo_mask = (brightness >= 184.0) & (chroma <= 18.0) & (base_alpha <= 0.24)
        cool_highlight_mask = (
            (brightness >= 190.0)
            & ((blue - green) >= 30.0)
            & ((blue - red) >= 50.0)
            & (base_alpha <= 0.30)
        )
        cool_gray_transition_mask = (
            (brightness >= 192.0)
            & (chroma <= 48.0)
            & ((blue - green) >= 35.0)
            & ((blue - red) >= 40.0)
            & (effect_like >= 0.10)
            & (base_alpha <= 0.30)
        )
        base_support = self._smoothstep(base_alpha, 0.18, 0.38)
        solid_support = self._green_screen_solid_layer(base_alpha, frame)
        rescue_mask = (
            non_screen_mask
            & soft_subject_mask
            & (base_support > 0.10)
            & (solid_support <= 0.0)
            & (effect_like < 0.25)
            & (~pale_halo_mask)
            & (~cool_highlight_mask)
            & (~cool_gray_transition_mask)
        )
        rescue_alpha = np.clip(base_alpha + 0.16 + 0.20 * base_support, 0.0, 0.72)
        return np.where(rescue_mask, rescue_alpha, 0.0).astype(np.float32, copy=False)

    def _green_screen_subject_integrity_layer(
        self,
        ai_alpha: np.ndarray,
        base_alpha: np.ndarray,
        frame: np.ndarray | None,
        subject_gate: np.ndarray,
    ) -> np.ndarray:
        """Fill non-screen subject holes that are connected to confident subject structure."""
        if frame is None or self.last_active_ai_model != "gvm":
            return np.zeros_like(base_alpha, dtype=np.float32)
        metrics = self.last_fallback_quality_metrics or {}
        if metrics.get("effect_damage_blocked"):
            return np.zeros_like(base_alpha, dtype=np.float32)

        non_screen_mask = self._green_screen_non_screen_mask(frame)
        solid_color_mask = self._green_screen_solid_foreground_mask(frame)
        effect_like = self._green_screen_effect_color_weight(frame)
        frame_f = frame.astype(np.float32, copy=False)
        brightness = frame_f.mean(axis=2)
        chroma = np.maximum.reduce(frame_f, axis=2) - np.minimum.reduce(frame_f, axis=2)
        blue = frame_f[:, :, 2]
        green = frame_f[:, :, 1]
        red = frame_f[:, :, 0]

        effect_veto = (
            ((effect_like >= 0.70) & (chroma <= 90.0) & (base_alpha <= 0.45))
            | ((brightness >= 190.0) & ((blue - green) >= 30.0) & ((blue - red) >= 50.0) & (base_alpha <= 0.35))
            | ((brightness >= 192.0) & (chroma <= 48.0) & ((blue - green) >= 35.0) & ((blue - red) >= 40.0) & (base_alpha <= 0.35))
        )
        color_coherent_subject = (base_alpha >= 0.20) | ((red >= 120.0) & (green < 180.0))
        anchor_seed = (
            non_screen_mask
            & (~effect_veto)
            & ((ai_alpha >= 0.75) | (subject_gate >= 0.75))
        )
        anchor_reach = cv2.dilate(
            anchor_seed.astype(np.uint8, copy=False),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)),
            iterations=1,
        ).astype(bool)
        seed_mask = (
            anchor_seed
            | (
                anchor_reach
                & non_screen_mask
                & solid_color_mask
                & color_coherent_subject
                & (~effect_veto)
            )
        )
        subject_instance_mask = self._green_screen_subject_instance_mask(seed_mask)
        candidate_mask = (
            (subject_instance_mask | (anchor_reach & color_coherent_subject))
            & non_screen_mask
            & (base_alpha <= 0.55)
            & color_coherent_subject
            & (~effect_veto)
        )
        if not np.any(candidate_mask) or not np.any(seed_mask):
            return np.zeros_like(base_alpha, dtype=np.float32)

        component_labels, labels = cv2.connectedComponents(candidate_mask.astype(np.uint8), connectivity=8)
        if component_labels <= 1:
            return np.zeros_like(base_alpha, dtype=np.float32)

        seed_reach = cv2.dilate(seed_mask.astype(np.uint8), np.ones((7, 7), dtype=np.uint8), iterations=2).astype(bool)
        connected_labels = np.unique(labels[seed_reach & candidate_mask])
        connected_labels = connected_labels[connected_labels != 0]
        if connected_labels.size == 0:
            return np.zeros_like(base_alpha, dtype=np.float32)

        component_sizes = np.bincount(labels.ravel())
        connected_labels = connected_labels[component_sizes[connected_labels] >= 64]
        if connected_labels.size == 0:
            return np.zeros_like(base_alpha, dtype=np.float32)

        connected_subject = np.isin(labels, connected_labels)
        base_support = self._smoothstep(base_alpha, 0.03, 0.55)
        recovery_alpha = np.clip(0.58 + 0.18 * base_support + 0.18 * subject_gate, 0.0, 0.82)
        return np.where(connected_subject, recovery_alpha, 0.0).astype(np.float32, copy=False)

    def _green_screen_semantic_subject_layer(
        self,
        semantic_subject_alpha: np.ndarray | None,
        frame: np.ndarray | None,
    ) -> np.ndarray:
        if semantic_subject_alpha is None:
            if frame is None:
                return np.array(0.0, dtype=np.float32)
            return np.zeros(frame.shape[:2], dtype=np.float32)

        semantic_gate = self._smoothstep(semantic_subject_alpha, 0.25, 0.75)
        semantic_subject = np.clip(0.82 * semantic_gate, 0.0, 0.92)
        if frame is None:
            return semantic_subject.astype(np.float32, copy=False)

        non_screen_mask = self._green_screen_non_screen_mask(frame)
        return np.where(non_screen_mask, semantic_subject, 0.0).astype(np.float32, copy=False)

    def _green_screen_subject_instance_mask(self, seed_mask: np.ndarray) -> np.ndarray:
        seed_uint = seed_mask.astype(np.uint8, copy=False)
        if not np.any(seed_uint):
            return np.zeros_like(seed_mask, dtype=bool)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
        instance_uint = cv2.morphologyEx(seed_uint, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        instance_uint = cv2.dilate(instance_uint, dilate_kernel, iterations=1)

        component_count, labels = cv2.connectedComponents(instance_uint, connectivity=8)
        if component_count <= 1:
            return instance_uint.astype(bool)

        component_sizes = np.bincount(labels.ravel())
        keep_labels = np.where(component_sizes >= 512)[0]
        keep_labels = keep_labels[keep_labels != 0]
        if keep_labels.size == 0:
            return np.zeros_like(seed_mask, dtype=bool)

        instance_mask = np.isin(labels, keep_labels).astype(np.uint8)
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(instance_mask)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        return filled.astype(bool)

    def _green_screen_ai_solid_layer(
        self,
        ai_alpha: np.ndarray,
        base_alpha: np.ndarray,
        frame: np.ndarray | None,
    ) -> np.ndarray:
        if frame is None:
            return np.where((ai_alpha >= 0.98) & (base_alpha >= 0.75), ai_alpha, 0.0)

        non_screen_mask = self._green_screen_non_screen_mask(frame)
        solid_color_mask = self._green_screen_solid_foreground_mask(frame)
        soft_subject_mask = self._green_screen_soft_subject_mask(frame)
        effect_like = self._green_screen_effect_color_weight(frame) > 0.45
        ai_solid_mask = (
            (ai_alpha >= 0.98)
            & (
                (base_alpha >= 0.75)
                | ((base_alpha >= 0.90) & non_screen_mask)
                | (soft_subject_mask & ((~effect_like) | (base_alpha >= 0.32)))
                | (solid_color_mask & (~effect_like) & (base_alpha >= 0.55))
            )
        )
        return np.where(ai_solid_mask, ai_alpha, 0.0)

    def _green_screen_ai_subject_layer(
        self,
        ai_alpha: np.ndarray,
        base_alpha: np.ndarray,
        frame: np.ndarray | None,
        subject_gate: np.ndarray,
    ) -> np.ndarray:
        """Promote AI subject alpha when confidence is high, but avoid white-ring takeover."""
        confident_subject = ai_alpha * np.clip(subject_gate.astype(np.float32, copy=False), 0.0, 1.0)
        return np.maximum(
            self._green_screen_ai_solid_layer(ai_alpha, base_alpha, frame),
            confident_subject,
        )

    def _green_screen_subject_confidence(
        self,
        ai_alpha: np.ndarray,
        base_alpha: np.ndarray,
        frame: np.ndarray | None,
    ) -> np.ndarray:
        """Estimate whether the AI alpha is a real subject region or just a transparent effect."""
        ai_alpha = np.clip(ai_alpha.astype(np.float32, copy=False), 0.0, 1.0)
        base_alpha = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)

        ai_support = self._smoothstep(ai_alpha, 0.45, 0.92)
        base_support = self._smoothstep(base_alpha, 0.08, 0.42)
        confidence = 0.08 + 0.58 * ai_support + 0.18 * base_support

        if frame is None:
            return np.clip(confidence, 0.0, 1.0)

        non_screen_mask = self._green_screen_non_screen_mask(frame).astype(np.float32)
        solid_color_mask = self._green_screen_solid_foreground_mask(frame).astype(np.float32)
        soft_subject_mask = self._green_screen_soft_subject_mask(frame).astype(np.float32)
        effect_like = self._green_screen_effect_color_weight(frame)
        subject_structure = np.maximum(solid_color_mask, soft_subject_mask) * base_support

        confidence += 0.10 * non_screen_mask
        confidence += 0.10 * np.maximum(solid_color_mask, soft_subject_mask)

        # White rings and soft glows can have high AI alpha, but should stay in the effect branch
        # when the traditional matte only provides weak structural support.
        confidence *= 1.0 - 0.90 * effect_like * (1.0 - 0.75 * base_support) * (1.0 - 0.85 * subject_structure)

        return np.clip(confidence, 0.0, 1.0)

    def _maybe_fallback_degenerate_gvm_sequence(
        self,
        ai_name: str,
        ai_alphas: List[np.ndarray] | None,
        base_alphas: List[np.ndarray],
        frames: List[np.ndarray],
        progress_callback=None,
        cancel_check=None,
    ) -> tuple[str, List[np.ndarray] | None]:
        self.last_fallback_quality_metrics = None
        if ai_name != "gvm" or not ai_alphas or not self._is_degenerate_gvm_sequence(ai_alphas, base_alphas, frames):
            return ai_name, ai_alphas

        fallback_name, fallback_engine = self._select_green_screen_fallback_ai_engine(exclude_name="gvm")
        if fallback_engine is None:
            return ai_name, ai_alphas

        logger.info("Detected degenerate GVM sequence, falling back to %s", fallback_name.upper())
        fallback_alphas = self._generate_with_engine(fallback_engine, frames, progress_callback, cancel_check)
        if not self._is_materially_better_fallback(ai_alphas, fallback_alphas, base_alphas, frames):
            logger.info("Fallback engine %s is not materially better than GVM, keeping GVM", fallback_name.upper())
            return ai_name, ai_alphas
        return fallback_name, fallback_alphas

    def _is_degenerate_gvm_sequence(
        self,
        ai_alphas: List[np.ndarray],
        base_alphas: List[np.ndarray],
        frames: List[np.ndarray],
    ) -> bool:
        if not ai_alphas or not frames or len(ai_alphas) != len(base_alphas) or len(ai_alphas) != len(frames):
            return False

        fallback_hits = 0
        broad_support_hits = 0
        for ai_alpha, base_alpha, frame in zip(ai_alphas, base_alphas, frames):
            subject_mask = self._green_screen_gvm_fallback_subject_mask(base_alpha, frame)
            if self._should_apply_gvm_subject_fallback(ai_alpha, base_alpha, subject_mask):
                fallback_hits += 1
                continue
            if self._is_broad_degenerate_gvm_frame(ai_alpha, base_alpha, frame):
                broad_support_hits += 1

        minimum_hits = max(2, (len(frames) + 1) // 2)
        return fallback_hits >= minimum_hits or broad_support_hits >= minimum_hits

    @staticmethod
    def _masked_mean(alpha: np.ndarray, mask: np.ndarray) -> float | None:
        return float(alpha[mask].mean()) if mask.any() else None

    def _build_region_weighted_fallback_masks(
        self,
        source_alpha: np.ndarray,
        fallback_alpha: np.ndarray,
        base_alpha: np.ndarray,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference_alpha = np.maximum(
            np.clip(np.asarray(source_alpha, dtype=np.float32), 0.0, 1.0),
            np.clip(np.asarray(fallback_alpha, dtype=np.float32), 0.0, 1.0),
        )
        subject_confidence = self._green_screen_subject_confidence(reference_alpha, base_alpha, frame)
        subject_gate = self._smoothstep(subject_confidence, 0.45, 0.80)
        entity_mask = subject_gate >= 0.55
        effect_mask = (self._green_screen_effect_color_weight(frame) >= 0.35) & (subject_gate <= 0.35)
        transition_mask = (~entity_mask) & (~effect_mask) & (base_alpha > 0.02)
        return entity_mask, effect_mask, transition_mask

    def _compute_region_weighted_fallback_quality(
        self,
        source_alphas: List[np.ndarray],
        fallback_alphas: List[np.ndarray],
        base_alphas: List[np.ndarray],
        frames: List[np.ndarray],
    ) -> dict[str, float] | None:
        if (
            not source_alphas
            or not fallback_alphas
            or not base_alphas
            or not frames
            or len(source_alphas) != len(fallback_alphas)
            or len(source_alphas) != len(base_alphas)
            or len(source_alphas) != len(frames)
        ):
            return None

        entity_deltas: List[float] = []
        effect_deltas: List[float] = []
        transition_deltas: List[float] = []
        source_means: List[float] = []
        fallback_means: List[float] = []

        for source_alpha, fallback_alpha, base_alpha, frame in zip(source_alphas, fallback_alphas, base_alphas, frames):
            source = np.clip(np.asarray(source_alpha, dtype=np.float32), 0.0, 1.0)
            fallback = np.clip(np.asarray(fallback_alpha, dtype=np.float32), 0.0, 1.0)
            base = np.clip(np.asarray(base_alpha, dtype=np.float32), 0.0, 1.0)
            entity_mask, effect_mask, transition_mask = self._build_region_weighted_fallback_masks(source, fallback, base, frame)

            source_means.append(float(source.mean()))
            fallback_means.append(float(fallback.mean()))

            source_entity_mean = self._masked_mean(source, entity_mask)
            fallback_entity_mean = self._masked_mean(fallback, entity_mask)
            source_effect_mean = self._masked_mean(source, effect_mask)
            fallback_effect_mean = self._masked_mean(fallback, effect_mask)
            source_transition_mean = self._masked_mean(source, transition_mask)
            fallback_transition_mean = self._masked_mean(fallback, transition_mask)

            entity_deltas.append((fallback_entity_mean or 0.0) - (source_entity_mean or 0.0))
            effect_deltas.append((fallback_effect_mean or 0.0) - (source_effect_mean or 0.0))
            transition_deltas.append((fallback_transition_mean or 0.0) - (source_transition_mean or 0.0))

        entity_delta = float(np.mean(entity_deltas)) if entity_deltas else 0.0
        effect_delta = float(np.mean(effect_deltas)) if effect_deltas else 0.0
        transition_delta = float(np.mean(transition_deltas)) if transition_deltas else 0.0
        global_mean_delta = float(np.mean(fallback_means) - np.mean(source_means)) if source_means and fallback_means else 0.0
        weighted_score = 0.65 * entity_delta + 0.20 * transition_delta + 0.15 * effect_delta

        return {
            "entity_delta": entity_delta,
            "effect_delta": effect_delta,
            "transition_delta": transition_delta,
            "global_mean_delta": global_mean_delta,
            "weighted_score": weighted_score,
        }

    def _is_materially_better_fallback(
        self,
        source_alphas: List[np.ndarray],
        fallback_alphas: List[np.ndarray],
        base_alphas: List[np.ndarray],
        frames: List[np.ndarray],
    ) -> bool:
        metrics = self._compute_region_weighted_fallback_quality(source_alphas, fallback_alphas, base_alphas, frames)
        if metrics is None:
            self.last_fallback_quality_metrics = None
            return False

        effect_delta = metrics["effect_delta"]
        weighted_score = metrics["weighted_score"]
        effect_damage_blocked = effect_delta < -0.03
        score_blocked = weighted_score < 0.02
        self.last_fallback_quality_metrics = {
            **metrics,
            "effect_damage_blocked": effect_damage_blocked,
            "score_blocked": score_blocked,
            "accepted": (not effect_damage_blocked) and (not score_blocked),
        }
        if effect_damage_blocked:
            logger.info(
                "Rejecting fallback by quality gate: effect_delta=%.4f weighted_score=%.4f global_mean_delta=%.4f",
                effect_delta,
                weighted_score,
                metrics["global_mean_delta"],
            )
            return False
        if score_blocked:
            logger.info(
                "Rejecting fallback by quality gate: weighted_score=%.4f entity_delta=%.4f effect_delta=%.4f transition_delta=%.4f global_mean_delta=%.4f",
                weighted_score,
                metrics["entity_delta"],
                effect_delta,
                metrics["transition_delta"],
                metrics["global_mean_delta"],
            )
            return False

        logger.info(
            "Accepting fallback by quality gate: weighted_score=%.4f entity_delta=%.4f effect_delta=%.4f transition_delta=%.4f global_mean_delta=%.4f",
            weighted_score,
            metrics["entity_delta"],
            effect_delta,
            metrics["transition_delta"],
            metrics["global_mean_delta"],
        )
        return True

    def _is_broad_degenerate_gvm_frame(
        self,
        ai_alpha: np.ndarray,
        base_alpha: np.ndarray,
        frame: np.ndarray,
    ) -> bool:
        ai_mean = float(np.mean(np.clip(ai_alpha, 0.0, 1.0)))
        if ai_mean >= 0.06:
            return False

        non_screen_mask = self._green_screen_non_screen_mask(frame)
        base_support = self._smoothstep(base_alpha, 0.20, 0.50)
        broad_support_mask = non_screen_mask & (base_support > 0.30)
        broad_support_ratio = float(np.mean(broad_support_mask))
        if broad_support_ratio < 0.18:
            return False

        transition_ratio = float(np.mean((base_alpha > 0.08) & (base_alpha < 0.75)))
        effect_like_mean = float(np.mean(self._green_screen_effect_color_weight(frame)[broad_support_mask]))
        return transition_ratio >= 0.20 and effect_like_mean < 0.65

    def _select_green_screen_fallback_ai_engine(self, exclude_name: str) -> tuple[str | None, object | None]:
        fallback_candidates = (
            ("corridorkey", self.corridorkey, "model"),
            ("rembg", self.rembg, "_available"),
            ("birefnet", self.birefnet, "model"),
            ("rvm", self.rvm, "model"),
        )
        for name, engine, attr in fallback_candidates:
            if name == exclude_name:
                continue
            if engine is None:
                engine = self._load_green_screen_fallback_engine(name)
                if engine is None:
                    continue
            if attr == "_available":
                if getattr(engine, "_available", False):
                    return name, engine
                continue
            if getattr(engine, attr, None) is not None:
                return name, engine
        return None, None

    def _load_green_screen_fallback_engine(self, name: str):
        loader_name = f"_load_{name}"
        loader = getattr(self, loader_name, None)
        if loader is None:
            return getattr(self, name, None)
        try:
            loader(self.config)
        except Exception as exc:
            logger.warning("Failed to lazy-load fallback engine %s: %s", name, exc)
        return getattr(self, name, None)

    def _recover_degenerate_gvm_subject_alpha(
        self,
        ai_alpha: np.ndarray,
        base_alpha: np.ndarray,
        frame: np.ndarray | None,
    ) -> np.ndarray:
        """Recover subject support from the chroma-key base when GVM collapses on a frame."""
        if frame is None:
            return ai_alpha

        subject_mask = self._green_screen_gvm_fallback_subject_mask(base_alpha, frame)
        if not self._should_apply_gvm_subject_fallback(ai_alpha, base_alpha, subject_mask):
            return ai_alpha

        recovered = np.where(subject_mask, np.maximum(ai_alpha, np.clip(base_alpha * 0.98, 0.0, 1.0)), ai_alpha)
        return recovered.astype(np.float32, copy=False)

    def _green_screen_gvm_fallback_subject_mask(
        self,
        base_alpha: np.ndarray,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Identify subject-like regions that can safely fall back to the traditional matte."""
        non_screen_mask = self._green_screen_non_screen_mask(frame)
        solid_color_mask = self._green_screen_solid_foreground_mask(frame)
        soft_subject_mask = self._green_screen_soft_subject_mask(frame)
        effect_like = self._green_screen_effect_color_weight(frame)

        reliable_base = self._smoothstep(base_alpha, 0.30, 0.72) > 0.35
        return (
            non_screen_mask
            & reliable_base
            & (solid_color_mask | soft_subject_mask | (base_alpha >= 0.50))
            & ((effect_like < 0.35) | ((solid_color_mask | soft_subject_mask) & (base_alpha >= 0.45)))
        )

    def _should_apply_gvm_subject_fallback(
        self,
        ai_alpha: np.ndarray,
        base_alpha: np.ndarray,
        subject_mask: np.ndarray,
    ) -> bool:
        if not np.any(subject_mask):
            return False

        ai_support = self._smoothstep(ai_alpha, 0.25, 0.70)
        base_support = self._smoothstep(base_alpha, 0.30, 0.72)
        ai_subject_strength = float(ai_support[subject_mask].mean())
        base_subject_strength = float(base_support[subject_mask].mean())
        return ai_subject_strength < 0.08 and base_subject_strength >= 0.35

    def _green_screen_non_screen_mask(self, frame: np.ndarray) -> np.ndarray:
        """Detect colors that are clearly not the backing screen for solid-subject fallback."""
        frame_f = frame.astype(np.float32, copy=False)
        r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
        screen_green = (g > r + 30.0) & (g > b + 20.0) & (g > 90.0)
        return ~screen_green

    def _green_screen_effect_color_weight(self, frame: np.ndarray) -> np.ndarray:
        """Keep pink/white glow while suppressing green-screen colored haze blobs."""
        frame_f = frame.astype(np.float32, copy=False)
        r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
        brightness = (r + g + b) / 3.0
        chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

        pink_score = np.clip((r - g - 20.0) / 45.0, 0.0, 1.0)
        pink_score *= np.clip((b - g + 5.0) / 45.0, 0.0, 1.0)
        white_glow_score = np.clip((brightness - 185.0) / 50.0, 0.0, 1.0)
        white_glow_score *= np.clip((90.0 - chroma) / 90.0, 0.0, 1.0)

        green_haze_score = np.minimum.reduce(
            [
                np.clip((g - b - 10.0) / 18.0, 0.0, 1.0),
                np.clip((g - r + 2.0) / 14.0, 0.0, 1.0),
                np.clip((brightness - 165.0) / 40.0, 0.0, 1.0),
                np.clip((85.0 - chroma) / 85.0, 0.0, 1.0),
            ]
        )
        white_glow_score *= 1.0 - 0.85 * green_haze_score

        return np.maximum(pink_score, white_glow_score)

    def _green_screen_effect_haze_suppression(self, frame: np.ndarray) -> np.ndarray:
        """Attenuate bright low-chroma green haze before effect alpha is fused."""
        frame_f = frame.astype(np.float32, copy=False)
        r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
        brightness = (r + g + b) / 3.0
        chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

        green_haze = np.minimum.reduce(
            [
                np.clip((g - b - 8.0) / 18.0, 0.0, 1.0),
                np.clip((g - r + 4.0) / 18.0, 0.0, 1.0),
                np.clip((brightness - 150.0) / 55.0, 0.0, 1.0),
                np.clip((95.0 - chroma) / 95.0, 0.0, 1.0),
            ]
        )
        bright_haze_bias = np.maximum(
            np.clip((brightness - 175.0) / 55.0, 0.0, 1.0),
            np.clip((210.0 - chroma) / 210.0, 0.0, 1.0),
        )
        suppression = 1.0 - green_haze * (0.55 + 0.20 * bright_haze_bias)
        return np.clip(suppression, 0.2, 1.0)

    def _green_screen_white_ring_boost(self, base_alpha: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Restore bright white rings whose mixed green edge gets under-weighted."""
        frame_f = frame.astype(np.float32, copy=False)
        r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
        brightness = (r + g + b) / 3.0
        chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        screen_mix = np.clip((g - np.maximum(r, b)) / 55.0, 0.0, 1.0)
        ring_mix = np.minimum.reduce(
            [
                np.clip((brightness - 175.0) / 45.0, 0.0, 1.0),
                np.clip((110.0 - chroma) / 110.0, 0.0, 1.0),
                screen_mix,
                np.clip(base_alpha / 0.28, 0.0, 1.0),
            ]
        )
        return 0.35 * ring_mix

    def _green_screen_solid_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Reject green-screen colored false positives while keeping solid subjects/effects."""
        frame_f = frame.astype(np.float32, copy=False)
        r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
        brightness = (r + g + b) / 3.0
        chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        screen_green = (g > r + 30.0) & (g > b + 20.0) & (g > 90.0)
        bright_soft_subject = (brightness > 155.0) & (chroma < 90.0)
        saturated_pink_subject = (r > g + 45.0) & (b > g + 15.0) & (brightness > 135.0)
        return (~screen_green) & (bright_soft_subject | saturated_pink_subject)

    def _green_screen_soft_subject_mask(self, frame: np.ndarray) -> np.ndarray:
        """Detect low-saturation bright subject colors such as the rabbit ears."""
        frame_f = frame.astype(np.float32, copy=False)
        r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
        brightness = (r + g + b) / 3.0
        chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        screen_green = (g > r + 30.0) & (g > b + 20.0) & (g > 90.0)
        neutral_soft = (np.abs(r - g) < 45.0) & (np.abs(b - g) < 45.0)
        soft_bright = (brightness > 165.0) & (chroma < 75.0)
        soft_midtone = (
            (brightness > 150.0)
            & (brightness < 190.0)
            & (chroma < 45.0)
            & (b >= r - 5.0)
        )
        return (~screen_green) & neutral_soft & (soft_bright | soft_midtone)
    
    def _black_background_matte(self, frames: List[np.ndarray], progress_callback, cancel_check=None) -> List[np.ndarray]:
        """
        黑底抠图 - 传统算法 + AI 边缘增强
        
        策略：
        1. 传统算法生成基础 alpha（保留暗部粒子/辉光）
        2. AI 辅助边缘细化（如果有 RVM）
        3. 融合两者优势
        """
        logger.info("Using hybrid black background matting for %s frames", len(frames))
        
        # 1. 传统算法生成基础 alpha
        base_alphas = []
        for i, frame in enumerate(frames):
            self._check_cancelled(cancel_check)
            alpha = self.black_matte.generate(frame)
            base_alphas.append(alpha)
            if progress_callback and i % max(1, len(frames) // 20) == 0:
                progress_callback(i, len(frames))
        
        # 2. 如果有 RVM，用 AI 做边缘细化
        if self.rvm is not None and self.rvm.model is not None:
            logger.info("Applying RVM edge refinement for black background")
            ai_alphas = []
            for frame in frames:
                self._check_cancelled(cancel_check)
                ai_alphas.append(self.rvm.generate(frame))
            return self._merge_black_background_effects(base_alphas, ai_alphas, frames)
        
        return base_alphas

    def _merge_black_background_effects(
        self,
        base_alphas: List[np.ndarray],
        ai_alphas: List[np.ndarray],
        frames: List[np.ndarray] | None = None,
    ) -> List[np.ndarray]:
        merged: List[np.ndarray] = []
        frame_iter = frames if frames is not None else [None] * len(base_alphas)
        for base_alpha, ai_alpha, frame in zip(base_alphas, ai_alphas, frame_iter):
            base_alpha = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            ai_alpha = np.clip(ai_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            solid_alpha = np.maximum(ai_alpha, np.where(base_alpha > 0.75, base_alpha, 0.0))
            effect_alpha = self._black_background_effect_layer(base_alpha, frame)
            fused_alpha = self._soft_fuse_layers(solid_alpha, effect_alpha)
            self._log_transparency_fusion_stats("black_background", solid_alpha, effect_alpha, fused_alpha)
            merged.append(fused_alpha)
        return merged

    def _black_background_effect_layer(self, base_alpha: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
        effect_alpha = self._smoothstep(base_alpha, 0.03, 0.45)
        if frame is None:
            return np.clip(effect_alpha, 0.0, 1.0)

        frame_f = frame.astype(np.float32, copy=False)
        brightness = frame_f.mean(axis=2)
        chroma = frame_f.max(axis=2) - frame_f.min(axis=2)
        glow_weight = self._smoothstep(brightness / 255.0, 0.18, 0.95)
        particle_weight = self._smoothstep(chroma / 255.0, 0.05, 0.55)
        return np.clip(effect_alpha * np.maximum(glow_weight, particle_weight), 0.0, 1.0)
