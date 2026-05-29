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
        
        if ai_alphas is not None:
            ai_alphas = self._merge_green_screen_effects(base_alphas, ai_alphas, frames)
            # 应用 Chroma Key 后处理
            from .chroma_key_postprocess import apply_chroma_key_postprocess
            return apply_chroma_key_postprocess(ai_alphas, self.config)
        
        return base_alphas

    def _merge_green_screen_effects(
        self,
        base_alphas: List[np.ndarray],
        ai_alphas: List[np.ndarray],
        frames: List[np.ndarray] | None = None,
    ) -> List[np.ndarray]:
        """Preserve transparent screen-space effects that AI subject mattes often drop."""
        preserve = float(np.clip(getattr(self.config, "transparency_preserve", 0.7), 0.0, 1.0))
        if preserve <= 0.0:
            return ai_alphas

        merged: List[np.ndarray] = []
        frame_iter = frames if frames is not None else [None] * len(base_alphas)
        for base_alpha, ai_alpha, frame in zip(base_alphas, ai_alphas, frame_iter):
            base_alpha = np.clip(base_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            ai_alpha = np.clip(ai_alpha.astype(np.float32, copy=False), 0.0, 1.0)
            solid_alpha = np.maximum(
                self._green_screen_ai_solid_layer(ai_alpha, base_alpha, frame),
                self._green_screen_solid_layer(base_alpha, frame),
            )
            effect_alpha = self._green_screen_effect_layer(base_alpha, frame) * preserve
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

    def _green_screen_solid_layer(self, base_alpha: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
        if frame is None:
            return np.where(base_alpha >= 0.92, base_alpha, 0.0)

        solid_color_mask = self._green_screen_solid_foreground_mask(frame)
        soft_subject_mask = self._green_screen_soft_subject_mask(frame)
        solid_mask = ((base_alpha >= 0.92) & solid_color_mask) | ((base_alpha >= 0.28) & soft_subject_mask)
        return np.where(solid_mask, 1.0, 0.0)

    def _green_screen_ai_solid_layer(
        self,
        ai_alpha: np.ndarray,
        base_alpha: np.ndarray,
        frame: np.ndarray | None,
    ) -> np.ndarray:
        if frame is None:
            return np.where((ai_alpha >= 0.98) & (base_alpha >= 0.75), ai_alpha, 0.0)

        solid_color_mask = self._green_screen_solid_foreground_mask(frame)
        soft_subject_mask = self._green_screen_soft_subject_mask(frame)
        effect_like = self._green_screen_effect_color_weight(frame) > 0.45
        ai_solid_mask = (
            (ai_alpha >= 0.98)
            & (
                (base_alpha >= 0.75)
                | soft_subject_mask
                | (solid_color_mask & (~effect_like) & (base_alpha >= 0.55))
            )
        )
        return np.where(ai_solid_mask, ai_alpha, 0.0)

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
