"""混合抠图引擎 - AI + 传统算法融合"""

import logging
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
    
    def generate_sequence(self, frames: List[np.ndarray], bg_mode: BackgroundMode, progress_callback=None) -> List[np.ndarray]:
        """序列抠图"""
        if bg_mode == BackgroundMode.GREEN_SCREEN:
            return self._green_screen_matte(frames, progress_callback)
        elif bg_mode == BackgroundMode.BLACK_BACKGROUND:
            return self._black_background_matte(frames, progress_callback)
        else:
            # Auto - 默认用绿幕逻辑
            return self._green_screen_matte(frames, progress_callback)
    
    def _green_screen_matte(self, frames: List[np.ndarray], progress_callback) -> List[np.ndarray]:
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
                logger.info("Applying GVM edge refinement")
            # 其次使用 MatAnyone2 (人物视频抠图)
            elif self.matanyone2 is not None and self.matanyone2.model is not None:
                ai_engine = self.matanyone2
                logger.info("Applying MatAnyone2 edge refinement")
            # 然后使用 CorridorKey
            elif self.corridorkey is not None and self.corridorkey.model is not None:
                ai_engine = self.corridorkey
                logger.info("Applying CorridorKey edge refinement")
            elif self.rembg is not None and self.rembg._available:
                ai_engine = self.rembg
                logger.info("Applying rembg edge refinement")
            elif self.rmbg is not None and self.rmbg.model is not None:
                ai_engine = self.rmbg
                logger.info("Applying RMBG-2 edge refinement")
            elif self.birefnet is not None and self.birefnet.model is not None:
                ai_engine = self.birefnet
                logger.info("Applying BiRefNet edge refinement")
            elif self.rvm is not None and self.rvm.model is not None:
                ai_engine = self.rvm
                logger.info("Applying RVM edge refinement")
            
            if ai_engine is not None:
                refined_alphas = []
                for i, (frame, base_alpha) in enumerate(zip(frames, base_alphas)):
                    # 只在边缘区域用 AI 辅助
                    edge_mask = (base_alpha > 0.1) & (base_alpha < 0.9)
                    
                    if np.any(edge_mask):
                        # 用 AI 生成边缘 alpha
                        ai_alpha = ai_engine.generate(frame)
                        
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
            # 获取用户指定的模型
            ai_model = getattr(self.config, 'ai_model', 'auto')
            
            # 根据用户选择使用特定模型
            if ai_model == "gvm" and self.gvm is not None and self.gvm.model is not None:
                logger.info("Using GVM for green screen")
                ai_alphas = self.gvm.generate_sequence(frames)
            elif ai_model == "matanyone2" and self.matanyone2 is not None and self.matanyone2.model is not None:
                logger.info("Using MatAnyone2 for green screen")
                ai_alphas = self.matanyone2.generate_sequence(frames)
            elif ai_model == "corridorkey" and self.corridorkey is not None and self.corridorkey.model is not None:
                logger.info("Using CorridorKey for green screen")
                ai_alphas = self.corridorkey.generate_sequence(frames, progress_callback)
            elif ai_model == "sam2" and self.sam2 is not None and (self.sam2.predictor is not None or self.sam2.model is not None):
                logger.info("Using SAM2 for green screen")
                ai_alphas = self.sam2.generate_sequence(frames, progress_callback)
            elif ai_model == "rembg" and self.rembg is not None and self.rembg._available:
                logger.info("Using rembg for green screen")
                ai_alphas = self.rembg.generate_sequence(frames, progress_callback)
            elif ai_model == "birefnet" and self.birefnet is not None and self.birefnet.model is not None:
                logger.info("Using BiRefNet for green screen")
                ai_alphas = self.birefnet.generate_sequence(frames, progress_callback)
            elif ai_model == "rvm" and self.rvm is not None and self.rvm.model is not None:
                logger.info("Using RVM for green screen")
                ai_alphas = self.rvm.generate_sequence(frames, progress_callback)
            # auto 模式：按优先级自动选择
            elif self.corridorkey is not None and self.corridorkey.model is not None:
                logger.info("Using CorridorKey for green screen (auto)")
                ai_alphas = self.corridorkey.generate_sequence(frames, progress_callback)
            elif self.rembg is not None and self.rembg._available:
                logger.info("Using rembg for green screen (auto)")
                ai_alphas = self.rembg.generate_sequence(frames, progress_callback)
            elif self.birefnet is not None and self.birefnet.model is not None:
                logger.info("Using BiRefNet for green screen (auto)")
                ai_alphas = self.birefnet.generate_sequence(frames, progress_callback)
            elif self.rvm is not None and self.rvm.model is not None:
                logger.info("Using RVM for green screen (auto)")
                ai_alphas = self.rvm.generate_sequence(frames, progress_callback)
        
        if ai_alphas is not None:
            ai_alphas = self._merge_green_screen_effects(base_alphas, ai_alphas)
            # 应用 Chroma Key 后处理
            from .chroma_key_postprocess import apply_chroma_key_postprocess
            return apply_chroma_key_postprocess(ai_alphas, self.config)
        
        return base_alphas

    def _merge_green_screen_effects(
        self,
        base_alphas: List[np.ndarray],
        ai_alphas: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Preserve transparent screen-space effects that AI subject mattes often drop."""
        preserve = float(np.clip(getattr(self.config, "transparency_preserve", 0.7), 0.0, 1.0))
        if preserve <= 0.0:
            return ai_alphas

        merged: List[np.ndarray] = []
        for base_alpha, ai_alpha in zip(base_alphas, ai_alphas):
            base_alpha = base_alpha.astype(np.float32, copy=False)
            ai_alpha = ai_alpha.astype(np.float32, copy=False)
            effect_alpha = base_alpha * preserve
            merged.append(np.maximum(ai_alpha, effect_alpha))
        return merged
    
    def _black_background_matte(self, frames: List[np.ndarray], progress_callback) -> List[np.ndarray]:
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
            alpha = self.black_matte.generate(frame)
            base_alphas.append(alpha)
            if progress_callback and i % max(1, len(frames) // 20) == 0:
                progress_callback(i, len(frames))
        
        # 2. 如果有 RVM，用 AI 做边缘细化
        if self.rvm is not None and self.rvm.model is not None:
            logger.info("Applying RVM edge refinement for black background")
            
            # 取关键帧用 RVM 细化边缘
            refined_alphas = []
            for i, (frame, base_alpha) in enumerate(zip(frames, base_alphas)):
                # 只在边缘区域用 AI 辅助
                edge_mask = (base_alpha > 0.05) & (base_alpha < 0.95)
                
                if np.any(edge_mask):
                    # 用 RVM 生成边缘 alpha
                    rvm_alpha = self.rvm.generate(frame)
                    
                    # 融合：核心区域用传统，边缘用 AI
                    core_mask = base_alpha > 0.8
                    bg_mask = base_alpha < 0.05
                    
                    # 核心：传统算法（保留粒子/辉光）
                    # 边缘：AI 细化
                    # 背景：透明
                    result = base_alpha.copy()
                    result[edge_mask] = rvm_alpha[edge_mask] * 0.7 + base_alpha[edge_mask] * 0.3
                    
                    refined_alphas.append(np.clip(result, 0, 1))
                else:
                    refined_alphas.append(base_alpha)
            
            return refined_alphas
        
        return base_alphas
