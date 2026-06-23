"""绿幕 Matte 生成模块

支持参数：
- screen_color: auto/green/blue
- key_strength: 0.1~10.0
- clip_black/clip_white: 0.0~1.0
- shrink_grow: -250~250
- edge_blur: 0~50
"""

import numpy as np
import cv2

from ..config import MattingConfig


class GreenScreenMatte:
    """绿幕抠图引擎"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self._key_color = None
    
    def generate(self, frame: np.ndarray) -> np.ndarray:
        """生成绿幕 Alpha Matte"""
        
        # 调试：打印参数
        print(f"[GreenScreen] key_strength={self.config.key_strength}, "
              f"clip_black={self.config.clip_black}, clip_white={self.config.clip_white}, "
              f"shrink_grow={self.config.shrink_grow}, edge_blur={self.config.edge_blur}")
        
        # 使用 Color-Difference 方法
        alpha = self._chroma_key_matte(frame)
        
        # 应用 Shrink/Grow
        if self.config.shrink_grow != 0:
            alpha = self._apply_shrink_grow(alpha)
        
        # 应用 Edge Blur
        if self.config.edge_blur > 0:
            alpha = self._apply_edge_blur(alpha)
        
        return alpha
    
    def _chroma_key_matte(self, frame: np.ndarray) -> np.ndarray:
        """核心 Chroma Key 算法 — 简化可靠版
        
        基于颜色差异的绿幕抠图：
        1. 检测绿色/蓝色屏幕
        2. 计算颜色差异得到 raw matte
        3. 应用参数调整
        """
        assert frame.ndim == 3 and frame.shape[2] == 3
        
        # 转换为 float32 [0, 1]
        if frame.dtype == np.uint8:
            img = frame.astype(np.float32) / 255.0
        else:
            img = np.clip(frame.astype(np.float32), 0.0, 1.0)
        
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        
        # 自动检测绿幕/蓝幕
        screen_type = self.config.screen_color
        if screen_type == "auto":
            g_avg = np.clip(g - np.maximum(r, b), 0, None).mean()
            b_avg = np.clip(b - np.maximum(r, g), 0, None).mean()
            screen_type = "blue" if b_avg > g_avg else "green"
        
        # 计算 screen color 的 "纯度"
        if screen_type == "green":
            # 绿色纯度 = G - max(R, B)
            screen_purity = np.clip(g - np.maximum(r, b), 0, None)
            # 非绿程度 = 其他通道的活跃程度
            other_activity = np.maximum(r, b)
        else:  # blue
            screen_purity = np.clip(b - np.maximum(r, g), 0, None)
            other_activity = np.maximum(r, g)
        
        # 参考纯度（基于实际绿幕颜色计算）
        # 标准绿幕 [0, 177, 64] 的纯度 = (177/255) - max(0/255, 64/255) ≈ 0.44
        if screen_type == "green":
            ref_purity = 0.35  # 典型绿幕的纯度（降低阈值）
        else:
            ref_purity = 0.30  # 蓝幕通常纯度更低

        similarity = float(np.clip(getattr(self.config, "green_similarity", 0.4), 0.1, 1.0))
        # 0.4 是历史默认值；更高相似度降低参考纯度，抠得更激进。
        similarity_scale = np.clip(1.0 + (0.4 - similarity) * 0.8, 0.52, 1.35)
        ref_purity *= similarity_scale
        
        # 归一化：纯度越高 → key 越接近 1（越像背景）
        key = np.clip(screen_purity / ref_purity, 0, 1)
        
        # 应用 key_strength：
        # strength > 1: 更容易识别为绿幕（抠得更狠）
        # strength < 1: 更保守（保留更多）
        strength = self.config.key_strength
        if strength != 1.0:
            key = np.power(key, 1.0 / max(strength, 0.1))
        
        if screen_type == "green":
            key = self._protect_warm_luminous_foreground_key(key, r, g, b)

        # 亮度保护：很暗的区域不处理（避免把阴影当绿幕）
        brightness = np.maximum(np.maximum(r, g), b)
        dark_mask = np.clip((brightness - 0.03) / 0.07, 0, 1)
        key = key * dark_mask
        
        # Alpha = 前景透明度 = 1 - 背景程度
        alpha = 1.0 - key
        alpha = self._suppress_screen_background_core(alpha, key)
        
        # Clip Black / Clip White
        clip_black = self.config.clip_black
        clip_white = self.config.clip_white
        if clip_black > 0 or clip_white < 1.0:
            cw = max(clip_white, clip_black + 0.001)
            alpha = np.clip((alpha - clip_black) / (cw - clip_black), 0.0, 1.0)
        
        # 形态学清理
        alpha_u8 = (alpha * 255).astype(np.uint8)
        if alpha_u8.max() > 0 and alpha_u8.min() < 255:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            alpha_u8 = cv2.morphologyEx(alpha_u8, cv2.MORPH_OPEN, kernel)
            alpha_u8 = cv2.morphologyEx(alpha_u8, cv2.MORPH_CLOSE, kernel)
        
        return alpha_u8.astype(np.float32) / 255.0

    def _protect_warm_luminous_foreground_key(
        self,
        key: np.ndarray,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """Keep yellow/orange glow cores from being keyed as backing-screen spill."""
        warm_luminous = (
            (r > 0.48)
            & (g > 0.42)
            & (b < 0.58)
            & ((r - b) > 0.16)
            & ((g - b) > 0.10)
            & ((r + 0.26) > g)
        )
        if not np.any(warm_luminous):
            return key

        protected_key = key.copy()
        solid_core = warm_luminous & (r > 0.62) & (g > 0.50) & (b < 0.50)
        protected_key[warm_luminous] *= 0.18
        protected_key[solid_core] = 0.0
        return protected_key

    def _suppress_screen_background_core(self, alpha: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Force confident backing-screen pixels to fully transparent."""
        if alpha.size < 4096:
            return alpha
        background_core = key >= 0.65
        if not np.any(background_core):
            return alpha
        cleaned = alpha.copy()
        cleaned[background_core] = 0.0
        transition = (key >= 0.50) & (key < 0.65)
        if np.any(transition):
            transition_weight = np.clip((key[transition] - 0.50) / 0.15, 0.0, 1.0)
            cleaned[transition] = cleaned[transition] * (1.0 - transition_weight)
        return cleaned
    
    def _compute_ref_excess(self, img: np.ndarray, sc: int, c1: int, c2: int, screen_type: str) -> float:
        """计算参考 excess 值
        
        支持多点采样，使用 10th percentile 避免异常值
        """
        samples = self.config.key_samples
        
        if samples and len(samples) > 1:
            excesses = []
            for s in samples:
                sf = np.array(s, dtype=np.float32) / 255.0
                exc = sf[sc] - max(sf[c1], sf[c2])
                if exc > 0.0:
                    excesses.append(exc)
            if excesses:
                excesses.sort()
                idx = max(0, int(len(excesses) * 0.1))
                return max(excesses[idx], 0.01)
        
        # 单点采样
        if self.config.key_color is not None:
            ref = np.array(self.config.key_color, dtype=np.float32) / 255.0
        elif screen_type == "blue":
            ref = np.array([0.0, 0.0, 0.9], dtype=np.float32)
        else:
            ref = np.array([0.0, 0.9, 0.0], dtype=np.float32)
        
        return max(ref[sc] - max(ref[c1], ref[c2]), 0.01)
    
    def _apply_shrink_grow(self, alpha: np.ndarray) -> np.ndarray:
        """应用 Shrink/Grow（腐蚀/膨胀）"""
        shrink_grow = self.config.shrink_grow
        abs_px = abs(shrink_grow)
        
        if abs_px == 0:
            return alpha
        
        alpha_u8 = (alpha * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (abs_px * 2 + 1, abs_px * 2 + 1)
        )
        
        if shrink_grow < 0:
            alpha_u8 = cv2.erode(alpha_u8, kernel)
        else:
            alpha_u8 = cv2.dilate(alpha_u8, kernel)
        
        return alpha_u8.astype(np.float32) / 255.0
    
    def _apply_edge_blur(self, alpha: np.ndarray) -> np.ndarray:
        """应用 Edge Blur（高斯模糊）"""
        edge_blur = self.config.edge_blur
        if edge_blur <= 0:
            return alpha
        
        alpha_u8 = (alpha * 255).astype(np.uint8)
        k = edge_blur * 2 + 1
        alpha_u8 = cv2.GaussianBlur(alpha_u8, (k, k), 0)
        
        return alpha_u8.astype(np.float32) / 255.0
    
    def _estimate_key_color(self, frame: np.ndarray) -> np.ndarray:
        """自动估计 Key 色"""
        h, w = frame.shape[:2]
        
        edge_regions = [
            frame[0:h//6, :],
            frame[-h//6:, :],
            frame[:, 0:w//6],
            frame[:, -w//6:],
        ]
        
        all_pixels = np.vstack([r.reshape(-1, 3) for r in edge_regions])
        greenness = all_pixels[:, 1].astype(np.float32) - np.maximum(
            all_pixels[:, 0].astype(np.float32), 
            all_pixels[:, 2].astype(np.float32)
        )
        
        green_mask = greenness > 20
        green_pixels = all_pixels[green_mask]
        
        if len(green_pixels) > 10:
            top_k = max(int(len(green_pixels) * 0.2), 10)
            top_indices = np.argpartition(greenness[green_mask], -top_k)[-top_k:]
            key_color = np.mean(green_pixels[top_indices], axis=0)
        else:
            key_color = np.array([20, 200, 20], dtype=np.float32)
        
        print(f"[GreenScreen] Key color: {key_color.astype(int)}")
        return key_color
