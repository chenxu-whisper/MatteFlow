"""颜色去污染模块 - 优化版"""

import numpy as np
import cv2

from ..analysis.region_ownership import RegionOwnershipAnalyzer
from ..config import MattingConfig, BackgroundMode


class ColorDecontaminate:
    """颜色去污染/边缘修复 - 优化版：更强去绿边"""
    
    def __init__(self, config: MattingConfig):
        self.config = config
        self._region_analyzer = RegionOwnershipAnalyzer()
    
    def process(self, frames, alphas, bg_mode, context=None):
        processed = []
        context = context or {}
        for index, (frame, alpha) in enumerate(zip(frames, alphas)):
            ownership = self._ownership_from_context(context, index)
            if bg_mode == BackgroundMode.GREEN_SCREEN:
                result = self._remove_green_spill(frame, alpha, ownership=ownership)
            elif bg_mode == BackgroundMode.BLACK_BACKGROUND:
                result = self._remove_black_spill(frame, alpha)
            else:
                result = frame.copy()
            processed.append(result)
        return processed

    def _transparency_band(self, alpha: np.ndarray, low: float = 0.02, high: float = 0.75) -> np.ndarray:
        return (alpha > low) & (alpha < high)

    def _estimate_green_screen_color(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self.config.key_color is not None:
            return np.array(self.config.key_color, dtype=np.float32)

        frame_f = frame.astype(np.float32, copy=False)
        h, w = alpha.shape
        border = max(min(min(h, w) // 6, 96), 1)
        border_mask = np.zeros_like(alpha, dtype=bool)
        border_mask[:border, :] = True
        border_mask[-border:, :] = True
        border_mask[:, :border] = True
        border_mask[:, -border:] = True

        bg_mask = alpha <= 0.02
        if int(bg_mask.sum()) < 64:
            bg_mask = alpha <= 0.08
        bg_mask = bg_mask & border_mask
        if int(bg_mask.sum()) < 64:
            bg_mask = border_mask

        r, g, b = frame_f[:, :, 0], frame_f[:, :, 1], frame_f[:, :, 2]
        green_score = g - np.maximum(r, b)
        green_candidates = bg_mask & (green_score > 10.0)
        if int(green_candidates.sum()) >= 32:
            pixels = frame_f[green_candidates]
        else:
            pixels = frame_f[bg_mask]

        if pixels.size == 0:
            return np.array([0.0, 255.0, 0.0], dtype=np.float32)

        return np.median(pixels, axis=0).astype(np.float32)

    def _recover_green_screen_foreground(
        self,
        frame: np.ndarray,
        alpha: np.ndarray,
        screen_rgb: np.ndarray,
    ) -> np.ndarray:
        frame_f = frame.astype(np.float32, copy=False)
        alpha_f = np.clip(alpha.astype(np.float32, copy=False), 0.0, 1.0)
        safe_alpha = np.maximum(alpha_f, 0.18)
        screen = screen_rgb.astype(np.float32, copy=False).reshape(1, 1, 3)
        recovered = (frame_f - (1.0 - alpha_f)[..., np.newaxis] * screen) / safe_alpha[..., np.newaxis]
        return np.clip(recovered, 0.0, 255.0)
    
    def _remove_green_spill(self, frame: np.ndarray, alpha: np.ndarray, ownership=None) -> np.ndarray:
        """去除绿溢色 - 保守版：只处理明显绿边，保护白色/浅色区域"""
        result = frame.astype(np.float32)
        r, g, b = result[:, :, 0], result[:, :, 1], result[:, :, 2]
        screen_rgb = self._estimate_green_screen_color(frame, alpha)
        recovered_fg = self._recover_green_screen_foreground(frame, alpha, screen_rgb)
        
        strength = self.config.green_despill_strength
        
        # 1. 检测绿色溢色。主体边缘仍使用严格阈值，半透明辉光区使用更敏感阈值。
        threshold = 15
        green_tint = (g > r + threshold) & (g > b + threshold)
        
        # 2. 强白色保护 - 避免把白色/灰色去成粉色
        brightness = np.mean(result, axis=2)
        # 白色 = 高亮度 + 低饱和度（RGB接近）
        rgb_diff = np.abs(r - g) + np.abs(g - b) + np.abs(r - b)
        white_brightness = getattr(self.config, "white_protect_brightness", 180)
        white_saturation = getattr(self.config, "white_protect_saturation", 25)
        white_mask = (brightness > white_brightness) & (rgb_diff < white_saturation)
        
        # 3. 只处理边缘/半透明区域，完全前景和完全背景不处理
        edge_mask = (alpha > 0.05) & (alpha < 0.95)
        glow_mask = (alpha > 0.005) & (alpha < 0.85)
        subtle_green_haze = (g > r + 4) & (g >= b - 3) & glow_mask
        pink_glow_haze = (
            glow_mask
            & (brightness > 135)
            & (r > b + 10)
            & (g > b + 6)
            & (g > 145)
        )
        
        # 4. 计算绿色过量（只在边缘且明显发绿的地方）
        green_excess = np.maximum(0, g - np.maximum(r, b))
        
        # 边缘去绿：alpha 越低（越接近背景）→ 去绿越强
        edge_factor = np.clip((0.95 - alpha) / 0.9, 0, 1)  # 0.05→1.0, 0.95→0.0
        despill = green_excess * strength * edge_factor * self.config.edge_despill_factor
        haze_excess = np.maximum(0, g - r - 2)
        haze_factor = np.clip((0.7 - alpha) / 0.7, 0, 1)
        haze_despill = haze_excess * strength * haze_factor * self.config.edge_despill_factor
        target_haze_despill = (
            np.maximum(0, g - np.maximum(r, b) - 3)
            * strength
            * self.config.edge_despill_factor
        )
        pink_haze_excess = np.maximum(0, g - b - 2)
        pink_haze_despill = (
            pink_haze_excess
            * strength
            * haze_factor
            * self.config.edge_despill_factor
            * 1.55
        )
        
        # 主体边缘只处理明显发绿；辉光区额外处理低饱和绿色雾边。
        despill_mask = (green_tint & edge_mask) | subtle_green_haze | pink_glow_haze
        glow_despill = np.maximum(haze_despill, target_haze_despill)
        despill = np.where(subtle_green_haze, np.maximum(despill, glow_despill), despill)
        despill = np.where(pink_glow_haze, np.maximum(despill, pink_haze_despill), despill)
        despill = despill * despill_mask
        
        # 白色区域：几乎不去绿（保护白色毛发/耳朵）
        despill = np.where(white_mask, despill * 0.05, despill)
        
        # 5. 应用去绿
        g_corrected = g - despill
        
        # 6. 补偿到 R 和 B（保持亮度）
        compensation = despill * 0.5
        r_corrected = r + compensation
        b_corrected = b + compensation

        # 黄色/橙色发光物周围经常有 alpha 已经变成 1 的绿幕混色边。
        # 这些像素不会进入普通 edge_mask，所以用邻接黄色发光区域单独中和。
        warm_luminous = (
            (r > 170)
            & (g > 115)
            & (b < 175)
            & ((r - b) > 45)
            & ((g - b) > 8)
        )
        if np.any(warm_luminous):
            warm_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            warm_reach = cv2.dilate(warm_luminous.astype(np.uint8), warm_kernel, iterations=1).astype(bool)
            solid_green_spill = (
                warm_reach
                & (alpha > 0.20)
                & (g > r + 12)
                & (g > b + 6)
                & (r < 170)
                & (g > 95)
            )
            if np.any(solid_green_spill):
                spill_gap = np.maximum(0, g - np.maximum(r, b))
                r_corrected = np.where(solid_green_spill, r_corrected + spill_gap * 0.70, r_corrected)
                g_corrected = np.where(solid_green_spill, g_corrected - spill_gap * 0.45, g_corrected)
                b_corrected = np.where(solid_green_spill, b_corrected + spill_gap * 0.18, b_corrected)

        # 6.5 对中高 alpha 的混色边做真正的前景反混色恢复。
        # 仅在存在明显绿色屏幕污染时启用，避免改坏正常蓝紫辉光。
        screen_mix_score = np.maximum(0.0, g - r) + 0.5 * np.maximum(0.0, g - b)
        screen_mix = np.clip((screen_mix_score - 12.0) / 64.0, 0.0, 1.0)
        unmix_zone = (alpha > 0.18) & (alpha < 0.88) & (screen_mix > 0.0)
        unmix_weight = screen_mix * np.clip((0.95 - alpha) / 0.20, 0.70, 1.0)
        unmix_weight = np.where(unmix_zone, unmix_weight, 0.0).astype(np.float32)
        r_corrected = r_corrected * (1.0 - unmix_weight) + recovered_fg[:, :, 0] * unmix_weight
        g_corrected = g_corrected * (1.0 - unmix_weight) + recovered_fg[:, :, 1] * unmix_weight
        b_corrected = b_corrected * (1.0 - unmix_weight) + recovered_fg[:, :, 2] * unmix_weight
        
        # 7. 半透明辉光补亮：AI/绿幕融合会保留 alpha，但 RGB 仍可能是暗背景色，
        # 合成到棋盘或任意背景上会表现为黑边。只修低亮度半透明区，避开主体白色区域。
        corrected_brightness = (r_corrected + g_corrected + b_corrected) / 3.0
        transparency_mask = self._transparency_band(alpha, 0.02, 0.78)
        dark_glow = transparency_mask & (corrected_brightness < 105) & (~white_mask)
        lift = np.clip((122 - corrected_brightness) * 0.9, 0, 72) * dark_glow
        r_corrected = r_corrected + lift * 1.20
        g_corrected = g_corrected + lift * 0.12
        b_corrected = b_corrected + lift * 1.05

        # 8. 亮青色辉光中仍可能混入绿幕色。当前规则主要处理纯绿边，
        # 这里额外中和亮的青色外缘，避免白环外侧残留一圈发绿的 teal halo。
        corrected_brightness = (r_corrected + g_corrected + b_corrected) / 3.0
        corrected_chroma = np.maximum.reduce([r_corrected, g_corrected, b_corrected]) - np.minimum.reduce(
            [r_corrected, g_corrected, b_corrected]
        )
        white_ring_cleanup_strength = float(
            np.clip(getattr(self.config, "white_ring_cleanup_strength", 1.0), 0.0, 2.0)
        )

        bright_teal_haze = (
            transparency_mask
            & (corrected_brightness > 135)
            & (corrected_chroma < 90)
            & (g_corrected > r_corrected + 10)
            & (b_corrected > r_corrected + 10)
        )
        teal_gap = np.maximum(0, np.minimum(g_corrected - r_corrected, b_corrected - r_corrected))
        teal_cleanup = white_ring_cleanup_strength * bright_teal_haze
        r_corrected = r_corrected + teal_gap * 0.50 * teal_cleanup
        g_corrected = g_corrected - teal_gap * 0.18 * teal_cleanup

        # 9. 白环外侧的混色边 alpha 往往并不低，单靠半透明规则不够。
        # 仅在亮白环的邻接带里继续中和偏青绿色，避免外沿残留一圈绿边。
        bright_ring = (alpha > 0.92) & (corrected_brightness > 210) & (corrected_chroma < 55)
        if np.any(bright_ring):
            ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            ring_neighbor = cv2.dilate(bright_ring.astype(np.uint8), ring_kernel, iterations=1).astype(bool)
            ring_neighbor &= ~bright_ring
            ring_teal_haze = (
                ring_neighbor
                & (alpha > 0.05)
                & (alpha < 0.88)
                & (corrected_chroma < 110)
                & (g_corrected > r_corrected + 10)
                & (b_corrected > r_corrected + 10)
            )
            ring_teal_gap = np.maximum(
                0, np.minimum(g_corrected - r_corrected, b_corrected - r_corrected)
            )
            ring_cleanup = white_ring_cleanup_strength * ring_teal_haze
            r_corrected = r_corrected + ring_teal_gap * 0.55 * ring_cleanup
            g_corrected = g_corrected - ring_teal_gap * 0.20 * ring_cleanup

        # 10. 对黄色发光道具邻域做局部色彩投影。这里不改变 alpha，只把仍然
        # 朝绿幕方向偏移的残留像素投回暖色/中性色方向，保护星星和细杆边缘。
        prop_reach = self._warm_luminous_prop_reach(frame, alpha)
        if ownership is None:
            ownership = self._region_analyzer.analyze(frame, alpha)
        region_prop = ownership.luminous_prop
        prop_reach = prop_reach | region_prop
        if np.any(prop_reach):
            prop_green_residue = (
                prop_reach
                & (alpha > 0.08)
                & (g_corrected > r_corrected + 6)
                & (g_corrected > b_corrected + 3)
            )
            if np.any(prop_green_residue):
                prop_gap = np.maximum(0.0, g_corrected - np.maximum(r_corrected, b_corrected))
                r_corrected = np.where(prop_green_residue, r_corrected + prop_gap * 0.45, r_corrected)
                g_corrected = np.where(prop_green_residue, g_corrected - prop_gap * 0.85, g_corrected)
                b_corrected = np.where(prop_green_residue, b_corrected + prop_gap * 0.10, b_corrected)

        result[:, :, 0] = np.clip(r_corrected, 0, 255)
        result[:, :, 1] = np.clip(g_corrected, 0, 255)
        result[:, :, 2] = np.clip(b_corrected, 0, 255)
        
        return result.astype(np.uint8)

    @staticmethod
    def _ownership_from_context(context: dict, index: int):
        ownerships = context.get("region_ownership") if context else None
        if ownerships is None or index >= len(ownerships):
            return None
        return ownerships[index]

    def _warm_luminous_prop_reach(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        frame_f = frame.astype(np.float32, copy=False)
        red = frame_f[:, :, 0]
        green = frame_f[:, :, 1]
        blue = frame_f[:, :, 2]
        screen_green = (green > red + 30.0) & (green > blue + 20.0) & (green > 90.0)
        warm_seed = (
            (red > 170.0)
            & (green > 115.0)
            & (blue < 175.0)
            & ((red - blue) > 45.0)
            & ((green - blue) > 8.0)
            & (~screen_green)
            & (alpha > 0.20)
        )
        gold_green_edge_seed = (
            (red > 110.0)
            & (green > 100.0)
            & (blue < 130.0)
            & ((red - blue) > 35.0)
            & ((green - blue) > 35.0)
            & ((green - red) < 55.0)
            & (~screen_green)
            & (alpha > 0.20)
        )
        warm_seed = warm_seed | gold_green_edge_seed
        if not np.any(warm_seed):
            return np.zeros(alpha.shape, dtype=bool)
        return cv2.dilate(
            warm_seed.astype(np.uint8, copy=False),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
            iterations=1,
        ).astype(bool)
    
    def _remove_black_spill(self, frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """去除黑边/发灰 - 修复版：保护粒子/辉光颜色"""
        result = frame.astype(np.float32)
        
        brightness = np.mean(result, axis=2)
        max_c = np.max(result, axis=2)
        min_c = np.min(result, axis=2)
        color_range = max_c - min_c
        
        # 1. 黑边区域（暗部边缘）
        black_edge = (alpha > 0.05) & (alpha < 0.95) & (brightness < 60)
        
        # 2. 提升暗部 - 保守提升，避免过曝粒子
        boost = self.config.black_despill_strength
        # 根据 alpha 调整提升量：半透明区域提升更多
        lift = np.maximum(0, 50 - brightness) * boost * (1.0 - alpha) * 0.3
        
        for c in range(3):
            result[:, :, c] = np.where(black_edge, result[:, :, c] + lift, result[:, :, c])
        
        # 3. 颜色增强（去灰）- 保护已有颜色的粒子
        hsv = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        transparency_mask = self._transparency_band(alpha, 0.02, 0.75)
        
        # 只对灰度区域增强饱和度
        gray_mask = (alpha > 0.1) & (s < 50) & (v > 15) & (color_range < 15)
        s_boost = s * (1.0 + self.config.black_contrast_restore * 0.5)
        s = np.where(gray_mask, np.clip(s_boost, 0, 255), s)
        
        # 4. 暗部亮度提升 - 保护极暗粒子
        dark_mask = (alpha > 0.05) & (v < 60) & (v > 5)  # v > 5 保护纯黑
        v_boost = v * (1.0 + self.config.black_contrast_restore * 0.2)
        v = np.where(dark_mask, np.clip(v_boost, 0, 255), v)

        # 4.1 半透明粒子/灰雾单独修复：更积极地抬亮，并避免仍然偏灰。
        dim_particle = transparency_mask & (v < 70) & (v > 5)
        gray_haze_mask = transparency_mask & (s < 32) & (color_range < 18) & (v > 10)
        particle_v_boost = v + np.clip((78 - v) * 0.35, 0, 18)
        v = np.where(dim_particle, np.clip(particle_v_boost, 0, 255), v)
        haze_s_boost = s + np.clip((40 - s) * 0.45, 0, 18)
        s = np.where(gray_haze_mask, np.clip(haze_s_boost, 0, 255), s)
        
        hsv[:, :, 1] = s
        hsv[:, :, 2] = v
        
        result = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # 5. 粒子颜色保护 - 暗部亮点保持原色
        particle_mask = (alpha > 0.1) & (brightness < 40) & (color_range > 10)
        # 这些区域不过度处理，保持原始颜色
        original = frame.astype(np.float32)
        blend = 0.3  # 保留 30% 原始颜色
        for c in range(3):
            result[:, :, c] = np.where(particle_mask, 
                                       result[:, :, c] * (1 - blend) + original[:, :, c] * blend,
                                       result[:, :, c])

        # 半透明低亮粒子再做一次轻量 RGB 提亮，避免只提 V 后仍显得发灰。
        dim_particle_rgb = transparency_mask & (np.mean(result, axis=2) < 52)
        rgb_lift = np.clip((58 - np.mean(result, axis=2)) * 0.35, 0, 10)
        result = result.astype(np.float32)
        result[:, :, 0] = np.where(dim_particle_rgb, result[:, :, 0] + rgb_lift * 0.85, result[:, :, 0])
        result[:, :, 1] = np.where(dim_particle_rgb, result[:, :, 1] + rgb_lift * 1.00, result[:, :, 1])
        result[:, :, 2] = np.where(dim_particle_rgb, result[:, :, 2] + rgb_lift * 0.90, result[:, :, 2])
        
        return np.clip(result, 0, 255).astype(np.uint8)
