"""Chroma Key 后处理工具 — 对齐 EZ-CorridorKey

所有 AI 模型生成的 Alpha 都可以应用这些后处理参数。
"""

import numpy as np
import cv2
from typing import List

from ..config import MattingConfig


def apply_chroma_key_postprocess(alphas: List[np.ndarray], config: MattingConfig) -> List[np.ndarray]:
    """应用 Chroma Key 后处理参数
    
    参数：
    - clip_black/clip_white: Alpha 裁剪
    - shrink_grow: 腐蚀/膨胀
    - edge_blur: 高斯模糊
    - Clip Black / Clip White
    - Shrink/Grow
    - Edge Blur
    """
    if not alphas:
        return alphas
    
    # 如果参数都是默认值，跳过处理
    if (config.clip_black == 0.0 and 
        config.clip_white == 1.0 and 
        config.shrink_grow == 0 and 
        config.edge_blur == 0):
        return alphas
    
    processed = []
    for alpha in alphas:
        # 1. Clip Black / Clip White
        clip_black = config.clip_black
        clip_white = config.clip_white
        if clip_black > 0 or clip_white < 1.0:
            cw = max(clip_white, clip_black + 0.001)
            alpha = np.clip((alpha - clip_black) / (cw - clip_black), 0.0, 1.0)
        
        # 2. Shrink/Grow (腐蚀/膨胀)
        shrink_grow = config.shrink_grow
        if shrink_grow != 0:
            abs_px = abs(shrink_grow)
            alpha_u8 = (alpha * 255).astype(np.uint8)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (abs_px * 2 + 1, abs_px * 2 + 1)
            )
            if shrink_grow < 0:
                alpha_u8 = cv2.erode(alpha_u8, kernel)
            else:
                alpha_u8 = cv2.dilate(alpha_u8, kernel)
            alpha = alpha_u8.astype(np.float32) / 255.0
        
        # 3. Edge Blur (高斯模糊)
        edge_blur = config.edge_blur
        if edge_blur > 0:
            alpha_u8 = (alpha * 255).astype(np.uint8)
            k = edge_blur * 2 + 1
            alpha_u8 = cv2.GaussianBlur(alpha_u8, (k, k), 0)
            alpha = alpha_u8.astype(np.float32) / 255.0
        
        processed.append(alpha)
    
    return processed
