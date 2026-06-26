"""BackgroundMattingV2 AI 抠图模块 - 绿幕最佳方案"""

from typing import List, Optional

import cv2
import numpy as np
import torch

from ..config import MattingConfig


class BGM2Matte:
    """基于 BackgroundMattingV2 的抠图引擎"""

    def __init__(self, config: MattingConfig, background_frame: Optional[np.ndarray] = None):
        self.config = config
        # BGM2 在 CUDA 上容易 ECC 错误，强制用 CPU
        self.device = torch.device("cpu")
        self.model = None
        self.background = None

        # 设置背景
        if background_frame is not None:
            self.set_background(background_frame)

        self._load_model()

    def set_background(self, background_frame: np.ndarray):
        """设置背景帧（纯绿幕/黑底）"""
        # 预处理背景 - 使用较小分辨率避免内存问题
        bg = cv2.resize(background_frame, (512, 288))  # BGM2 推荐分辨率
        bg = bg.astype(np.float32) / 255.0
        bg = torch.from_numpy(bg).permute(2, 0, 1).unsqueeze(0)
        self.background = bg.to(self.device)
        print(f"[BGM2] Background set: {background_frame.shape} -> 512x288")

    def _load_model(self):
        """加载 BackgroundMattingV2 模型"""
        try:
            # 使用 torch.hub 加载预训练模型
            print("[BGM2] Loading model from torch.hub (CPU mode)...")

            # 加载 mobilenetv2 版本（轻量快速）
            self.model = torch.hub.load(
                'PeterL1n/BackgroundMattingV2',
                'mobilenetv2',
                pretrained=True
            )
            self.model.eval().to(self.device)
            print(f"[BGM2] Loaded mobilenetv2 on {self.device}")

        except Exception as e:
            print(f"[BGM2] Failed to load from torch.hub: {e}")
            print("[BGM2] Fallback to traditional green screen")
            self.model = None

    def generate(self, frame: np.ndarray) -> np.ndarray:
        """单帧抠图"""
        if self.model is None:
            from .green_screen_matte import GreenScreenMatte
            return GreenScreenMatte(self.config).generate(frame)

        if self.background is None:
            print("[BGM2] Warning: No background set, using green screen fallback")
            from .green_screen_matte import GreenScreenMatte
            return GreenScreenMatte(self.config).generate(frame)

        # 预处理
        tensor, orig_size = self._preprocess(frame)

        # 推理
        with torch.no_grad():
            pha, fgr = self.model(tensor, self.background)[:2]

        # 后处理
        alpha = self._postprocess(pha, orig_size)

        return alpha

    def generate_sequence(self, frames: List[np.ndarray], progress_callback=None) -> List[np.ndarray]:
        """序列抠图"""
        alphas = []
        total = len(frames)

        for i, frame in enumerate(frames):
            alpha = self.generate(frame)
            alphas.append(alpha)

            if progress_callback and i % max(1, total // 20) == 0:
                progress_callback(i, total)

        return alphas

    def _preprocess(self, frame: np.ndarray):
        """预处理 - BGM2 需要 512x288 或 1920x1080"""
        h, w = frame.shape[:2]

        # 缩放到模型输入尺寸
        if w > h * 2:
            # 宽屏，用 512x288
            new_w, new_h = 512, 288
        else:
            # 标准比例，用 1920x1080
            new_w, new_h = 1920, 1080

        frame_resized = cv2.resize(frame, (new_w, new_h))

        # 归一化
        if frame_resized.dtype == np.uint8:
            frame_f = frame_resized.astype(np.float32) / 255.0
        else:
            frame_f = frame_resized.astype(np.float32)

        # 转换格式
        tensor = torch.from_numpy(frame_f).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        # 如果背景尺寸不匹配，重新设置
        if self.background is not None and self.background.shape != tensor.shape:
            bg_resized = cv2.resize(frame_resized, (new_w, new_h))
            bg_f = bg_resized.astype(np.float32) / 255.0
            self.background = torch.from_numpy(bg_f).permute(2, 0, 1).unsqueeze(0).to(self.device)

        return tensor, (h, w)

    def _postprocess(self, pha: torch.Tensor, orig_size) -> np.ndarray:
        """后处理"""
        h, w = orig_size

        # 提取 alpha
        alpha = pha[0, 0].cpu().numpy()

        # 缩放回原始尺寸
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

        # 归一化
        alpha = np.clip(alpha, 0, 1).astype(np.float32)

        return alpha
