"""输入解码模块"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


class VideoDecoder:
    """视频解码器"""
    
    def decode(self, video_path: Path) -> Tuple[List[np.ndarray], Dict]:
        """
        解码视频文件
        
        Returns:
            frames: RGB 帧列表
            meta: 元信息字典
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        meta = {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "actual_frames": len(frames),
        }
        
        return frames, meta


class SequenceDecoder:
    """序列帧解码器"""
    
    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".exr", ".tiff", ".tif"}
    
    def decode(self, dir_path: Path) -> Tuple[List[np.ndarray], Dict]:
        """
        解码序列帧目录
        
        Returns:
            frames: RGB 帧列表
            meta: 元信息字典
        """
        # 收集所有支持的图片文件
        image_files = []
        for ext in self.SUPPORTED_EXTS:
            image_files.extend(dir_path.glob(f"*{ext}"))
            image_files.extend(dir_path.glob(f"*{ext.upper()}"))
        
        # 去重并排序
        image_files = sorted(list(set(image_files)))
        
        if not image_files:
            raise ValueError(f"No supported images found in {dir_path}")
        
        frames = []
        for img_path in image_files:
            # 读取图片
            if img_path.suffix.lower() == ".exr":
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                # EXR 通常是 BGR，转换为 RGB
                if len(img.shape) == 3 and img.shape[2] >= 3:
                    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            frames.append(img)
        
        if not frames:
            raise ValueError(f"Failed to load any frames from {dir_path}")
        
        height, width = frames[0].shape[:2]
        
        meta = {
            "width": width,
            "height": height,
            "fps": 30.0,  # 序列帧默认 30fps
            "total_frames": len(frames),
            "actual_frames": len(frames),
            "source_files": [str(f.name) for f in image_files],
        }
        
        return frames, meta
