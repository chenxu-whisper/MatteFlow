"""输入解码模块"""

import logging

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict

from .formats import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def _read_image_rgb(image_path: Path) -> np.ndarray | None:
    """Read a supported image file as RGB uint8/float array."""
    if image_path.suffix.lower() == ".exr":
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if len(img.shape) == 3 and img.shape[2] >= 3:
            return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        return img

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class VideoDecoder:
    """视频解码器"""
    
    def decode(self, video_path: Path) -> Tuple[List[np.ndarray], Dict]:
        """
        解码视频文件
        
        Returns:
            frames: RGB 帧列表
            meta: 元信息字典
        """
        logger.info("Decoding video: path=%s format=%s", video_path, video_path.suffix.lower())
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.info("Failed to open video: path=%s", video_path)
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
        logger.info(
            "Loaded video: path=%s width=%s height=%s fps=%.3f expected_frames=%s actual_frames=%s",
            video_path,
            width,
            height,
            fps,
            total_frames,
            len(frames),
        )
        
        meta = {
            "input_kind": "video",
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "actual_frames": len(frames),
        }
        
        return frames, meta


class SequenceDecoder:
    """序列帧解码器"""
    
    SUPPORTED_EXTS = IMAGE_EXTENSIONS
    
    def decode(self, dir_path: Path) -> Tuple[List[np.ndarray], Dict]:
        """
        解码序列帧目录
        
        Returns:
            frames: RGB 帧列表
            meta: 元信息字典
        """
        logger.info(
            "Decoding image sequence: path=%s supported_formats=%s",
            dir_path,
            sorted(self.SUPPORTED_EXTS),
        )
        # 收集所有支持的图片文件
        image_files = []
        for ext in self.SUPPORTED_EXTS:
            image_files.extend(dir_path.glob(f"*{ext}"))
            image_files.extend(dir_path.glob(f"*{ext.upper()}"))
        
        # 去重并排序
        image_files = sorted(list(set(image_files)))
        logger.info("Discovered image sequence files: path=%s discovered=%s", dir_path, len(image_files))
        
        if not image_files:
            logger.info("No supported image files found in sequence directory: path=%s", dir_path)
            raise ValueError(f"No supported images found in {dir_path}")
        
        frames = []
        for img_path in image_files:
            img = _read_image_rgb(img_path)
            if img is None:
                logger.info("Skipping unreadable sequence frame: path=%s format=%s", img_path, img_path.suffix.lower())
                continue
            frames.append(img)
        
        if not frames:
            logger.info("Failed to load any sequence frames: path=%s discovered=%s", dir_path, len(image_files))
            raise ValueError(f"Failed to load any frames from {dir_path}")
        
        height, width = frames[0].shape[:2]
        logger.info(
            "Loaded image sequence: path=%s frames=%s width=%s height=%s first=%s last=%s",
            dir_path,
            len(frames),
            width,
            height,
            image_files[0].name,
            image_files[-1].name,
        )
        
        meta = {
            "input_kind": "sequence",
            "width": width,
            "height": height,
            "fps": 30.0,  # 序列帧默认 30fps
            "total_frames": len(frames),
            "actual_frames": len(frames),
            "source_files": [str(f.name) for f in image_files],
        }
        
        return frames, meta


class ImageDecoder:
    """单张图片解码器"""

    def decode(self, image_path: Path) -> Tuple[List[np.ndarray], Dict]:
        logger.info("Decoding single image: path=%s format=%s", image_path, image_path.suffix.lower())
        img = _read_image_rgb(image_path)
        if img is None:
            logger.info("Failed to open image: path=%s format=%s", image_path, image_path.suffix.lower())
            raise ValueError(f"Cannot open image: {image_path}")

        height, width = img.shape[:2]
        logger.info(
            "Loaded single image: path=%s width=%s height=%s channels=%s dtype=%s",
            image_path,
            width,
            height,
            img.shape[2] if img.ndim == 3 else 1,
            img.dtype,
        )
        meta = {
            "input_kind": "image",
            "width": width,
            "height": height,
            "fps": 1.0,
            "total_frames": 1,
            "actual_frames": 1,
            "source_files": [image_path.name],
        }

        return [img], meta
