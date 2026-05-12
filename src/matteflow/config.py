"""MatteFlow 配置模块"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple


class BackgroundMode(Enum):
    """背景类型"""
    AUTO = "auto"
    GREEN_SCREEN = "green_screen"
    BLACK_BACKGROUND = "black_background"
    UNKNOWN = "unknown"


class QualityMode(Enum):
    """质量模式"""
    FAST = "fast"       # 预览模式，低质量快出
    STANDARD = "standard"  # 标准生产模式
    HIGH = "high"       # 高质量离线导出


@dataclass
class MattingConfig:
    """抠图配置参数 - 全面对齐 EZ-CorridorKey"""
    
    # ==================== 背景模式 ====================
    background_mode: BackgroundMode = BackgroundMode.AUTO
    
    # ==================== 质量模式 ====================
    quality_mode: QualityMode = QualityMode.STANDARD
    
    # ==================== AI 模型选择 ====================
    ai_model: str = "auto"  # "auto" | "gvm" | "matanyone2" | "corridorkey" | "birefnet" | "rvm" | "rembg" | "sam2"
    
    # ==================== Chroma Key 参数 (对齐 EZ-CorridorKey) ====================
    # 屏幕颜色检测/设置
    screen_color: str = "auto"          # "auto" | "green" | "blue" — 屏幕颜色
    screen_color_auto_detect: bool = True  # 自动检测绿/蓝屏
    
    # Key 色采样（支持多点采样）
    key_color: Optional[Tuple[int, int, int]] = None  # 单点采样颜色
    key_samples: list = field(default_factory=list)   # 多点采样列表 [(r,g,b), ...]
    
    # Key Strength — 抠像强度
    key_strength: float = 1.0           # 0.1 ~ 10.0，默认 1.0
    
    # Clip Black / Clip White — Alpha 裁剪
    clip_black: float = 0.0             # 0.0 ~ 1.0，低于此值的 Alpha 置为 0
    clip_white: float = 1.0             # 0.0 ~ 1.0，高于此值的 Alpha 置为 1
    
    # Shrink/Grow — 边缘收缩/扩张
    shrink_grow: int = 0                # -250 ~ 250 px，负值腐蚀，正值膨胀
    
    # Edge Blur — 边缘模糊
    edge_blur: int = 0                  # 0 ~ 50 px，高斯模糊半径
    
    # ==================== 传统绿幕参数（保留兼容） ====================
    green_key_color: Optional[Tuple[int, int, int]] = None  # 自定义 key 色，None 为自动采样
    green_similarity: float = 0.4       # 绿色相似度阈值 (0-1)
    green_despill_strength: float = 0.7  # 去绿边强度 (0-1)
    green_highlight_protect: float = 0.8  # 高光保护
    green_shadow_protect: float = 0.3    # 阴影保护
    green_hair_detail: float = 0.7       # 毛发细节保护
    
    # 白色保护参数（防止白色耳朵/毛发被误抠）
    white_protect_brightness: float = 180  # 白色保护亮度阈值 (0-255)
    white_protect_saturation: float = 25   # 白色保护饱和度阈值 (0-255)
    
    # 边缘去绿参数
    edge_despill_factor: float = 1.2      # 边缘去绿强度系数 (0.5-2.0)
    
    # ==================== 黑底参数 ====================
    black_threshold: float = 0.05         # 黑场阈值
    black_glow_preserve: float = 0.8      # 辉光保留
    black_particle_boost: float = 0.5     # 弱粒子增强
    black_despill_strength: float = 0.6   # 去黑边强度
    black_contrast_restore: float = 0.4   # 对比恢复
    
    # ==================== 通用参数 ====================
    edge_softness: float = 0.3            # 边缘柔化
    transparency_preserve: float = 0.7    # 半透明保留强度
    temporal_strength: float = 0.5        # 时序稳定强度
    
    # 毛发/羽毛专项
    hair_protect: bool = True             # 发丝/毛发保护
    feather_enhance: bool = False         # 羽毛/绒毛增强
    motion_blur_protect: bool = True      # 运动模糊边缘保护
    
    # 纯色绿幕模式
    pure_color_mode: bool = False         # 纯色绿幕模式（更激进的背景去除）
    use_guided_filter: bool = False       # 是否使用导向滤波
    
    # ==================== EZ-CorridorKey 风格后处理参数 ====================
    # 去溢色 (Despill)
    despill_enable: bool = True           # 启用去溢色
    despill_strength: float = 0.5         # 去溢色强度 0.0~1.0，对齐 EZ-CorridorKey 默认 0.5
    despill_color: str = "auto"           # "green" | "blue" | "auto"
    
    # 去噪点 (Auto-Despeckle)
    auto_despeckle: bool = True           # 启用自动去噪点
    despeckle_size: int = 400             # 噪点最小面积（像素数），对齐 EZ-CorridorKey
    despeckle_dilation: int = 25          # 去噪膨胀半径
    despeckle_blur: int = 5               # 去噪模糊半径
    
    # 边缘细化 (Refiner)
    refiner_enable: bool = True           # 启用边缘细化
    refiner_scale: float = 1.0            # 细化强度倍数 0.0~2.0，对齐 EZ-CorridorKey
    
    # 源像素直通 (Source Passthrough)
    source_passthrough: bool = True       # 不透明区域使用原始像素
    edge_erode_px: Optional[int] = None   # 边缘腐蚀像素（防止绿边）
    edge_blur_px: Optional[int] = None    # 边缘模糊像素（过渡柔和度）
    
    # 垃圾遮罩 (Garbage Matte)
    garbage_matte_px: int = 0             # 垃圾遮罩扩展像素
    
    # ==================== 输出参数（对齐 EZ-CorridorKey） ====================
    output_format: str = "png"            # png / exr / tga
    output_premultiplied: bool = False    # 是否预乘 alpha
    output_mask: bool = False             # 是否输出黑白遮罩
    output_debug: bool = False            # 是否输出中间调试图
    
    # EZ-CorridorKey 风格四通道输出
    output_fg: bool = True                # 输出直接前景 (Straight FG)
    output_matte: bool = True             # 输出线性 Alpha (Linear Matte)
    output_comp: bool = True              # 输出预乘合成 (Premultiplied Comp)
    output_processed: bool = True         # 输出处理后 RGBA
    
    # EXR 压缩选项
    exr_compression: str = "dwab"         # "dwab" | "piz" | "zip" | "none"
    
    # ==================== 色彩空间 ====================
    color_space: str = "sRGB"             # "sRGB" | "Rec709" | "ACES" | "Linear"
    input_is_linear: bool = False         # 输入是否为线性光
    
    # ==================== AI 参数 ====================
    use_ai: bool = True                   # 是否使用 AI 抠图
    ai_enhance: bool = False              # 是否启用 AI 增强模式
    ai_enhance_gamma: float = 0.5         # Gamma 提升
    ai_enhance_threshold: float = 0.3     # 阈值拉伸临界点
    ai_enhance_gain: float = 1.5          # 低值区域增益
    ai_enhance_sharpen: float = 0.3       # 边缘锐化强度
    
    # ==================== 性能参数 ====================
    max_resolution: Optional[int] = None  # 最大处理分辨率
    batch_size: int = 1                   # 批处理大小
    num_workers: int = 4                  # 并行 worker 数
    
    # ==================== 模型参数 ====================
    model_resolution: int = 2048          # 模型推理分辨率 1024 | 2048
    optimization_mode: str = "auto"       # "auto" | "speed" | "lowvram"
