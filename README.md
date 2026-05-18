# MatteFlow - 视频 / 图片 / 序列帧抠图工具

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MatteFlow 是一个高质量抠图工具，支持视频、图片和序列帧输入，适用于绿幕 / 黑底透明化处理，并集成多种 AI 和传统算法。

## ✨ 特性

- 🏆 **CorridorKey** - 物理分离算法，绿幕最佳效果
- 🎬 **BackgroundMattingV2** - 已知背景时效果最优
- 🤖 **AI 增强** - 传统 + AI 边缘细化
- 📐 **传统算法** - 快速色度键抠图
- 🎨 **Web UI** - 浏览器界面，实时预览
- 📦 **多输入支持** - 支持视频、图片、序列帧处理与导出

## 🚀 快速开始

### 安装

```bash
# 1. 克隆仓库（首次）
git clone <仓库地址>
cd MatteFlow

# 2. 创建虚拟环境并安装依赖
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 下载模型
python scripts/download_models.py --all

# 4. 启动 Web UI
python scripts/web_gui.py

# 5. 命令行处理示例
python -m matteflow --input assets/frame/test_frame_1.png --output ./output/image --mode green
python -m matteflow --input assets/video/test_green_2.mp4 --output ./output/video --mode green
python -m matteflow --input ./frames --output ./output/sequence --mode auto
```

示例测试资源位于 `assets/frame/` 与 `assets/video/`，也支持直接传入序列帧目录。

### 启动 Web UI

```bash
python scripts/web_gui.py
```

访问 http://localhost:7860

### 模型与运行时

- 模型权重默认位于项目根目录的 `models/`
- 已 vendored 的运行时代码位于 `src/matteflow/vendor/`
- 主工作区当前已切换 vendored 的入口优先包含 `GVM`

### 命令行使用

```bash
# 单图
python -m matteflow --input assets/frame/test_frame_1.png --output ./output/image --mode green

# 视频
python -m matteflow --input assets/video/test_green_2.mp4 --output ./output/video --mode green

# 序列帧目录
python -m matteflow --input ./frames --output ./output/sequence --mode auto
```

## 📖 使用指南

### 推荐参数组合

| 场景 | 背景模式 | 算法 | 参数 |
|------|----------|------|------|
| 纯色绿幕 | 绿幕 | CorridorKey | 默认 |
| 绿幕+白色物体 | 绿幕 | CorridorKey | 白色保护调高 |
| 有背景图 | 绿幕 | BackgroundMattingV2 | 上传背景图 |
| 黑底特效 | 黑底 | 传统算法 | 辉光保留 0.9 |
| 黑底粒子 | 黑底 | 传统算法 | 粒子增强 0.7 |

## 🏗️ 项目结构

```
MatteFlow/
├── src/matteflow/          # 核心代码
│   ├── config.py           # 配置
│   ├── pipeline.py         # 处理流程
│   ├── input/              # 输入解码
│   ├── analysis/           # 背景分析
│   ├── matte/              # 抠图算法
│   │   ├── green_screen_matte.py
│   │   ├── black_background_matte.py
│   │   ├── corridorkey_matte.py
│   │   ├── bgm2_matte.py
│   │   ├── rvm_matte.py
│   │   ├── birefnet_matte.py
│   │   └── rembg_matte.py
│   ├── refine/             # 边缘细化
│   ├── temporal/           # 时序稳定
│   ├── utils/              # 工具与模型检查
│   └── vendor/             # 内聚的第三方运行时代码
├── assets/                 # 示例图片与测试视频
├── scripts/                # 脚本
│   └── web_gui.py          # Web 界面
├── tests/                  # 测试
├── docs/                   # 文档
├── requirements.txt        # 依赖
└── README.md              # 说明
```

## 🛠️ 开发

### 运行测试

```bash
pytest tests/
```

### 添加新算法

1. 在 `src/matteflow/matte/` 创建新文件
2. 继承基类，实现 `generate()` 方法
3. 在 `hybrid_matte.py` 注册

## 📄 许可证

MIT License

## 🙏 致谢

- [CorridorKey](https://github.com/nikopueringer/CorridorKey) by Corridor Digital
- [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- [RVM](https://github.com/PeterL1n/RobustVideoMatting)
