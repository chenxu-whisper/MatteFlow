# MatteFlow - 专业视频抠图工具

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MatteFlow 是一个高质量视频抠图工具，支持绿幕/黑底视频透明化处理，集成多种 AI 和传统算法。

## ✨ 特性

- 🏆 **CorridorKey** - 物理分离算法，绿幕最佳效果
- 🎬 **BackgroundMattingV2** - 已知背景时效果最优
- 🤖 **AI 增强** - 传统 + AI 边缘细化
- 📐 **传统算法** - 快速色度键抠图
- 🎨 **Web UI** - 浏览器界面，实时预览
- 📦 **批量处理** - 支持视频序列帧输出

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/MatteFlow.git
cd MatteFlow

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 启动 Web UI

```bash
python scripts/web_gui.py
```

访问 http://localhost:7860

### 命令行使用

```bash
python -m matteflow --input video.mp4 --output ./output --mode green
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
│   └── output/             # 输出编码
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
