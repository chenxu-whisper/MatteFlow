# MatteFlow

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MatteFlow 是一个面向视频、图片和序列帧的抠图工具，支持绿幕、黑底透明化、多种 AI 模型与传统算法，并提供命令行与 Web GUI 两种使用方式。

## 特性

- 支持视频、单图和序列帧输入
- 支持绿幕、黑底和自动背景模式
- 集成传统算法与多种 AI 抠图后端
- 提供 Web GUI 预览、队列处理和导出能力
- 提供命令行入口，适合批处理与自动化场景

## 快速开始

### 安装

```bash
git clone https://github.com/chenxu-whisper/MatteFlow.git
cd MatteFlow

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 下载模型

```bash
python scripts/download_models.py --all
```

### 启动 Web GUI

```bash
python scripts/web_gui.py
```

启动后访问 `http://localhost:7860`。

### 命令行处理

```bash
# 单图
python -m matteflow --input assets/frame/test_frame_1.png --output ./output/image --mode green

# 视频
python -m matteflow --input assets/video/test_green_2.mp4 --output ./output/video --mode green

# 序列帧目录
python -m matteflow --input ./frames --output ./output/sequence --mode auto
```

### 预览缓存清理验证

```bash
python -m matteflow verify-preview-cleanup
```

这个子命令会构造一组本地临时数据，验证 Gradio 预览缓存与 `downloads/` 目录的纯时间驱动清理逻辑是否按预期生效。

## 模型与运行时

- 模型权重默认位于项目根目录的 `models/`
- 已 vendored 的第三方运行时代码位于 `src/matteflow/vendor/`
- 示例输入资源位于 `assets/frame/` 与 `assets/video/`

## 项目结构

```text
MatteFlow/
├── assets/                 # 示例图片与视频
├── scripts/                # 辅助脚本
├── src/matteflow/          # 核心实现与 CLI
│   ├── analysis/           # 背景分析与自动模式辅助判断
│   ├── input/              # 视频 / 图片 / 序列帧输入识别与解码
│   ├── matte/              # 传统算法、多种 AI 抠图后端与融合调度
│   ├── output/             # RGBA、序列帧、预览等结果编码与导出
│   ├── refine/             # 边缘细化、去溢色、去噪点等后处理
│   ├── temporal/           # 视频时序稳定，降低 Alpha 闪烁与抖动
│   ├── utils/              # 模型检查、模型下载、路径与兼容工具
│   ├── vendor/             # vendored 第三方运行时代码与包装层
│   ├── config.py           # 质量模式、背景模式与运行参数定义
│   ├── pipeline.py         # 主处理流水线，串联解码、抠图、后处理与导出
│   ├── service.py          # 面向 GUI / 队列的任务化服务封装
│   ├── job_queue.py        # GUI 任务排队、状态管理与历史记录
│   ├── job_worker.py       # 队列任务执行器与串行消费逻辑
│   ├── diagnostics.py      # 统一诊断报告与错误分类
│   ├── errors.py           # 自定义异常类型定义
│   ├── cli_app.py          # argparse CLI 主实现与子命令分发
│   ├── verify_preview_cleanup.py  # 预览缓存清理验证子命令
│   └── __main__.py         # `python -m matteflow` 入口
├── tests/core/             # 核心链路测试
├── tests/web_gui/          # Web GUI 测试
├── pyproject.toml          # 项目配置
├── requirements.txt        # 依赖清单
└── README.md
```

## 开发

### 运行测试

```bash
pytest tests
```

也可以分别运行：

```bash
pytest tests/core
pytest tests/web_gui
```

### 添加新算法

1. 在 `src/matteflow/matte/` 中新增实现文件
2. 按现有接口接入配置与推理逻辑
3. 在组合调度层中注册并补对应测试

## 许可证

MIT License

## 致谢

- [CorridorKey](https://github.com/nikopueringer/CorridorKey) by Corridor Digital
- [BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2)
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- [RVM](https://github.com/PeterL1n/RobustVideoMatting)
