# MatteFlow

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MatteFlow 是一个面向视频、图片和序列帧的抠图工具，支持绿幕、黑底透明化、多种 AI 模型与传统算法，并提供命令行与 Web GUI 两种使用方式。

## 特性

- 支持视频、单图和序列帧输入
- 支持绿幕、黑底和自动背景模式
- 集成传统算法与多种 AI 抠图后端
- 支持可选质量选择系统，按区域评估 traditional、MatAnyone2、BiRefNet、SAM2-guided 等候选结果
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

# 推荐最高质量路径：启用 high、质量选择和 BiRefNet 候选自动加载
python -m matteflow --input assets/frame/test_frame_1.png --output ./output/quality --mode green --quality-preset best
```

CLI 默认使用 `--quality-preset default`，保持兼容的 `standard` 质量模式，不会自动启用质量选择或 BiRefNet 候选自动加载。需要最高质量路径时，显式使用 `--quality-preset best`；它会统一启用 `high`、质量选择和 BiRefNet 候选自动加载。质量选择会生成多个候选 matte，按主体、发丝边缘、透明效果、背景残留等区域评分，并将决策摘要写入 `processing_report.json` 的 `quality_selection` 字段。也可以继续使用 `--quality-selection` 与 `--quality-birefnet-auto-load` 细粒度控制。Web GUI 和 Service API 也提供 `quality_selection_enable` 与 `quality_birefnet_auto_load` 开关。

### 黑底增强与处理报告

黑底模式使用统一的 `BlackEffectEnhancer` 处理烟雾、辉光、粒子和主体暗边修复。主黑底流程只生成保守的结构 baseline，特效透明度由增强器统一抬升，避免旧的亮度/颜色规则重复放大黑底噪声。

`processing_report.json` 当前 schema version 为 `2`，包含以下黑底与质量诊断字段：

- `black_effect_enhancement`：黑底增强多帧聚合诊断，包括 `frames`、`smoke_pixels`、`glow_pixels`、`particle_pixels`、`subject_edge_pixels`、`black_residue_suppressed_pixels` 和 `mean_alpha_delta`
- `quality_selection`：候选数量、选中模型统计、跳过候选和逐区域选择决策
- `region_supervision`：区域弱监督统计和失败项
- `edge_reconstruction`：边缘重建变更像素、保护像素和平均 alpha 变化

### 质量回归

```bash
python -m matteflow quality-regression --reports ./output/quality --output-json ./output/quality_regression.json
```

质量回归会读取目录下的 `processing_report.json` 并生成聚合报告，用于检查核心质量指标、候选数量和报告结构是否退化。

### 预览缓存清理验证

```bash
python -m matteflow verify-preview-cleanup
```

这个子命令会构造一组本地临时数据，验证 Gradio 预览缓存与 `downloads/` 目录的纯时间驱动清理逻辑是否按预期生效。

## 模型与运行时

- 模型权重默认位于项目根目录的 `models/`，该目录用于本地大模型文件，不纳入版本控制
- 已 vendored 的第三方运行时代码位于 `src/matteflow/vendor/`，其中源码目录即使名为 `models` 也必须纳入版本控制
- GVM、MatAnyone2、CorridorKey 等运行时通过 vendored wrapper 延迟导入；`rembg` 只有在 `remove` symbol 可导入时才会作为可用后端
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
├── tests/fixtures/         # 核心测试所需轻量 fixture
├── pyproject.toml          # 项目配置
├── requirements.txt        # 依赖清单
└── README.md
```

## 开发

### 运行测试

```bash
python -m pytest tests -q
```

质量回归可以单独运行：

```bash
python -m matteflow quality-regression --reports ./output/quality --output-json ./output/quality_regression.json
```

发布前建议同时执行 Ruff 门禁：

```bash
python -m ruff check src scripts tests
```

质量回归支持对视频时序与 Alpha 细节设置更严格门禁：

```bash
python -m matteflow quality-regression \
  --reports ./output/quality \
  --max-edge-temporal-flicker 0.08 \
  --max-transparent-temporal-flicker 0.08 \
  --max-hair-low-alpha-ratio 0.30 \
  --max-effect-low-alpha-ratio 0.30
```

Manifest 级质量回归由 `MattingQualityRegressionRunner` 使用 `tests/fixtures/matting_quality/manifest.json` 驱动，适合在测试或脚本中批量跑样本。它还支持强监督 Alpha 和视频候选贡献校验：

- `expected_alpha_path`：GT Alpha 图
- `alpha_mae_ceiling` / `alpha_mse_ceiling`：预测 Alpha 与 GT 的误差上限
- `required_temporal_models`：要求 MatAnyone2、SAM2 等视频候选真实参与质量选择，未贡献时样本失败
- `region_expectations`：要求指定区域出现或达到最低比例，未满足时样本失败

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
