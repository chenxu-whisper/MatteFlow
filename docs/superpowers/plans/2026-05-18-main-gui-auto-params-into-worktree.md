# Main GUI Auto Params Into Worktree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将主工程的 `auto_params` 与 GUI 自动优化能力合并到 worktree，同时保持 worktree 当前 GVM、透明融合与 `despeckle` 链路不回退。

**Architecture:** 以 worktree 现有抠图链路为运行基线，只迁移主工程中轻量的 `auto_params` 模块与 `web_gui.py` 上的自动优化接线。通过测试先行锁住 `auto_params` 建议逻辑、GUI 默认值、状态区实际参数展示，以及不影响已有 worktree 抠图能力。

**Tech Stack:** Python, pytest, Gradio, NumPy, OpenCV, Pillow

---

### Task 1: 锁住缺失的自动调参与 GUI 行为

**Files:**
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_auto_params.py`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_web_gui_defaults.py`

- [ ] **Step 1: 写入会失败的自动调参与 GUI 测试**
- [ ] **Step 2: 运行 `pytest tests/test_auto_params.py tests/test_web_gui_defaults.py -q`，确认因为 `auto_params` 缺失与 `web_gui` 未接线而失败**

### Task 2: 最小实现 auto_params 模块

**Files:**
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\auto_params.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_auto_params.py`

- [ ] **Step 1: 迁移主工程的 `AutoParamSuggestion`、代表帧采样、图片分析与 `apply_suggestion()`**
- [ ] **Step 2: 运行 `pytest tests/test_auto_params.py -q`，确认自动调参测试转绿**

### Task 3: 在 worktree GUI 中接入自动优化能力

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\scripts\web_gui.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_web_gui_defaults.py`

- [ ] **Step 1: 接入 `auto_optimize` 默认值、控件、`process_video()` 参数与状态区“本次实际参数”摘要**
- [ ] **Step 2: 运行 `pytest tests/test_web_gui_defaults.py -q`，确认 GUI 自动优化相关测试转绿**

### Task 4: 做联合回归并检查诊断

**Files:**
- Verify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_despeckle_soft_alpha.py`
- Verify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_gvm_alpha_resize.py`
- Verify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_logging_instrumentation.py`
- Verify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`
- Verify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_transparency_layered_fusion.py`

- [ ] **Step 1: 运行 `pytest tests/test_auto_params.py tests/test_web_gui_defaults.py tests/test_despeckle_soft_alpha.py tests/test_gvm_alpha_resize.py tests/test_logging_instrumentation.py tests/test_model_checker_runtime.py tests/test_transparency_layered_fusion.py -q`**
- [ ] **Step 2: 对最近编辑文件执行诊断检查并修复新增问题**
