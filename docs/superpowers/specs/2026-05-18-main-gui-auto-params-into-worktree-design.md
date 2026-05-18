# Main GUI And Auto Params Into Worktree Design

**背景**

当前主工程与 `feature/transparency-layered-fusion-v1` worktree 的 `HEAD` 相同，差异都在未提交工作区。

- 主工程领先点：`scripts/web_gui.py` 的 `auto_optimize` 入口、`src/matteflow/auto_params.py`、状态区展示“本次实际参数”、对应 GUI/自动调参测试与素材。
- worktree 领先点：`GVM` vendored runtime 恢复、`model_paths` 回退、`model_checker` 真可用校验、透明分层融合、`GVM` 旋涡内容补回、`despeckle` 收紧保护、`pipeline` 阶段差异日志，以及更多回归测试。

本次目标不是双向大合并，而是以 worktree 的抠图链路为基线，把主工程新增的 GUI 与自动调参能力安全回补进 worktree。

## Goal

将主工程中的 `auto_params` 与 GUI 自动优化能力合并到 worktree 中，同时保持 worktree 当前的 `GVM`、透明融合、`despeckle`、模型可用性与输出链路行为不回退。

## 非目标

- 不把 worktree 当前的抠图链路回退到主工程版本。
- 不在本次处理中重新设计 GUI 布局。
- 不引入新的自动调参策略范围，先保持主工程已有实现与测试语义。
- 不处理与本次范围无关的其它未提交主工程改动。

## 方案

### 1. 以 worktree 为运行基线

保留 worktree 的以下实现不变，避免破坏当前已验证的抠图能力：

- `src/matteflow/matte/gvm_matte.py`
- `src/matteflow/matte/hybrid_matte.py`
- `src/matteflow/refine/despeckle.py`
- `src/matteflow/refine/color_decontaminate.py`
- `src/matteflow/pipeline.py`
- `src/matteflow/utils/model_checker.py`
- `src/matteflow/utils/model_paths.py`

### 2. 从主工程回补 GUI 自动优化能力

将主工程中的以下能力迁入 worktree：

- 新增 `src/matteflow/auto_params.py`
- 在 `scripts/web_gui.py` 中接入：
  - `auto_optimize` 控件与默认值
  - `suggest_input_params()` / `apply_suggestion()`
  - 视频/序列帧取中间帧、图片直接分析
  - 状态区输出“自动优化摘要 + 本次实际参数”

### 3. 以“保留 worktree 行为”为合并准则解决冲突

`scripts/web_gui.py` 合并时，凡涉及以下行为，以 worktree 当前逻辑为准，再叠加主工程新增能力：

- 可用模型来源与展示逻辑
- `GVM` 作为推荐/默认模型的选择路径
- 输出目录解析
- 透明 PNG 下载与预览生成
- worktree 当前已接入的抠图链路参数字段

### 4. 测试对齐

需要把主工程下列测试能力补到 worktree：

- `tests/test_auto_params.py`
- `tests/test_web_gui_defaults.py` 中与 `auto_optimize`、状态区实际参数相关的新增断言

同时保留并继续通过 worktree 现有测试：

- `tests/test_despeckle_soft_alpha.py`
- `tests/test_gvm_alpha_resize.py`
- `tests/test_logging_instrumentation.py`
- `tests/test_model_checker_runtime.py`
- `tests/test_transparency_layered_fusion.py`

## 数据流

本次合并后的 GUI 处理链路应为：

1. 用户在 `web_gui` 选择素材与参数
2. 若启用 `auto_optimize`
3. GUI 调用 `suggest_input_params()` 生成建议
4. GUI 通过 `apply_suggestion()` 仅覆盖支持的 `MattingConfig` 字段
5. GUI 状态区显示：
   - 自动优化摘要
   - 本次实际参数
6. `MattingPipeline` 继续走 worktree 当前抠图链路：
   - `HybridMatte`
   - `refine`
   - `despeckle`
   - `decontaminate`
   - encode/output

## 风险与处理

### 风险 1：`web_gui.py` 同时被两边修改，冲突密集

处理：

- 不直接整文件覆盖。
- 以功能块为单位迁移：默认值、处理函数签名、自动优化接入、状态摘要、UI 控件。

### 风险 2：自动调参覆盖了 worktree 当前关键抠图参数

处理：

- 保持主工程已有 `apply_suggestion()` 只改有限字段。
- 用测试锁住 `screen_color`、`similarity`、`key_strength`、`transparency_preserve` 等核心自动覆盖项。
- 不让自动调参直接触碰 worktree 的 `GVM` 旋涡保护式 `despeckle` 行为。

### 风险 3：GUI 对齐后模型默认选择回退

处理：

- 保留 worktree 当前模型可用性判断。
- 用 GUI 默认值测试锁住“有 `gvm` 时优先默认 `gvm`”。

## 验证标准

满足以下条件即认为本次合并完成：

1. worktree 中存在 `src/matteflow/auto_params.py`
2. `scripts/web_gui.py` 支持 `auto_optimize`
3. GUI 状态区能展示自动优化摘要与实际参数
4. `gvm` 仍是可用时的默认推荐模型
5. worktree 现有 `GVM/透明融合/despeckle` 回归测试不回退
6. 主工程新增的 `auto_params` 与 GUI 默认值测试在 worktree 中通过
