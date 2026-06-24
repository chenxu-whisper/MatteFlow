# 抠图质量系统设计

日期：2026-06-24

## 概要

MatteFlow 需要增加一层质量决策能力，让 MatAnyone2、SAM2 和 BiRefNet 变得可评估、可选择、可回归。目标不是再增加一条模型优先级规则，而是生成多路候选 matte，按区域评估它们的可靠性，为每个区域选择或融合最合适的候选结果，并记录足够证据，用于后续质量对比和回归判断。

本设计聚焦当前视觉 review 中暴露出的质量问题：

- 模型分歧：base、GVM 和融合输出可能互相矛盾，导致主体漏抠或背景误保留。
- 软边和发丝细节损失：alpha 过渡可能变硬、断裂，或被后续阶段过度清理。
- 残留和污染：绿色或灰色边缘残留仍可能保留，颜色去污染也可能过度处理。

第一阶段实现应使用规则型质量指标和确定性选择策略。接口需要为后续 learned quality evaluator 预留空间。

## 目标

1. 通过统一候选输出协议，让 MatAnyone2、SAM2 和 BiRefNet 可以被横向比较。
2. 为每个候选结果产出帧级和区域级质量信号。
3. 按区域选择或融合候选结果，而不是整帧只选择一个模型。
4. 在现有 processing report 流程中记录模型决策、质量信号和视觉证据。
5. 增加固定困难样本集的质量回归工作流，用于比较质量变化。
6. 在新质量系统逐步上线期间，保留现有绿幕、黑底和 fallback 行为。

## 非目标

1. 第一阶段不训练或微调新模型。
2. 不在一次改动中替换整个 `HybridMatte` 实现。
3. 不建设云服务或分布式推理系统。
4. 不要求在第一版可用回归套件之前为所有样本提供人工标注。
5. 不让 SAM2 成为所有运行的强制依赖。SAM2 应在存在目标引导，或 selector 需要语义约束时使用。

## 当前上下文

当前代码库已经具备一些关键基础：

- `src/matteflow/matte/hybrid_matte.py` 协调传统 matte、AI 模型、fallback 逻辑和绿幕专项融合。
- `src/matteflow/matte/fusion_quality_gate.py` 已经有初步的区域感知融合机制。
- `src/matteflow/analysis/region_ownership.py` 可以分类 subject、hair edge、transparent effect、luminous prop、uncertain edge 和 background residue 区域。
- `src/matteflow/analysis/alpha_quality.py` 可以计算轻量 alpha 质量信号。
- `src/matteflow/analysis/p0_quality.py` 可以分类高层质量风险。
- `src/matteflow/reporting/processing_report.py` 已经能输出结构化处理报告。
- `tests/core/test_quality_regression.py` 为基于报告的回归检查提供了起点。

当前缺口是：候选生成、质量评估、候选选择和回归机制还没有围绕 MatAnyone2、SAM2、BiRefNet 形成统一系统。

## 推荐架构

新增一层质量决策能力，承担四类职责：

1. 候选生成
2. 候选评估
3. 区域级选择
4. 回归证据生成

数据流如下：

```text
已解码帧
  |
  v
背景分析 / 可选用户引导
  |
  v
候选生成器
  |-- MatAnyone2 候选
  |-- SAM2-guided 候选
  |-- BiRefNet 候选
  |-- 现有 traditional/base 候选
  |
  v
质量评估器
  |
  v
区域级选择器
  |
  v
选中的 alpha 序列 + 决策诊断
  |
  v
现有 refine / despeckle / repair / temporal / decontaminate / encode 阶段
  |
  v
处理报告 + 回归证据产物
```

## 模块设计

### 1. 候选结果协议

在 `src/matteflow/matte/candidates/` 下引入候选结果协议。这里应使用 package，而不是单个大文件，因为真实模型适配器会独立增长。

每个候选生成器返回：

```python
@dataclass(frozen=True)
class MatteCandidate:
    name: str
    alpha: np.ndarray
    confidence: np.ndarray | None
    source: str
    runtime_ms: float
    diagnostics: dict[str, Any]
```

序列场景返回：

```python
@dataclass(frozen=True)
class MatteCandidateSequence:
    name: str
    alphas: list[np.ndarray]
    confidences: list[np.ndarray | None]
    source: str
    runtime_ms: float
    diagnostics: dict[str, Any]
```

规则：

- `alpha` 必须是 float32，裁剪到 `[0.0, 1.0]`，并且与帧的高度和宽度一致。
- `confidence` 在第一阶段是可选字段。如果缺失，评估器从质量指标和区域归属中推导置信度。
- `source` 必须稳定，例如 `matanyone2`、`sam2_guided`、`birefnet`、`traditional_green` 或 `traditional_black`。
- `diagnostics` 必须可 JSON 序列化。

### 2. 候选生成器

新增 wrapper 来适配现有模型模块。第一阶段先不改动模型内部推理代码：

- `MatAnyone2CandidateGenerator`
- `SAM2GuidedCandidateGenerator`
- `BiRefNetCandidateGenerator`
- `TraditionalCandidateGenerator`

wrapper 负责处理模型可用性、取消检查、进度回调、输出归一化和耗时统计。如果模型不可用，生成器应返回结构化的 skipped 结果；只有用户显式指定该模型时，才应抛出错误。

SAM2 应按“引导优先”处理：

- 如果存在首帧 mask、box 或 point prompt，SAM2 产出目标约束候选结果。
- 如果没有引导信息，第一阶段默认可以跳过 SAM2，避免把语义分割的不确定性误当成 alpha 置信度。
- 即使 SAM2 不是最终 alpha 来源，selector 也可以把 SAM2 输出作为区域约束使用。

### 3. 质量评估器

新增 `src/matteflow/evaluation/matte_quality.py`。

评估器消费 frames、candidate alphas、可选 confidences 和 region ownership，返回候选级和区域级分数。

第一阶段质量信号：

- `subject_coverage`：已知主体类区域是否保留足够 alpha。
- `soft_edge_continuity`：软 alpha 带是否连续，避免断裂或阶梯状边缘。
- `hair_edge_preservation`：发丝/羽毛边缘区域是否避免低 alpha 坍缩。
- `background_cleanliness`：背景残留区域是否接近零 alpha。
- `transparent_effect_preservation`：透明或发光区域是否避免被抹掉。
- `spill_risk`：绿色/灰色污染边缘是否仍可能可见。
- `temporal_stability`：可比较区域的帧间 alpha 变化。
- `model_disagreement`：候选结果在重要区域中与其他候选差异的强度。

评估器输出：

```python
@dataclass(frozen=True)
class CandidateQuality:
    candidate_name: str
    frame_index: int
    overall_score: float
    region_scores: dict[str, float]
    signals: dict[str, float | int | str]
```

分数方向必须一致：越高越好。风险信号可以作为独立字段记录。

### 4. 区域级选择器

新增 `src/matteflow/matte/quality_selector.py`。

selector 接收 candidates、quality scores 和 region ownership，返回选中的 alpha 以及诊断信息。

选择原则：

- 主体核心：优先选择主体覆盖强、时序稳定的候选。
- 发丝和不确定边缘：优先选择软边连续性更好、低 alpha 坍缩更少的候选。
- 透明特效和发光道具：优先选择能保留低到中等 alpha 的候选，避免硬裁剪。
- 背景残留：优先选择 alpha 更低、spill 风险更低的候选。
- SAM2-guided mask：可用时作为语义约束，不应直接作为硬 alpha 替换，除非它的质量分数胜出。
- 现有 traditional green/black matte 继续作为候选，尤其用于绿幕基础结构和黑底透明特效。

selector 必须输出诊断信息：

```python
@dataclass(frozen=True)
class SelectionDecision:
    frame_index: int
    selected_by_region: dict[str, str]
    region_scores: dict[str, dict[str, float]]
    rejected_takeovers: dict[str, int]
    warnings: list[str]
```

这里应扩展或复用 `FusionQualityGate` 的概念，避免重复建设一套无关的融合逻辑。

### 5. 流水线集成

新增配置字段：

```python
quality_selection_enable: bool = False
quality_candidate_models: tuple[str, ...] = ("matanyone2", "sam2", "birefnet", "traditional")
quality_selection_mode: str = "region"
```

初始上线方式：

- 默认保持当前行为。
- 当 `quality_selection_enable=True` 时，`HybridMatte` 将 matte 生成委托给新的 `QualityDrivenMatte` 协调器。`HybridMatte` 仍作为 pipeline 的公开集成点，`QualityDrivenMatte` 负责候选生成、评估和选择。
- 选中的 alpha 序列继续进入现有 refine、despeckle、repair、temporal stabilization、decontamination 和 encoding 阶段。

这样可以避免影响当前 CLI 和 GUI 默认行为，同时允许定向测试新质量系统。

### 6. 报告

扩展 processing report，增加：

- 候选模型列表和可用性状态。
- 各模型运行耗时。
- 各模型质量摘要。
- 各区域选中模型统计。
- 各风险类别下最差帧。
- 可选 debug 产物路径。

debug 产物应包括：

- 候选 alpha contact sheet。
- 区域选择 overlay。
- 最终选择结果与各候选之间的 alpha difference heatmap。
- 高风险区域的局部放大对比图。

现有 `output_debug` 字段可以控制是否写出图片。即使图片 debug 输出关闭，也应写出 JSON 摘要。

### 7. 回归套件

在现有 evaluation 结构下增加固定困难样本回归工作流。

推荐目录结构：

```text
tests/fixtures/matting_quality/
  manifest.json
  green_screen/
  black_background/
  video_short/
```

manifest 应定义：

- 输入路径
- 背景模式
- 质量模式
- 启用的候选模型
- 预期风险上限
- 可选 baseline report 路径
- 可选 ROI，用于生成局部放大产物

回归输出：

- `quality_summary.json`
- `candidate_decisions.json`
- `p0_risks.json`
- `contact_sheet.png` when debug output is enabled
- `diff_heatmaps/` when debug output is enabled

自动化门禁：

- 任一 P0 风险类别不得超过配置容忍度发生回退。
- 背景残留和发丝边缘损失风险不得超过样本级上限。
- 相同输入和配置下，候选选择必须是确定性的。
- 可选模型缺失时，应根据 manifest 标记为 skipped 或 degraded，而不是变成无关失败。

## 错误处理

质量系统应区分以下情况：

- 用户指定的模型不可用：快速失败，并给出明确的模型可用性错误。
- 可选候选模型不可用：记录 skipped candidate，并继续运行。
- 候选输出 shape 不匹配：质量选择阶段失败。
- 候选 alpha 包含非法值：有限值可清理；如果仍存在 NaN 或 infinite，则失败。
- SAM2 引导信息缺失：第一阶段跳过 SAM2-guided candidate，并记录原因。
- 没有 AI 候选成功：回退到现有传统行为，并记录 degraded mode。

## 测试策略

单元测试：

- 候选输出归一化。
- 质量评估器的分数方向和信号提取。
- selector 在合成 subject、hair、transparent-effect 和 background-residue mask 上的行为。
- 候选分数相等时的确定性选择。
- 缺失模型处理。

集成测试：

- `quality_selection_enable=False` 时 pipeline 保持现有行为。
- `quality_selection_enable=True` 时 pipeline 可使用 fake candidate generators 跑通。
- processing report 包含候选决策和质量摘要。
- regression evaluator 能识别阈值回退和 baseline 回退。

视觉/人工测试：

- 为当前绿幕诊断样本生成 contact sheets。
- 对比模型分歧、软边/发丝、残留/污染区域的局部放大图。

## 上线计划

阶段 1：基础设施，不改变默认行为

- 增加候选协议。
- 增加 fake 或轻量候选测试。
- 增加质量评估器和 selector。
- 增加 report schema 扩展。
- 保持 `quality_selection_enable=False` 为默认值。

阶段 2：真实模型 wrapper

- 将 MatAnyone2、BiRefNet 和 traditional matte 包装为候选。
- 在存在引导信息时增加 SAM2-guided candidate。
- 在短样本上运行质量选择。

阶段 3：回归与 GUI 可见性

- 增加困难样本 manifest。
- 生成 contact sheets 和局部放大产物。
- 在 GUI report view 中展示候选决策。

阶段 4：质量调优

- 基于困难样本集调优区域分数。
- 判断 MatAnyone2 是否应成为高质量视频默认路径。
- 基于剩余失败情况判断是否需要 learned quality evaluator。

## 本 Spec 已明确的决策

- 第一阶段使用规则型质量评分，而不是 learned evaluator。
- 在回归证据足够强之前，新路径通过配置显式开启。
- SAM2 是 guidance-first，不应在没有 prompt 的情况下被当作通用 alpha 生成器。
- 选择发生在区域级，而不是整帧级。
- 现有 traditional matte 保持为候选，而不是被丢弃的 fallback。

## 成功标准

满足以下条件时，说明本设计成功：

1. 单次运行可以在可用时产出 MatAnyone2、BiRefNet、SAM2-guided mode 和 traditional matte 的候选结果。
2. 报告能解释 subject、hair edge、transparent effect、uncertain edge 和 background residue 区域分别由哪个候选胜出。
3. 回归运行能在人工 review 之前发现更差的发丝边缘损失、背景残留、透明特效损失或时序不稳定。
4. debug 产物能直接暴露模型分歧和边缘失败，不需要反复运行临时脚本。
5. 在质量系统验证期间，当前默认行为仍然可用。
