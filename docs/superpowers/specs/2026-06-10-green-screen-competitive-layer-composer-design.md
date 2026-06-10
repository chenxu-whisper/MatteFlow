# Green Screen Competitive Layer Composer Design

**Goal:** 将当前 GVM 绿幕融合从“主体层和效果层直接取最大值”的叠加流程，升级为“主体 / 效果 / 背景三层竞争式合成”流程，解决 `test_frame_3.jpg` 中主体、闪电、蓝色背景同时失真的问题，并避免继续通过局部阈值调参修补效果。

## Background

当前 GVM 绿幕路径已经具备以下能力：

- `GreenScreenMatte` 生成传统 chroma key `base_alpha`
- GVM 生成 AI subject alpha
- BiRefNet 在 GVM 模式下懒加载，生成 semantic subject alpha
- 主体补洞逻辑恢复紫色毛绒主体低 alpha 区
- luminous effect reconstruction 恢复白色 / 蓝白 / 黄白闪电核心
- cyan halo reconstruction 恢复靠近闪电核心的青色光晕

这些能力分别能在局部测试中通过，但真实 GUI 视觉结果仍不好。最新分层诊断已经导出到：

- `.superpowers/diagnostics/semantic_layer_debug/contact_sheet.png`
- `.superpowers/diagnostics/semantic_layer_debug/summary.json`

关键证据：

- GUI 单帧 GVM 路径中 `last_fallback_quality_metrics` 为 `null`，所以依赖 `score_blocked=True` 的主体 rescue 链路没有真实触发。
- BiRefNet semantic alpha 会把部分闪电 / 亮带区域识别成主体。
- 当前 `solid_total = max(subject layers...)` 后再与 `effect_total` 软融合，缺少主体和效果之间的互斥归属判断。
- 闪电区域同时拥有较强 subject evidence 和 effect evidence，最终被合成为过实、过厚的 alpha。
- 蓝色背景虽然整体被压住，但仍存在局部高 alpha 误保留，需要作为背景层参与竞争，而不是只靠效果层阈制。

## Problem Statement

现有流程的核心问题不是单个阈值不准，而是像素归属没有建模。

当前流程隐含假设：

- subject layer 负责实体主体
- effect layer 负责透明效果
- 最终 `max` 或 `solid + effect * (1 - solid)` 就能得到正确 alpha

这个假设在主体与透明效果空间重叠时失效：

- BiRefNet / GVM 可能把亮带效果当成主体。
- subject gate 会抑制传统 effect branch，但 luminous reconstruction 又会补回一部分效果。
- 最终合成无法判断“这个高 alpha 是主体毛发，还是发光闪电”。
- 背景层没有独立 ownership，只能通过低 alpha 或 veto 被动参与。

因此下一步需要新增一层 ownership / competition，而不是继续给 subject 或 effect 分支添加局部规则。

## Scope

本设计覆盖 GVM 绿幕路径中的单帧和序列融合逻辑。

本阶段要做：

- 新增竞争式 layer composer。
- 在 composer 内显式建模 `subject_candidate`、`effect_candidate`、`background_candidate`。
- 生成 `subject_ownership`、`effect_ownership`、`background_ownership`。
- 使用 ownership 重新合成 `final_alpha`。
- 输出可选 debug layers，便于继续定位视觉问题。
- 用真实 `test_frame_3.jpg` 建立回归测试。

本阶段不做：

- 不新增用户可调参数来“调效果”。
- 不引入新的 GUI 控件。
- 不替换 GVM / BiRefNet 模型权重。
- 不引入 SAM2 交互式点选流程。
- 不重构完整 `MattingPipeline`。
- 不删除现有 subject / effect 生成能力。

## Design Principles

- 先判断像素归属，再合成 alpha。
- 主体、效果、背景互相竞争，而不是互相无条件叠加。
- 新能力以结构化 evidence 和 ownership 表达，不以散落阈值表达。
- 真实样本驱动测试，但测试目标是层语义，不是单点像素魔法数。
- 旧路径保留为 fallback，首阶段只替换 GVM 绿幕融合核心。

## Proposed Architecture

### 1. Candidate Builder

Candidate Builder 负责收集已有分支输出，并整理成可比较的候选层。

输入：

- `frame`
- `base_alpha`
- `gvm_alpha`
- `semantic_subject_alpha`
- 当前 subject layers
- 当前 effect layers

输出：

- `subject_candidate.alpha`
- `subject_candidate.confidence`
- `effect_candidate.alpha`
- `effect_candidate.confidence`
- `background_candidate.alpha`
- `background_candidate.confidence`
- `evidence_maps`

`subject_candidate` 来源：

- GVM alpha
- BiRefNet semantic alpha
- solid foreground mask
- subject integrity recovery
- non-screen subject evidence

`effect_candidate` 来源：

- existing green screen effect layer
- luminous effect reconstruction
- cyan halo reconstruction
- bright / low-chroma / lightning-like evidence

`background_candidate` 来源：

- green-screen similarity evidence
- low base alpha + low AI alpha + low semantic alpha evidence
- far-from-effect-core cyan / blue background evidence

### 2. Evidence Maps

Evidence maps 是竞争式 composer 的中间语言，避免后续逻辑直接散落在颜色判断里。

首阶段至少生成：

- `subject_evidence`
- `semantic_subject_evidence`
- `effect_evidence`
- `luminous_core_evidence`
- `halo_evidence`
- `background_evidence`
- `effect_over_subject_evidence`

`effect_over_subject_evidence` 专门处理本次根因：

- 当 BiRefNet / GVM 把闪电识别成 subject，但 luminous / halo evidence 更强时，该像素应归属 effect。
- 这不是把 subject alpha 降低，而是把 ownership 从 subject 转给 effect。

### 3. Competitive Ownership

Competitive Ownership 负责为每个像素计算三层归属。

输出：

- `subject_ownership`
- `effect_ownership`
- `background_ownership`

约束：

- 三个 ownership 均在 `[0, 1]`。
- `subject_ownership + effect_ownership + background_ownership` 不要求严格等于 1，但同一像素不能同时出现高 subject ownership 和高 effect ownership。
- effect ownership 在 luminous / halo evidence 强且 subject evidence 主要来自 semantic bleed 时优先。
- subject ownership 在紫色毛绒主体、GVM 高置信主体、BiRefNet 主体核心中优先。
- background ownership 在远离闪电核心的蓝色 / 青色云雾中优先。

### 4. Composer

Composer 使用 ownership 合成最终 alpha。

合成语义：

- `subject_alpha_out = subject_candidate.alpha * subject_ownership`
- `effect_alpha_out = effect_candidate.alpha * effect_ownership`
- `background_alpha_out = 0`
- `final_alpha = subject_alpha_out + effect_alpha_out * (1 - subject_alpha_out)`

重要变化：

- effect 不再被已成型的 `solid_total` 无条件吞掉。
- subject 也不能无条件覆盖强 effect evidence 区域。
- background 通过 ownership 直接压制误保留，而不是后置清理。

### 5. Debug Outputs

当 `output_debug` 或诊断脚本启用时，导出：

- `subject_candidate_alpha`
- `effect_candidate_alpha`
- `background_confidence`
- `subject_ownership`
- `effect_ownership`
- `background_ownership`
- `subject_alpha_out`
- `effect_alpha_out`
- `final_alpha`

这些调试层用于判断失败属于：

- candidate 不准
- evidence 不准
- ownership 竞争不准
- final composition 不准

## Integration Plan Shape

首阶段优先保持外部接口稳定。

建议模块位置：

- `src/matteflow/matte/green_screen_layer_composer.py`

建议核心数据结构：

- `LayerCandidate`
- `LayerEvidence`
- `LayerOwnership`
- `CompetitiveLayerResult`
- `GreenScreenCompetitiveLayerComposer`

`HybridMatte._merge_green_screen_effects()` 继续作为入口，但内部在 GVM 绿幕模式下调用 composer。

旧融合逻辑保留为 fallback：

- composer 不可用时回退旧 `_soft_fuse_layers`
- 非 GVM 绿幕路径暂不强制切换
- 缺少 semantic alpha 时仍可用 GVM + base + effect evidence 运行

## Testing

新增测试以真实 `test_frame_3.jpg` 为主，不只验证局部函数。

至少覆盖：

- `test_green_screen_competitive_composer_keeps_subject_complete`
  - 紫色主体低 base alpha 区最终 alpha 保持足够高。
  - 主体内部低 alpha 比例低于当前路径。

- `test_green_screen_competitive_composer_routes_luminous_bands_to_effect`
  - 闪电核心和带状亮边拥有高 `effect_ownership`。
  - 闪电区域不被 `subject_ownership` 主导。

- `test_green_screen_competitive_composer_suppresses_far_blue_background`
  - 远离 luminous core 的蓝色 / 青色背景保持低 alpha。
  - 背景区域 `background_ownership` 高于 `effect_ownership`。

- `test_green_screen_competitive_composer_exports_debug_layers`
  - 诊断可导出 candidate / ownership / final layers。

旧测试继续保留：

- bright cool effect 负样本保护
- cool gray transition 负样本保护
- purple subject recovery
- blue background cloud negative
- luminous lightning reconstruction
- cyan halo near lightning
- BiRefNet list output compatibility

## Success Criteria

- `test_frame_3.jpg` 在 GVM 绿幕路径中进入 competitive composer。
- `summary.json` 或 debug export 能显示 subject/effect/background ownership。
- 紫色主体低 base alpha 区不再大面积破洞。
- 闪电核心和 halo 不再被 subject layer 合成成实心主体。
- 远端蓝色 / 青色背景不被 halo 或 semantic bleed 大面积抬起。
- 不新增 GUI 调参项。
- 现有绿幕融合和 BiRefNet 回归测试继续通过。

## Risks And Mitigations

- 风险：竞争式 ownership 初版仍可能把部分主体边缘分给 effect。
- 缓解：先以 debug ownership 图验证归属，再调整 evidence 来源，不直接改 final alpha。

- 风险：真实视觉改善依赖多个候选层质量，单个测试可能不足。
- 缓解：用 `test_frame_3.jpg` 建立主体、闪电、背景三类区域断言，并保留 debug export 便于继续扩样本。

- 风险：新增 composer 让 `HybridMatte` 更复杂。
- 缓解：把竞争逻辑放进独立模块，`HybridMatte` 只负责收集输入和调用。

- 风险：BiRefNet semantic bleed 仍会污染 subject candidate。
- 缓解：用 `effect_over_subject_evidence` 在 ownership 层处理 bleed，而不是关闭 semantic subject 能力。

## Stage-One Decisions

- Debug layers 首阶段只服务测试和诊断脚本，不接入 GUI 下载包。
- SAM2 自动提示不进入首阶段实现，只作为后续 subject ownership 增强方向。
- Composer 首阶段只接入 GVM 绿幕路径，不推广到非 GVM 绿幕路径。
- GUI 首阶段不新增控制项，用户仍通过现有 `GVM` 模式触发新流程。
