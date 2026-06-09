# Diagnose GVM Input Priority Design

**Goal:** 让 `scripts/diagnose_gvm_fusion.py` 生成的 `summary.json` 在 `inputs[*]` 总览层优先展示带 `effect-risk` 的输入，帮助诊断先聚焦真实高风险素材，同时不改变 `samples[*]` 和单个 input 内 `priority_debug_crops` 的现有语义。

## Background

当前诊断脚本已经具备以下输入级结论能力：

- `priority_debug_crops`
- `top_debug_focus`
- `dominant_effect_risk`
- 带 effect-risk 后缀的 `dominant_decision_reason`
- effect-risk 驱动的 `recommended_action`

这些字段已经能表达“为什么这个 input 值得关注”，但 `summary.json` 的 `inputs[*]` 仍保持原始分组顺序，没有把 effect-risk 输入自动顶到前面。结果是：

- 已知真实负样本与普通输入混排
- 报告阅读顺序不能直接反映风险优先级
- 后续 UI 或报表层若要高亮风险输入，只能重复实现排序规则

## Scope

本次设计只覆盖 `inputs[*]` 总览顺序。

明确不做的事情：

- 不调整 `samples[*]` 的输出顺序
- 不调整单个 input 内 `priority_debug_crops` 的排序
- 不新增或修改 sample 级诊断字段
- 不修改现有 `recommended_action` 和 `dominant_decision_reason` 的生成逻辑
- 不引入新的磁盘产物或 crop 导出规则

## Design

### Sorting Target

排序只发生在 `_build_summary_payload()` 组装出的 `inputs` 列表返回前。

这样可以保证：

- input 级总览更符合人工排查顺序
- 现有 sample 级与 crop 级调试语义保持不变
- 变更范围最小，测试影响面可控

### Sorting Strategy

新增一个独立 helper 负责生成 input 排序 key，并统一用于 `inputs.sort(...)`。

排序优先级从高到低如下：

1. `dominant_effect_risk`
   - `bright_cool_effect_risk`
   - `cool_gray_transition_effect_risk`
   - `None`
2. `dominant_decision_reason`
   - `fallback_blocked_by_effect_damage*`
   - `fallback_blocked_by_low_weighted_score*`
   - `sequence_gvm_retained_without_fallback_evaluation*`
   - `single_frame_or_sequence_gvm_retained*`
3. `priority_debug_crops[0].debug_crop_peak_abs_diff`
4. `sample_count`
5. `input`

其中前两层负责表达“风险类别”和“决策严重度”，后两层负责在同类输入之间保持稳定且更可解释的展示顺序。

### Risk Ordering

`bright_cool_effect_risk` 排在 `cool_gray_transition_effect_risk` 前面，理由是：

- 前者已经与真实负样本和明确的 rescue 误抬风险直接闭环
- 当前已有更强的输入级 reason/action 语义与之配套
- 先把更确定、更可操作的风险放到报告前面，更利于快速人工验证

如果后续增加更多 risk 类型，可以在同一个 helper 里扩展 rank 映射，而不需要改动现有结论生成函数。

### Decision Severity Ordering

`dominant_decision_reason` 的排序不重写语义，只做前缀分组：

- `fallback_blocked_by_effect_damage*` 最优先
- `fallback_blocked_by_low_weighted_score*` 次之
- `sequence_gvm_retained_without_fallback_evaluation*` 再次之
- `single_frame_or_sequence_gvm_retained*` 最后

这个顺序反映的是诊断优先级，而不是模型绝对质量高低：

- 发生 effect damage block 的输入通常更值得先看，因为它更接近“保护真实透明效果”的主问题
- score blocked 输入仍重要，但更偏向主体恢复边界
- retained 类输入保留在后面，作为“暂未触发 fallback 决策变化”的次级排查对象

### Implementation Shape

实现上保持最小侵入：

- 新增 `_build_input_priority_sort_key(...)` helper
- helper 只读取已经写入 input entry 的字段
- `_build_summary_payload()` 在返回前对 `inputs` 做一次稳定排序

不新增 `summary.json` 字段。这样可以先验证排序是否满足诊断使用，再决定是否需要把 priority 分数显式暴露给 UI。

## Testing

增加或调整测试时，只验证 `inputs[*]` 顺序，不改变已有字段断言的语义。

至少覆盖以下场景：

- 带 `bright_cool_effect_risk` 的 input 排在普通 retained input 前面
- 带 `cool_gray_transition_effect_risk` 的 input 排在无 risk input 前面
- 同为 effect-risk 时，`fallback_blocked_by_effect_damage*` 排在 `fallback_blocked_by_low_weighted_score*` 前面
- 同类 reason/risk 时，使用 `debug_crop_peak_abs_diff` 和 `sample_count` 保持稳定排序

## Risks And Mitigations

- 风险：输入顺序变化可能影响现有依赖默认分组顺序的测试
- 缓解：新增专门的排序回归测试，并把旧测试限制在字段语义而非顺序假设上

- 风险：未来增加更多 effect-risk 类型后，硬编码排序可能不够直观
- 缓解：把 risk rank 独立收敛在 helper 中，后续只扩映射表

- 风险：诊断排序与人工直觉可能存在偏差
- 缓解：本次先只改 `inputs[*]`，避免同时改变 samples/crops 阅读路径，便于快速校正

## Success Criteria

- `summary.json` 中 `inputs[*]` 能把 effect-risk 输入稳定排到前面
- 不改变 sample 级和 crop 级已有诊断字段与排序语义
- 现有 effect-risk / decision reason / recommended action 测试继续通过
- 新增排序回归测试能稳定表达优先级规则
