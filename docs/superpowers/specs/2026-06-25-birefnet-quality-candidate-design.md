# BiRefNet 质量候选接入设计

## 背景

MatteFlow 已经完成质量选择系统的第一阶段：候选协议、规则型质量评估、区域级选择器、`QualityDrivenMatte`、`ProcessingReport.quality_selection`、manifest 回归和质量门禁都已经打通。当前 MatAnyone2、BiRefNet、SAM2 的 candidate wrapper 可以结构化跳过，也支持注入 fake engine 做测试，但默认业务路径里仍主要由 `traditional` 候选产生结果。

下一阶段优先接入 BiRefNet。原因是 BiRefNet 已有 `BiRefNetMatte` 引擎，输入输出是单帧 alpha，和当前 candidate 协议最匹配；它不需要 SAM2 prompt，也比 MatAnyone2 的视频时序约束更容易先打通真实候选链路。

## 目标

1. 让 `BiRefNetCandidateGenerator` 在显式配置下可以懒加载 `BiRefNetMatte` 并生成真实候选。
2. 保持默认安全：不自动下载模型，不在普通运行或测试中触发 HuggingFace 权重加载。
3. 加载失败、依赖缺失或推理失败时，继续以结构化 `skipped_candidates` 记录，不影响 `traditional` fallback。
4. 让 `quality_selection` report 能区分 BiRefNet 是参与选择、加载失败、不可用，还是推理失败。
5. 通过 fake engine 和 monkeypatch 覆盖真实候选路径，不依赖外部模型权重完成回归测试。

## 非目标

- 不默认下载 HuggingFace 权重。
- 不在 CI 或核心单元测试中执行真实 BiRefNet 模型推理。
- 不引入完整模型资产管理系统，例如权重版本锁定、缓存清理、模型镜像源配置。
- 不同时接入 MatAnyone2 和 SAM2 的真实推理路径。
- 不改变现有 `ai_model="birefnet"` 的传统 HybridMatte 选择路径，只扩展 quality selection 候选路径。

## 用户入口

### 配置字段

在 `MattingConfig` 中新增：

```python
quality_birefnet_auto_load: bool = False
```

语义：

- `False`：默认值。`BiRefNetCandidateGenerator` 只有在外部传入可用 engine 时才生成候选；未传 engine 时结构化 skip。
- `True`：当 `quality_candidate_models` 包含 `"birefnet"` 且没有注入 engine 时，generator 尝试构造 `BiRefNetMatte(config)`。

### CLI 参数

新增：

```text
--quality-birefnet-auto-load
```

该参数只在同时启用 `--quality-selection` 且候选列表包含 `birefnet` 时有实际效果。它是显式 opt-in，不改变默认运行行为。

## 组件设计

### BiRefNetCandidateGenerator

当前 wrapper 已能接收 `engine` 并调用 `engine.generate_sequence()`。下一阶段扩展为：

- 初始化参数增加 `engine_factory`，默认指向 `BiRefNetMatte` 的延迟导入工厂。
- `generate()` 开始时调用 `_ensure_engine()`。
- `_ensure_engine()` 逻辑：
  - 如果已有 `engine` 且 `engine.model is not None`，直接使用。
  - 如果已有 `engine` 但模型不可用，返回 `MODEL_UNAVAILABLE` skip。
  - 如果没有 `engine` 且 `quality_birefnet_auto_load=False`，返回 `MODEL_UNAVAILABLE` skip，message 说明 auto-load 未开启。
  - 如果没有 `engine` 且 auto-load 开启，调用 `engine_factory(config)`。
  - 如果构造失败或构造后 `model is None`，返回 `MODEL_UNAVAILABLE` skip，并把错误写入 message/diagnostics。
- 推理调用失败时返回 `GENERATION_FAILED` skip，而不是抛出到 `QualityDrivenMatte` 外层。

### QualityDrivenMatte

无需大改。保持：

```python
BiRefNetCandidateGenerator(
    self.config,
    engine=self.candidate_engines.get("birefnet"),
)
```

`HybridMatte._build_quality_driven_matte()` 已经把 `self.birefnet` 传入 candidate engines。新增 auto-load 后，即使 `HybridMatte` 没有预加载 BiRefNet，只要用户显式打开 `quality_birefnet_auto_load`，wrapper 也能自行尝试加载。

### Reporting

沿用已有 `quality_selection` 结构，不新增 schema 字段。通过以下内容表达状态：

```json
{
  "quality_selection": {
    "candidate_count": 2,
    "candidate_quality": {
      "traditional": {"overall_score": 0.72},
      "birefnet": {"overall_score": 0.81}
    },
    "selected_model_counts": {
      "birefnet": 1,
      "traditional": 1
    },
    "skipped_candidates": []
  }
}
```

加载失败时：

```json
{
  "name": "birefnet",
  "reason": "model_unavailable",
  "message": "BiRefNet auto-load failed: ..."
}
```

推理失败时：

```json
{
  "name": "birefnet",
  "reason": "generation_failed",
  "message": "..."
}
```

## 错误处理

BiRefNet 候选接入必须遵循“失败降级，不中断整体质量选择”的原则：

- BiRefNet 不可用时，如果 `traditional` 候选可用，最终仍输出结果。
- BiRefNet 是唯一候选且不可用时，`QualityDrivenMatte` 保持现有行为：记录 `last_quality_selection.available=False` 并抛出 `RuntimeError("No quality selection candidates were generated")`。
- 加载失败和推理失败要进入日志和 report 的 `skipped_candidates`。

## 测试策略

1. 单元测试 `BiRefNetCandidateGenerator`：
   - 默认不自动加载，未传 engine 时 skipped。
   - fake engine 可生成候选，alpha 被规范化。
   - auto-load 开启时调用 fake factory。
   - auto-load 构造失败时 skipped，reason 为 `model_unavailable`。
   - 推理失败时 skipped，reason 为 `generation_failed`。

2. 集成测试 `QualityDrivenMatte`：
   - `quality_candidate_models=("birefnet", "traditional")` 时，fake BiRefNet 参与候选质量评估和选择。
   - BiRefNet 加载失败时，traditional 仍能输出最终 alpha。

3. CLI 测试：
   - `_build_config()` 能读取 `--quality-birefnet-auto-load`。
   - 默认值保持 `False`。

4. 回归测试：
   - manifest 继续使用 `traditional` 默认样本，保证不触发真实模型下载。
   - 新增 fake/mocked BiRefNet 测试，不依赖外部权重。

## 验收标准

- 默认运行不触发 BiRefNet 自动加载。
- 开启 `quality_birefnet_auto_load=True` 后，generator 会尝试构造 `BiRefNetMatte`。
- fake BiRefNet engine 可以作为真实候选参与 selector，并在 `selected_model_counts` 或 `candidate_quality` 中可见。
- BiRefNet 加载失败或推理失败时，`traditional` fallback 不受影响。
- `python -m pytest tests\core -q` 通过。
- 真实 manifest 回归仍通过，且不依赖 BiRefNet 权重。

## 风险和后续

- 真实 BiRefNet 权重下载和推理环境仍受 HuggingFace、transformers、torch、CUDA/CPU 性能影响。本阶段只保证显式 opt-in 和失败降级。
- 后续可以在模型资产管理阶段补充本地权重路径、缓存检测、离线模式和 benchmark 多样本集。
- 如果 BiRefNet 候选质量明显优于 traditional，下一阶段可以把 manifest 扩展为多样本、多模型对比报告。
