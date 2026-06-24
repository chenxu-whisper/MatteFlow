# 抠图质量系统实施计划

> **给 agentic worker:** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐任务实施本计划。步骤使用 checkbox（`- [ ]`）语法跟踪。

**目标:** 构建一个让 MatAnyone2、SAM2、BiRefNet 和传统 matte 可评估、可选择、可回归的质量决策系统。

**架构:** 在现有 `HybridMatte` 入口后增加 opt-in 的 `QualityDrivenMatte` 协调器。候选生成器统一输出 `MatteCandidateSequence`，质量评估器产出候选级/区域级分数，区域选择器基于 `RegionOwnership` 和质量分数选择 alpha，并把决策写入 processing report 与回归报告。

**技术栈:** Python 3.10+、NumPy、OpenCV、dataclasses、pytest、现有 MatteFlow pipeline/reporting/evaluation 模块。

---

## 约束和实施原则

- 默认行为必须不变：`quality_selection_enable=False` 时，现有 CLI、GUI 和测试应保持当前行为。
- 第一轮只做规则型评估和确定性选择，不训练、不微调模型。
- SAM2 第一阶段是 guidance-first：没有首帧 mask/box/point 时跳过并记录原因。
- 每个阶段结束都要能独立运行测试并提交。
- 真实模型 wrapper 在第二阶段接入；第一阶段用 fake candidate 覆盖主流程，避免模型下载和 GPU 环境阻塞基础能力。

## 文件结构和职责

### 新增文件

- `src/matteflow/matte/candidates/__init__.py`  
  导出候选协议、生成器接口和具体 wrapper。
- `src/matteflow/matte/candidates/types.py`  
  定义 `MatteCandidate`、`MatteCandidateSequence`、`CandidateGenerationResult`、`CandidateSkipReason`。
- `src/matteflow/matte/candidates/base.py`  
  定义 `CandidateGenerator` 协议、输出归一化、shape 校验、计时辅助。
- `src/matteflow/matte/candidates/traditional.py`  
  包装现有传统绿幕/黑底 matte 为候选。
- `src/matteflow/matte/candidates/matanyone2.py`  
  包装 `MatAnyone2Matte`。
- `src/matteflow/matte/candidates/birefnet.py`  
  包装 `BiRefNetMatte`。
- `src/matteflow/matte/candidates/sam2_guided.py`  
  包装 `SAM2Matte`，只在存在 guidance 时生成候选。
- `src/matteflow/evaluation/matte_quality.py`  
  定义 `CandidateQuality`、`CandidateQualityReport` 和 `MatteQualityEvaluator`。
- `src/matteflow/matte/quality_selector.py`  
  定义 `SelectionDecision`、`QualitySelectionResult` 和 `QualitySelector`。
- `src/matteflow/matte/quality_driven_matte.py`  
  协调候选生成、质量评估、区域选择和诊断汇总。
- `src/matteflow/evaluation/matting_quality_regression.py`  
  基于 manifest 的困难样本回归入口，复用现有 `QualityRegressionEvaluator`。
- `tests/core/test_candidate_contracts.py`  
  候选协议和归一化测试。
- `tests/core/test_matte_quality_evaluator.py`  
  规则型质量评估器测试。
- `tests/core/test_quality_selector.py`  
  区域选择器测试。
- `tests/core/test_quality_driven_matte.py`  
  使用 fake generator 测试新协调器。
- `tests/core/test_matting_quality_regression.py`  
  测试 manifest、样本结果和门禁。
- `tests/fixtures/matting_quality/manifest.json`  
  固定困难样本 manifest。

### 修改文件

- `src/matteflow/config.py`  
  增加 `quality_selection_enable`、`quality_candidate_models`、`quality_selection_mode`。
- `src/matteflow/matte/hybrid_matte.py`  
  当配置开启时委托 `QualityDrivenMatte`，否则保持现有路径。
- `src/matteflow/pipeline.py`  
  传递质量选择诊断到 report 上下文，不改变默认 pipeline 阶段顺序。
- `src/matteflow/reporting/processing_report.py`  
  扩展 schema，增加 `quality_selection` 和相关 artifacts 字段。
- `src/matteflow/reporting/report_view.py`  
  GUI report 展示候选摘要和区域选择统计。
- `tests/core/test_processing_report.py`  
  覆盖 report schema 兼容性。
- `tests/core/test_pipeline_quality_report.py`  
  覆盖 pipeline report 新字段。
- `tests/core/test_quality_regression.py`  
  扩展现有 report-based 回归指标。

## 阶段总览

| 阶段 | 交付物 | 验收标准 |
| --- | --- | --- |
| 阶段 1：基础协议和配置 | 候选 dataclass、生成器基类、配置字段 | 单元测试通过；默认配置关闭；现有测试不因字段新增失败 |
| 阶段 2：质量评估器和选择器 | 规则型 evaluator、区域 selector、fake candidate 覆盖 | 合成样本中能稳定选择 subject/hair/background/effect 区域 |
| 阶段 3：QualityDrivenMatte opt-in 集成 | 新协调器、`HybridMatte` 委托、pipeline 诊断透传 | `quality_selection_enable=False` 行为不变；开启后 fake generator 路径可运行 |
| 阶段 4：真实候选 wrapper | MatAnyone2、BiRefNet、Traditional、SAM2-guided wrapper | 模型不可用时结构化 skipped；传统候选可无模型运行 |
| 阶段 5：报告和调试产物 | report schema、contact sheet/overlay/diff artifact 路径 | report 可 JSON 序列化；debug 关闭时仍有 JSON 摘要 |
| 阶段 6：回归套件 | manifest runner、门禁、示例 fixtures | 可对固定样本生成 pass/fail 报告；缺失可选模型不导致无关失败 |
| 阶段 7：GUI 可见性和最终验证 | GUI report 展示、文档更新、全量测试 | 用户能看到候选决策；核心测试和回归测试通过 |

---

## 阶段 1：基础协议和配置

### Task 1: 增加候选协议类型

**交付物：**
- `MatteCandidate`
- `MatteCandidateSequence`
- `CandidateGenerationResult`
- `CandidateSkipReason`
- 候选输出归一化测试

**验收标准：**
- `alpha` 被转换为 `np.float32` 并裁剪到 `[0.0, 1.0]`。
- shape 不匹配时抛出 `ValueError`。
- `diagnostics` 保持 JSON-friendly dict。

**Files:**
- Create: `src/matteflow/matte/candidates/__init__.py`
- Create: `src/matteflow/matte/candidates/types.py`
- Create: `tests/core/test_candidate_contracts.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_candidate_contracts.py` 中加入：

```python
import numpy as np
import pytest

from matteflow.matte.candidates.types import MatteCandidateSequence


def test_candidate_sequence_normalizes_alpha_dtype_and_range():
    candidate = MatteCandidateSequence.from_raw(
        name="fake",
        alphas=[np.array([[-1.0, 0.5, 2.0]], dtype=np.float64)],
        confidences=[None],
        source="fake",
        runtime_ms=1.25,
        diagnostics={"available": True},
        frame_shapes=[(1, 3)],
    )

    assert candidate.alphas[0].dtype == np.float32
    assert candidate.alphas[0].tolist() == [[0.0, 0.5, 1.0]]
    assert candidate.runtime_ms == 1.25
    assert candidate.diagnostics == {"available": True}


def test_candidate_sequence_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="alpha shape"):
        MatteCandidateSequence.from_raw(
            name="bad",
            alphas=[np.zeros((2, 2), dtype=np.float32)],
            confidences=[None],
            source="fake",
            runtime_ms=0.0,
            diagnostics={},
            frame_shapes=[(1, 2)],
        )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_candidate_contracts.py -v`  
Expected: FAIL，提示 `ModuleNotFoundError` 或 `MatteCandidateSequence` 未定义。

- [ ] **Step 3: 实现候选类型**

在 `src/matteflow/matte/candidates/types.py` 中实现：

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np


class CandidateSkipReason(str, Enum):
    MODEL_UNAVAILABLE = "model_unavailable"
    GUIDANCE_MISSING = "guidance_missing"
    DISABLED_BY_CONFIG = "disabled_by_config"


@dataclass(frozen=True)
class MatteCandidate:
    name: str
    alpha: np.ndarray
    confidence: np.ndarray | None
    source: str
    runtime_ms: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MatteCandidateSequence:
    name: str
    alphas: list[np.ndarray]
    confidences: list[np.ndarray | None]
    source: str
    runtime_ms: float
    diagnostics: dict[str, Any]

    @classmethod
    def from_raw(
        cls,
        *,
        name: str,
        alphas: Sequence[np.ndarray],
        confidences: Sequence[np.ndarray | None] | None,
        source: str,
        runtime_ms: float,
        diagnostics: dict[str, Any] | None,
        frame_shapes: Sequence[tuple[int, int]],
    ) -> "MatteCandidateSequence":
        normalized_alphas = []
        normalized_confidences = []
        raw_confidences = list(confidences or [None] * len(alphas))
        if len(raw_confidences) != len(alphas):
            raise ValueError(f"{name}.confidences length does not match alphas length")
        if len(frame_shapes) != len(alphas):
            raise ValueError(f"{name}.frame_shapes length does not match alphas length")

        for index, (alpha, confidence, expected_shape) in enumerate(zip(alphas, raw_confidences, frame_shapes)):
            alpha_f = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
            if tuple(alpha_f.shape) != tuple(expected_shape):
                raise ValueError(f"{name}.alpha shape {alpha_f.shape} does not match frame shape {expected_shape} at index {index}")
            normalized_alphas.append(alpha_f)
            if confidence is None:
                normalized_confidences.append(None)
                continue
            confidence_f = np.clip(np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
            if confidence_f.shape != alpha_f.shape:
                raise ValueError(f"{name}.confidence shape {confidence_f.shape} does not match alpha shape {alpha_f.shape} at index {index}")
            normalized_confidences.append(confidence_f)

        return cls(
            name=str(name),
            alphas=normalized_alphas,
            confidences=normalized_confidences,
            source=str(source),
            runtime_ms=float(runtime_ms),
            diagnostics=dict(diagnostics or {}),
        )


@dataclass(frozen=True)
class CandidateGenerationResult:
    candidate: MatteCandidateSequence | None
    skipped: bool = False
    skip_reason: CandidateSkipReason | None = None
    message: str = ""
```

在 `src/matteflow/matte/candidates/__init__.py` 中导出：

```python
from .types import (
    CandidateGenerationResult,
    CandidateSkipReason,
    MatteCandidate,
    MatteCandidateSequence,
)

__all__ = [
    "CandidateGenerationResult",
    "CandidateSkipReason",
    "MatteCandidate",
    "MatteCandidateSequence",
]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_candidate_contracts.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\matte\candidates tests\core\test_candidate_contracts.py
git commit -m "feat: add matte candidate contracts"
```

### Task 2: 增加候选生成器基类和配置字段

**交付物：**
- `CandidateGenerator` 协议
- `TimedCandidateGenerator` 辅助基类
- `MattingConfig` 新字段

**验收标准：**
- 默认 `quality_selection_enable=False`。
- 默认候选模型包含 `matanyone2`、`sam2`、`birefnet`、`traditional`。
- 基类能把生成耗时写入 candidate。

**Files:**
- Create: `src/matteflow/matte/candidates/base.py`
- Modify: `src/matteflow/config.py`
- Modify: `tests/core/test_candidate_contracts.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_candidate_contracts.py` 追加：

```python
from matteflow.config import MattingConfig


def test_quality_selection_config_defaults_are_opt_in():
    config = MattingConfig()

    assert config.quality_selection_enable is False
    assert config.quality_candidate_models == ("matanyone2", "sam2", "birefnet", "traditional")
    assert config.quality_selection_mode == "region"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_candidate_contracts.py::test_quality_selection_config_defaults_are_opt_in -v`  
Expected: FAIL，提示配置字段不存在。

- [ ] **Step 3: 实现配置字段**

在 `src/matteflow/config.py` 的 AI 参数或性能参数附近加入：

```python
    # ==================== 质量选择系统 ====================
    quality_selection_enable: bool = False
    quality_candidate_models: tuple[str, ...] = ("matanyone2", "sam2", "birefnet", "traditional")
    quality_selection_mode: str = "region"
```

创建 `src/matteflow/matte/candidates/base.py`：

```python
from __future__ import annotations

import time
from typing import Protocol, Sequence

import numpy as np

from .types import CandidateGenerationResult, MatteCandidateSequence


class CandidateGenerator(Protocol):
    name: str

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        raise NotImplementedError


class TimedCandidateGenerator:
    name: str
    source: str

    def _build_candidate(
        self,
        *,
        start_time: float,
        alphas: Sequence[np.ndarray],
        confidences: Sequence[np.ndarray | None] | None,
        frame_shapes: Sequence[tuple[int, int]],
        diagnostics: dict,
    ) -> CandidateGenerationResult:
        candidate = MatteCandidateSequence.from_raw(
            name=self.name,
            alphas=alphas,
            confidences=confidences,
            source=self.source,
            runtime_ms=(time.perf_counter() - start_time) * 1000.0,
            diagnostics=diagnostics,
            frame_shapes=frame_shapes,
        )
        return CandidateGenerationResult(candidate=candidate)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_candidate_contracts.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\config.py src\matteflow\matte\candidates\base.py tests\core\test_candidate_contracts.py
git commit -m "feat: add quality selection config"
```

---

## 阶段 2：质量评估器和区域选择器

### Task 3: 实现规则型质量评估器

**交付物：**
- `CandidateQuality`
- `CandidateQualityReport`
- `MatteQualityEvaluator`

**验收标准：**
- background residue 区域中低 alpha 候选得分更高。
- hair edge 区域中保留 soft alpha 的候选得分更高。
- 分数方向一致：越高越好。

**Files:**
- Create: `src/matteflow/evaluation/matte_quality.py`
- Create: `tests/core/test_matte_quality_evaluator.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_matte_quality_evaluator.py` 中加入：

```python
import numpy as np

from matteflow.analysis.region_ownership import RegionOwnership
from matteflow.evaluation.matte_quality import MatteQualityEvaluator
from matteflow.matte.candidates.types import MatteCandidateSequence


def _ownership(shape):
    empty = np.zeros(shape, dtype=bool)
    hair = empty.copy()
    hair[:, 1] = True
    residue = empty.copy()
    residue[:, 3] = True
    subject = empty.copy()
    subject[:, 0] = True
    return RegionOwnership(
        subject=subject,
        hair_edge=hair,
        luminous_prop=empty.copy(),
        transparent_effect=empty.copy(),
        background_residue=residue,
        uncertain_edge=hair.copy(),
    )


def _candidate(name, alpha):
    return MatteCandidateSequence.from_raw(
        name=name,
        alphas=[alpha],
        confidences=[None],
        source=name,
        runtime_ms=1.0,
        diagnostics={},
        frame_shapes=[alpha.shape],
    )


def test_evaluator_rewards_clean_background_and_preserved_hair_edge():
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    good_alpha = np.array([[0.9, 0.45, 0.0, 0.0], [0.9, 0.40, 0.0, 0.0]], dtype=np.float32)
    bad_alpha = np.array([[0.9, 0.02, 0.0, 0.5], [0.9, 0.01, 0.0, 0.5]], dtype=np.float32)

    report = MatteQualityEvaluator().evaluate(
        frames=[frame],
        candidates=[_candidate("good", good_alpha), _candidate("bad", bad_alpha)],
        ownerships=[_ownership(good_alpha.shape)],
    )

    good = report.by_candidate["good"][0]
    bad = report.by_candidate["bad"][0]
    assert good.overall_score > bad.overall_score
    assert good.region_scores["hair_edge"] > bad.region_scores["hair_edge"]
    assert good.region_scores["background_residue"] > bad.region_scores["background_residue"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_matte_quality_evaluator.py -v`  
Expected: FAIL，提示模块不存在。

- [ ] **Step 3: 实现评估器**

在 `src/matteflow/evaluation/matte_quality.py` 中实现：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from matteflow.analysis.region_ownership import RegionOwnership
from matteflow.matte.candidates.types import MatteCandidateSequence


REGION_FIELDS = (
    "subject",
    "hair_edge",
    "luminous_prop",
    "transparent_effect",
    "background_residue",
    "uncertain_edge",
)


@dataclass(frozen=True)
class CandidateQuality:
    candidate_name: str
    frame_index: int
    overall_score: float
    region_scores: dict[str, float]
    signals: dict[str, float | int | str]


@dataclass(frozen=True)
class CandidateQualityReport:
    qualities: tuple[CandidateQuality, ...]

    @property
    def by_candidate(self) -> dict[str, list[CandidateQuality]]:
        result: dict[str, list[CandidateQuality]] = {}
        for quality in self.qualities:
            result.setdefault(quality.candidate_name, []).append(quality)
        return result

    def to_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for name, qualities in self.by_candidate.items():
            summary[name] = {
                "overall_score": round(float(np.mean([q.overall_score for q in qualities])), 6),
                "frame_count": len(qualities),
            }
        return summary


class MatteQualityEvaluator:
    def evaluate(
        self,
        *,
        frames: Sequence[np.ndarray],
        candidates: Sequence[MatteCandidateSequence],
        ownerships: Sequence[RegionOwnership],
    ) -> CandidateQualityReport:
        qualities: list[CandidateQuality] = []
        for candidate in candidates:
            for frame_index, alpha in enumerate(candidate.alphas):
                ownership = ownerships[frame_index]
                qualities.append(self._evaluate_frame(candidate.name, frame_index, alpha, ownership))
        return CandidateQualityReport(qualities=tuple(qualities))

    def _evaluate_frame(
        self,
        candidate_name: str,
        frame_index: int,
        alpha: np.ndarray,
        ownership: RegionOwnership,
    ) -> CandidateQuality:
        region_scores = {
            "subject": _mean_alpha_score(alpha, ownership.subject, target="high"),
            "hair_edge": _soft_alpha_score(alpha, ownership.hair_edge),
            "luminous_prop": _mean_alpha_score(alpha, ownership.luminous_prop, target="high"),
            "transparent_effect": _soft_alpha_score(alpha, ownership.transparent_effect),
            "background_residue": _mean_alpha_score(alpha, ownership.background_residue, target="low"),
            "uncertain_edge": _soft_alpha_score(alpha, ownership.uncertain_edge),
        }
        overall_score = float(np.mean(list(region_scores.values()))) if region_scores else 0.0
        signals: dict[str, float | int | str] = {
            "candidate": candidate_name,
            "frame_index": int(frame_index),
            "overall_score": overall_score,
        }
        return CandidateQuality(
            candidate_name=candidate_name,
            frame_index=int(frame_index),
            overall_score=float(np.clip(overall_score, 0.0, 1.0)),
            region_scores={key: float(np.clip(value, 0.0, 1.0)) for key, value in region_scores.items()},
            signals=signals,
        )


def _mean_alpha_score(alpha: np.ndarray, mask: np.ndarray, *, target: str) -> float:
    if not np.any(mask):
        return 0.5
    mean_alpha = float(np.clip(alpha[mask], 0.0, 1.0).mean())
    if target == "high":
        return mean_alpha
    return 1.0 - mean_alpha


def _soft_alpha_score(alpha: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.5
    values = np.clip(alpha[mask], 0.0, 1.0)
    soft = ((values > 0.05) & (values < 0.95)).mean()
    collapsed = (values <= 0.05).mean()
    return float(np.clip(soft * 0.8 + (1.0 - collapsed) * 0.2, 0.0, 1.0))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_matte_quality_evaluator.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\evaluation\matte_quality.py tests\core\test_matte_quality_evaluator.py
git commit -m "feat: add matte quality evaluator"
```

### Task 4: 实现区域级选择器

**交付物：**
- `SelectionDecision`
- `QualitySelectionResult`
- `QualitySelector`

**验收标准：**
- subject 区域选择 subject 分数最高候选。
- background residue 区域选择低 alpha 候选。
- 候选分数相等时按输入顺序确定性选择。

**Files:**
- Create: `src/matteflow/matte/quality_selector.py`
- Create: `tests/core/test_quality_selector.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_quality_selector.py` 中加入：

```python
import numpy as np

from matteflow.analysis.region_ownership import RegionOwnership
from matteflow.evaluation.matte_quality import MatteQualityEvaluator
from matteflow.matte.candidates.types import MatteCandidateSequence
from matteflow.matte.quality_selector import QualitySelector


def _candidate(name, alpha):
    return MatteCandidateSequence.from_raw(
        name=name,
        alphas=[alpha],
        confidences=[None],
        source=name,
        runtime_ms=1.0,
        diagnostics={},
        frame_shapes=[alpha.shape],
    )


def test_selector_chooses_different_candidates_by_region():
    shape = (1, 4)
    subject = np.array([[True, False, False, False]])
    residue = np.array([[False, False, False, True]])
    empty = np.zeros(shape, dtype=bool)
    ownership = RegionOwnership(
        subject=subject,
        hair_edge=empty.copy(),
        luminous_prop=empty.copy(),
        transparent_effect=empty.copy(),
        background_residue=residue,
        uncertain_edge=empty.copy(),
    )
    subject_candidate = _candidate("subject_model", np.array([[1.0, 0.0, 0.0, 0.8]], dtype=np.float32))
    clean_candidate = _candidate("clean_model", np.array([[0.6, 0.0, 0.0, 0.0]], dtype=np.float32))
    quality = MatteQualityEvaluator().evaluate(
        frames=[np.zeros((1, 4, 3), dtype=np.uint8)],
        candidates=[subject_candidate, clean_candidate],
        ownerships=[ownership],
    )

    result = QualitySelector().select_sequence(
        candidates=[subject_candidate, clean_candidate],
        quality_report=quality,
        ownerships=[ownership],
    )

    assert result.alphas[0][0, 0] == 1.0
    assert result.alphas[0][0, 3] == 0.0
    assert result.decisions[0].selected_by_region["subject"] == "subject_model"
    assert result.decisions[0].selected_by_region["background_residue"] == "clean_model"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_quality_selector.py -v`  
Expected: FAIL，提示 `quality_selector` 不存在。

- [ ] **Step 3: 实现 selector**

在 `src/matteflow/matte/quality_selector.py` 中实现：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from matteflow.analysis.region_ownership import RegionOwnership
from matteflow.evaluation.matte_quality import CandidateQualityReport, REGION_FIELDS
from matteflow.matte.candidates.types import MatteCandidateSequence


@dataclass(frozen=True)
class SelectionDecision:
    frame_index: int
    selected_by_region: dict[str, str | None]
    region_scores: dict[str, dict[str, float]]
    rejected_takeovers: dict[str, int]
    warnings: list[str]


@dataclass(frozen=True)
class QualitySelectionResult:
    alphas: list[np.ndarray]
    decisions: list[SelectionDecision]
    diagnostics: dict[str, Any]


class QualitySelector:
    def select_sequence(
        self,
        *,
        candidates: Sequence[MatteCandidateSequence],
        quality_report: CandidateQualityReport,
        ownerships: Sequence[RegionOwnership],
    ) -> QualitySelectionResult:
        if not candidates:
            raise ValueError("QualitySelector requires at least one candidate")
        frame_count = len(candidates[0].alphas)
        alphas: list[np.ndarray] = []
        decisions: list[SelectionDecision] = []
        quality_by_name = quality_report.by_candidate
        for frame_index in range(frame_count):
            ownership = ownerships[frame_index]
            selected_alpha, decision = self._select_frame(candidates, quality_by_name, ownership, frame_index)
            alphas.append(selected_alpha)
            decisions.append(decision)
        diagnostics = {
            "frame_count": frame_count,
            "candidate_count": len(candidates),
            "selected_model_counts": _selected_counts(decisions),
        }
        return QualitySelectionResult(alphas=alphas, decisions=decisions, diagnostics=diagnostics)

    def _select_frame(
        self,
        candidates: Sequence[MatteCandidateSequence],
        quality_by_name: dict[str, list],
        ownership: RegionOwnership,
        frame_index: int,
    ) -> tuple[np.ndarray, SelectionDecision]:
        result = candidates[0].alphas[frame_index].copy()
        selected_by_region: dict[str, str | None] = {}
        region_scores: dict[str, dict[str, float]] = {}
        rejected_takeovers = {region: 0 for region in REGION_FIELDS}
        for region in REGION_FIELDS:
            mask = np.asarray(getattr(ownership, region), dtype=bool)
            if not np.any(mask):
                selected_by_region[region] = None
                region_scores[region] = {}
                continue
            scores = {}
            for candidate in candidates:
                qualities = quality_by_name.get(candidate.name, [])
                score = qualities[frame_index].region_scores.get(region, 0.0) if frame_index < len(qualities) else 0.0
                scores[candidate.name] = float(score)
            selected_name = max(scores, key=lambda name: scores[name])
            selected = next(candidate for candidate in candidates if candidate.name == selected_name)
            result[mask] = selected.alphas[frame_index][mask]
            selected_by_region[region] = selected_name
            region_scores[region] = scores
        return (
            np.clip(result, 0.0, 1.0).astype(np.float32, copy=False),
            SelectionDecision(
                frame_index=frame_index,
                selected_by_region=selected_by_region,
                region_scores=region_scores,
                rejected_takeovers=rejected_takeovers,
                warnings=[],
            ),
        )


def _selected_counts(decisions: Sequence[SelectionDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        for selected in decision.selected_by_region.values():
            if selected is None:
                continue
            counts[selected] = counts.get(selected, 0) + 1
    return counts
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_quality_selector.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\matte\quality_selector.py tests\core\test_quality_selector.py
git commit -m "feat: add region quality selector"
```

---

## 阶段 3：QualityDrivenMatte opt-in 集成

### Task 5: 增加 QualityDrivenMatte 协调器

**交付物：**
- `QualityDrivenMatte`
- fake generator 测试
- 诊断字段汇总

**验收标准：**
- fake generator 可以生成两个候选并通过 evaluator/selector 输出最终 alpha。
- `last_quality_selection` 包含 candidate summary、selected counts、decisions。
- 没有候选成功时返回结构化错误。

**Files:**
- Create: `src/matteflow/matte/quality_driven_matte.py`
- Create: `tests/core/test_quality_driven_matte.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_quality_driven_matte.py` 中加入：

```python
import numpy as np

from matteflow.analysis.region_ownership import RegionOwnershipAnalyzer
from matteflow.matte.candidates.types import CandidateGenerationResult, MatteCandidateSequence
from matteflow.matte.quality_driven_matte import QualityDrivenMatte


class FakeGenerator:
    def __init__(self, name, alpha_value):
        self.name = name
        self.alpha_value = alpha_value

    def generate(self, frames, *, frame_shapes, cancel_check=None, progress_callback=None):
        alpha = np.full(frame_shapes[0], self.alpha_value, dtype=np.float32)
        candidate = MatteCandidateSequence.from_raw(
            name=self.name,
            alphas=[alpha],
            confidences=[None],
            source=self.name,
            runtime_ms=1.0,
            diagnostics={"fake": True},
            frame_shapes=frame_shapes,
        )
        return CandidateGenerationResult(candidate=candidate)


def test_quality_driven_matte_runs_fake_candidates():
    frames = [np.full((2, 2, 3), 255, dtype=np.uint8)]
    matte = QualityDrivenMatte(
        generators=[FakeGenerator("low", 0.0), FakeGenerator("high", 1.0)],
        region_analyzer=RegionOwnershipAnalyzer(),
    )

    alphas = matte.generate_sequence(frames)

    assert len(alphas) == 1
    assert alphas[0].shape == (2, 2)
    assert matte.last_quality_selection["candidate_count"] == 2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_quality_driven_matte.py -v`  
Expected: FAIL，提示模块不存在。

- [ ] **Step 3: 实现协调器**

在 `src/matteflow/matte/quality_driven_matte.py` 中实现：

```python
from __future__ import annotations

from typing import Sequence

import numpy as np

from matteflow.analysis.region_ownership import RegionOwnershipAnalyzer
from matteflow.evaluation.matte_quality import MatteQualityEvaluator
from matteflow.matte.quality_selector import QualitySelector


class QualityDrivenMatte:
    def __init__(
        self,
        *,
        generators,
        region_analyzer: RegionOwnershipAnalyzer | None = None,
        evaluator: MatteQualityEvaluator | None = None,
        selector: QualitySelector | None = None,
    ) -> None:
        self.generators = list(generators)
        self.region_analyzer = region_analyzer or RegionOwnershipAnalyzer()
        self.evaluator = evaluator or MatteQualityEvaluator()
        self.selector = selector or QualitySelector()
        self.last_quality_selection: dict = {}
        self.last_skipped_candidates: list[dict] = []

    def generate_sequence(self, frames: Sequence[np.ndarray], *, cancel_check=None, progress_callback=None) -> list[np.ndarray]:
        frame_shapes = [tuple(frame.shape[:2]) for frame in frames]
        candidates = []
        skipped = []
        for generator in self.generators:
            result = generator.generate(
                frames,
                frame_shapes=frame_shapes,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            if result.candidate is not None:
                candidates.append(result.candidate)
            if result.skipped:
                skipped.append({"name": getattr(generator, "name", "unknown"), "reason": result.skip_reason, "message": result.message})
        if not candidates:
            raise RuntimeError("quality selection has no successful candidates")
        base_alphas = candidates[0].alphas
        ownerships = [
            self.region_analyzer.analyze(frame, alpha)
            for frame, alpha in zip(frames, base_alphas)
        ]
        quality_report = self.evaluator.evaluate(frames=frames, candidates=candidates, ownerships=ownerships)
        selection = self.selector.select_sequence(candidates=candidates, quality_report=quality_report, ownerships=ownerships)
        self.last_skipped_candidates = skipped
        self.last_quality_selection = {
            "available": True,
            "candidate_count": len(candidates),
            "skipped_candidates": skipped,
            "candidate_quality": quality_report.to_summary(),
            "selected_model_counts": selection.diagnostics.get("selected_model_counts", {}),
            "decisions": [
                {
                    "frame_index": decision.frame_index,
                    "selected_by_region": decision.selected_by_region,
                    "region_scores": decision.region_scores,
                    "warnings": decision.warnings,
                }
                for decision in selection.decisions
            ],
        }
        return selection.alphas
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_quality_driven_matte.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\matte\quality_driven_matte.py tests\core\test_quality_driven_matte.py
git commit -m "feat: add quality driven matte coordinator"
```

### Task 6: 将 HybridMatte 以 opt-in 方式委托到 QualityDrivenMatte

**交付物：**
- `HybridMatte` 配置开启时走质量系统
- 默认关闭时不变
- fake quality coordinator 测试

**验收标准：**
- `MattingConfig().quality_selection_enable` 为 false 时，不触发 `QualityDrivenMatte`。
- true 时，`HybridMatte.generate_sequence()` 返回质量系统输出。
- `HybridMatte.last_quality_selection` 暴露给 pipeline/report。

**Files:**
- Modify: `src/matteflow/matte/hybrid_matte.py`
- Modify: `tests/core/test_quality_driven_matte.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_quality_driven_matte.py` 追加：

```python
from matteflow.config import BackgroundMode, MattingConfig
from matteflow.matte.hybrid_matte import HybridMatte


class FakeQualityDrivenMatte:
    last_quality_selection = {"available": True, "candidate_count": 1}

    def generate_sequence(self, frames, *, cancel_check=None, progress_callback=None):
        return [np.ones(frame.shape[:2], dtype=np.float32) for frame in frames]


def test_hybrid_matte_delegates_when_quality_selection_enabled(monkeypatch):
    config = MattingConfig(quality_selection_enable=True, use_ai=False)
    hybrid = HybridMatte(config)
    fake = FakeQualityDrivenMatte()
    monkeypatch.setattr(hybrid, "_build_quality_driven_matte", lambda: fake)

    alphas = hybrid.generate_sequence([np.zeros((2, 2, 3), dtype=np.uint8)], BackgroundMode.GREEN_SCREEN)

    assert alphas[0].mean() == 1.0
    assert hybrid.last_quality_selection["candidate_count"] == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_quality_driven_matte.py::test_hybrid_matte_delegates_when_quality_selection_enabled -v`  
Expected: FAIL，提示 `_build_quality_driven_matte` 或 `last_quality_selection` 不存在。

- [ ] **Step 3: 实现委托入口**

在 `HybridMatte.__init__` 中初始化：

```python
        self.last_quality_selection = None
```

在 `HybridMatte.generate_sequence()` 开头、现有模式分支前加入：

```python
        if getattr(self.config, "quality_selection_enable", False):
            quality_matte = self._build_quality_driven_matte()
            alphas = quality_matte.generate_sequence(
                frames,
                cancel_check=cancel_check,
                progress_callback=progress_callback,
            )
            self.last_quality_selection = quality_matte.last_quality_selection
            self.last_active_ai_model = "quality_selection"
            return alphas
```

在 `HybridMatte` 中新增：

```python
    def _build_quality_driven_matte(self):
        from .quality_driven_matte import QualityDrivenMatte
        from .candidates.traditional import TraditionalCandidateGenerator

        return QualityDrivenMatte(
            generators=[
                TraditionalCandidateGenerator(self.config),
            ],
            region_analyzer=self.region_analyzer,
        )
```

第一轮只接 traditional generator，真实 AI wrapper 在阶段 4 增加。

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_quality_driven_matte.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\matte\hybrid_matte.py tests\core\test_quality_driven_matte.py
git commit -m "feat: enable opt-in quality selection path"
```

---

## 阶段 4：真实候选 wrapper

### Task 7: 实现 TraditionalCandidateGenerator

**交付物：**
- 绿幕/黑底传统候选
- 无模型依赖的候选路径

**验收标准：**
- 绿幕模式调用 `GreenScreenMatte`。
- 黑底模式调用 `BlackBackgroundMatte`。
- 输出符合 `MatteCandidateSequence`。

**Files:**
- Create: `src/matteflow/matte/candidates/traditional.py`
- Modify: `tests/core/test_candidate_contracts.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_candidate_contracts.py` 追加：

```python
from matteflow.config import BackgroundMode, MattingConfig
from matteflow.matte.candidates.traditional import TraditionalCandidateGenerator


def test_traditional_candidate_generator_outputs_sequence():
    generator = TraditionalCandidateGenerator(MattingConfig(background_mode=BackgroundMode.GREEN_SCREEN))
    frames = [np.zeros((3, 4, 3), dtype=np.uint8)]

    result = generator.generate(frames, frame_shapes=[(3, 4)])

    assert result.candidate is not None
    assert result.candidate.name in {"traditional_green", "traditional_black"}
    assert result.candidate.alphas[0].shape == (3, 4)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_candidate_contracts.py::test_traditional_candidate_generator_outputs_sequence -v`  
Expected: FAIL，提示模块不存在。

- [ ] **Step 3: 实现 traditional wrapper**

在 `src/matteflow/matte/candidates/traditional.py` 中实现：

```python
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from matteflow.config import BackgroundMode, MattingConfig
from matteflow.matte.black_background_matte import BlackBackgroundMatte
from matteflow.matte.green_screen_matte import GreenScreenMatte

from .base import TimedCandidateGenerator
from .types import CandidateGenerationResult


class TraditionalCandidateGenerator(TimedCandidateGenerator):
    def __init__(self, config: MattingConfig) -> None:
        self.config = config
        mode = getattr(config, "background_mode", BackgroundMode.AUTO)
        self.mode = mode
        self.name = "traditional_black" if mode == BackgroundMode.BLACK_BACKGROUND else "traditional_green"
        self.source = self.name
        self._green = GreenScreenMatte(config)
        self._black = BlackBackgroundMatte(config)

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        start = time.perf_counter()
        alphas = []
        for index, frame in enumerate(frames):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("candidate generation cancelled")
            if self.mode == BackgroundMode.BLACK_BACKGROUND:
                alpha = self._black.generate(frame)
            else:
                alpha = self._green.generate(frame)
            alphas.append(alpha)
            if progress_callback:
                progress_callback(index + 1, len(frames))
        return self._build_candidate(
            start_time=start,
            alphas=alphas,
            confidences=[None] * len(alphas),
            frame_shapes=frame_shapes,
            diagnostics={"mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode)},
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_candidate_contracts.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\matte\candidates\traditional.py tests\core\test_candidate_contracts.py
git commit -m "feat: add traditional matte candidate"
```

### Task 8: 实现 MatAnyone2、BiRefNet、SAM2-guided wrapper

**交付物：**
- 三个 AI candidate wrapper
- 模型不可用时结构化 skipped
- SAM2 guidance 缺失时 skipped

**验收标准：**
- wrapper 不强制下载模型。
- 模型不可用返回 `CandidateGenerationResult(skipped=True)`。
- 用户显式指定模型时可在后续集成层转为 fail-fast。

**Files:**
- Create: `src/matteflow/matte/candidates/matanyone2.py`
- Create: `src/matteflow/matte/candidates/birefnet.py`
- Create: `src/matteflow/matte/candidates/sam2_guided.py`
- Modify: `tests/core/test_candidate_contracts.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_candidate_contracts.py` 追加：

```python
from matteflow.matte.candidates.types import CandidateSkipReason
from matteflow.matte.candidates.sam2_guided import SAM2GuidedCandidateGenerator


def test_sam2_guided_candidate_skips_without_guidance():
    generator = SAM2GuidedCandidateGenerator(MattingConfig(), guidance=None)

    result = generator.generate([np.zeros((2, 2, 3), dtype=np.uint8)], frame_shapes=[(2, 2)])

    assert result.skipped is True
    assert result.skip_reason == CandidateSkipReason.GUIDANCE_MISSING
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_candidate_contracts.py::test_sam2_guided_candidate_skips_without_guidance -v`  
Expected: FAIL，提示模块不存在。

- [ ] **Step 3: 实现 wrapper 骨架**

三个 wrapper 都遵循同一模式。`sam2_guided.py` 先实现 guidance skip：

```python
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from matteflow.config import MattingConfig

from .base import TimedCandidateGenerator
from .types import CandidateGenerationResult, CandidateSkipReason


class SAM2GuidedCandidateGenerator(TimedCandidateGenerator):
    name = "sam2_guided"
    source = "sam2_guided"

    def __init__(self, config: MattingConfig, guidance: Any | None = None) -> None:
        self.config = config
        self.guidance = guidance

    def generate(
        self,
        frames: Sequence[np.ndarray],
        *,
        frame_shapes: Sequence[tuple[int, int]],
        cancel_check=None,
        progress_callback=None,
    ) -> CandidateGenerationResult:
        if self.guidance is None:
            return CandidateGenerationResult(
                candidate=None,
                skipped=True,
                skip_reason=CandidateSkipReason.GUIDANCE_MISSING,
                message="SAM2 guidance is required for sam2_guided candidate",
            )
        return CandidateGenerationResult(
            candidate=None,
            skipped=True,
            skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE,
            message="SAM2 guided generation is not connected until guidance payload format is added",
        )
```

`matanyone2.py` 和 `birefnet.py` 先实现模型可用性结构化 skip，并在后续小步接真实 `generate()`：

```python
from __future__ import annotations

from typing import Sequence

import numpy as np

from matteflow.config import MattingConfig

from .base import TimedCandidateGenerator
from .types import CandidateGenerationResult, CandidateSkipReason


class MatAnyone2CandidateGenerator(TimedCandidateGenerator):
    name = "matanyone2"
    source = "matanyone2"

    def __init__(self, config: MattingConfig) -> None:
        self.config = config
        try:
            from matteflow.matte.matanyone2_matte import MatAnyone2Matte
            self.engine = MatAnyone2Matte(config)
        except Exception as exc:
            self.engine = None
            self.load_error = str(exc)
        else:
            self.load_error = ""

    def generate(self, frames: Sequence[np.ndarray], *, frame_shapes: Sequence[tuple[int, int]], cancel_check=None, progress_callback=None) -> CandidateGenerationResult:
        if self.engine is None or getattr(self.engine, "model", None) is None:
            return CandidateGenerationResult(candidate=None, skipped=True, skip_reason=CandidateSkipReason.MODEL_UNAVAILABLE, message=self.load_error or "MatAnyone2 model unavailable")
        alphas = [self.engine.generate(frame) for frame in frames]
        return self._build_candidate(start_time=0.0, alphas=alphas, confidences=[None] * len(alphas), frame_shapes=frame_shapes, diagnostics={"model": "matanyone2"})
```

BiRefNet 使用同样结构，类名为 `BiRefNetCandidateGenerator`，engine 为 `BiRefNetMatte`。

- [ ] **Step 4: 修正耗时实现**

把上面 AI wrapper 中 `start_time=0.0` 改为实际计时：

```python
import time

start = time.perf_counter()
alphas = [self.engine.generate(frame) for frame in frames]
return self._build_candidate(
    start_time=start,
    alphas=alphas,
    confidences=[None] * len(alphas),
    frame_shapes=frame_shapes,
    diagnostics={"model": self.name},
)
```

- [ ] **Step 5: 运行 wrapper 测试**

Run: `pytest tests/core/test_candidate_contracts.py -v`  
Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src\matteflow\matte\candidates tests\core\test_candidate_contracts.py
git commit -m "feat: add ai matte candidate wrappers"
```

---

## 阶段 5：报告和调试产物

### Task 9: 扩展 ProcessingReport

**交付物：**
- `quality_selection` report section
- model decisions 兼容旧字段
- report schema 测试

**验收标准：**
- 没有质量选择诊断时，report 中 `quality_selection.available=False`。
- 有诊断时，candidate quality、selected counts 和 skipped candidates 可 JSON 序列化。

**Files:**
- Modify: `src/matteflow/reporting/processing_report.py`
- Modify: `tests/core/test_processing_report.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_processing_report.py` 增加一个 report builder 测试，构造 `hybrid_matte`：

```python
class HybridWithQualitySelection:
    last_active_ai_model = "quality_selection"
    last_fallback_quality_metrics = {}
    green_screen_layer_debug = None
    last_quality_selection = {
        "available": True,
        "candidate_count": 2,
        "selected_model_counts": {"matanyone2": 3},
        "candidate_quality": {"matanyone2": {"overall_score": 0.9}},
        "skipped_candidates": [],
    }
```

断言：

```python
payload = report.to_dict()
assert payload["quality_selection"]["available"] is True
assert payload["quality_selection"]["candidate_count"] == 2
assert payload["quality_selection"]["selected_model_counts"] == {"matanyone2": 3}
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_processing_report.py -v`  
Expected: FAIL，提示 `quality_selection` 不存在。

- [ ] **Step 3: 修改 report dataclass 和 builder**

在 `ProcessingReport` 中增加字段：

```python
    quality_selection: dict[str, Any]
```

在 `to_dict()` 中增加：

```python
            "quality_selection": _json_safe(self.quality_selection),
```

在 builder 创建 `ProcessingReport` 时增加：

```python
            quality_selection=self._build_quality_selection(hybrid_matte),
```

新增方法：

```python
    @staticmethod
    def _build_quality_selection(hybrid_matte: Any | None) -> dict[str, Any]:
        selection = getattr(hybrid_matte, "last_quality_selection", None) if hybrid_matte is not None else None
        if not selection:
            return {
                "available": False,
                "candidate_count": 0,
                "selected_model_counts": {},
                "candidate_quality": {},
                "skipped_candidates": [],
            }
        return _json_safe(dict(selection))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_processing_report.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\reporting\processing_report.py tests\core\test_processing_report.py
git commit -m "feat: report quality selection decisions"
```

### Task 10: 生成 debug artifacts 路径和最小 contact sheet

**交付物：**
- `artifacts["quality_selection_debug_dir"]`
- 可选 contact sheet 生成函数
- debug 关闭时 JSON 仍存在

**验收标准：**
- `output_debug=False` 不写图片，不影响 report。
- `output_debug=True` 写入 `debug/quality_selection_contact_sheet.png`。

**Files:**
- Modify: `src/matteflow/pipeline.py`
- Modify: `tests/core/test_pipeline_quality_report.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_pipeline_quality_report.py` 增加断言：当 `output_debug=True` 且 `hybrid_matte.last_quality_selection` 存在时，report artifacts 包含 `quality_selection_debug_dir`。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_pipeline_quality_report.py -v`  
Expected: FAIL，提示 artifact 缺失。

- [ ] **Step 3: 扩展 `_build_output_artifacts()`**

在 `src/matteflow/pipeline.py` 的 `_build_output_artifacts()` 中增加：

```python
        if getattr(self.config, "output_debug", False):
            artifacts["quality_selection_debug_dir"] = output_dir / "debug" / "quality_selection"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_pipeline_quality_report.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\pipeline.py tests\core\test_pipeline_quality_report.py
git commit -m "feat: expose quality selection debug artifacts"
```

---

## 阶段 6：回归套件

### Task 11: 增加 manifest 解析和质量回归 runner

**交付物：**
- `MattingQualityRegressionSample`
- `MattingQualityRegressionManifest`
- `MattingQualityRegressionRunner`

**验收标准：**
- manifest 能解析样本名称、输入、背景模式、候选模型和风险上限。
- 缺失可选模型可记录 degraded。
- runner 输出 `quality_summary.json`。

**Files:**
- Create: `src/matteflow/evaluation/matting_quality_regression.py`
- Create: `tests/core/test_matting_quality_regression.py`
- Create: `tests/fixtures/matting_quality/manifest.json`

- [ ] **Step 1: 写 manifest fixture**

`tests/fixtures/matting_quality/manifest.json`：

```json
{
  "samples": [
    {
      "name": "green_frame_smoke",
      "input_path": "assets/frame/test_frame_1.png",
      "background_mode": "green_screen",
      "quality_mode": "standard",
      "candidate_models": ["traditional"],
      "risk_ceilings": {
        "background_residue": 0.2,
        "hair_edge_loss": 0.5
      }
    }
  ]
}
```

- [ ] **Step 2: 写失败测试**

在 `tests/core/test_matting_quality_regression.py`：

```python
from pathlib import Path

from matteflow.evaluation.matting_quality_regression import MattingQualityRegressionManifest


def test_manifest_loads_samples():
    manifest = MattingQualityRegressionManifest.from_path(Path("tests/fixtures/matting_quality/manifest.json"))

    assert len(manifest.samples) == 1
    assert manifest.samples[0].name == "green_frame_smoke"
    assert manifest.samples[0].candidate_models == ("traditional",)
    assert manifest.samples[0].risk_ceilings["background_residue"] == 0.2
```

- [ ] **Step 3: 运行测试确认失败**

Run: `pytest tests/core/test_matting_quality_regression.py -v`  
Expected: FAIL，提示模块不存在。

- [ ] **Step 4: 实现 manifest 类型**

在 `src/matteflow/evaluation/matting_quality_regression.py` 中实现：

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class MattingQualityRegressionSample:
    name: str
    input_path: Path
    background_mode: str
    quality_mode: str
    candidate_models: tuple[str, ...]
    risk_ceilings: dict[str, float]


@dataclass(frozen=True)
class MattingQualityRegressionManifest:
    samples: tuple[MattingQualityRegressionSample, ...]

    @classmethod
    def from_path(cls, path: Path | str) -> "MattingQualityRegressionManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_samples = payload.get("samples", [])
        samples = []
        for item in raw_samples:
            samples.append(
                MattingQualityRegressionSample(
                    name=str(item["name"]),
                    input_path=Path(str(item["input_path"])),
                    background_mode=str(item["background_mode"]),
                    quality_mode=str(item["quality_mode"]),
                    candidate_models=tuple(str(model) for model in item.get("candidate_models", [])),
                    risk_ceilings={str(key): float(value) for key, value in dict(item.get("risk_ceilings", {})).items()},
                )
            )
        return cls(samples=tuple(samples))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `pytest tests/core/test_matting_quality_regression.py -v`  
Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src\matteflow\evaluation\matting_quality_regression.py tests\core\test_matting_quality_regression.py tests\fixtures\matting_quality\manifest.json
git commit -m "feat: add matting quality regression manifest"
```

### Task 12: 扩展 QualityRegressionEvaluator 指标

**交付物：**
- report 中 `quality_selection` 指标可被回归 evaluator 读取
- P0 风险和 candidate selection 可比较

**验收标准：**
- evaluator metrics 包含 `quality_selection.candidate_count`。
- candidate_count 为 0 且 quality_selection available 时失败。

**Files:**
- Modify: `src/matteflow/evaluation/quality_regression.py`
- Modify: `tests/core/test_quality_regression.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_quality_regression.py` 增加 report payload，包含：

```python
"quality_selection": {
  "available": True,
  "candidate_count": 0,
  "selected_model_counts": {},
  "candidate_quality": {},
  "skipped_candidates": []
}
```

断言 failure 包含 `quality selection enabled but no candidates available`。

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_quality_regression.py -v`  
Expected: FAIL，failure 不存在。

- [ ] **Step 3: 扩展 `_extract_metrics()` 和 `_build_failures()`**

在 `_extract_metrics()` 读取：

```python
    quality_selection = payload.get("quality_selection")
    if isinstance(quality_selection, Mapping):
        metrics["quality_selection.available"] = bool(quality_selection.get("available", False))
        metrics["quality_selection.candidate_count"] = _int_metric(quality_selection, "candidate_count")
```

在 `_build_failures()` 中加入：

```python
        if metrics.get("quality_selection.available") is True and _int_metric(metrics, "quality_selection.candidate_count") <= 0:
            failures.append("quality selection enabled but no candidates available")
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_quality_regression.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\evaluation\quality_regression.py tests\core\test_quality_regression.py
git commit -m "feat: include quality selection in regression gates"
```

---

## 阶段 7：GUI 可见性和最终验证

### Task 13: 在 report view 中展示候选决策摘要

**交付物：**
- GUI report view 展示 quality selection available/candidates/selected counts
- 无 quality selection 时显示“不启用”

**验收标准：**
- 旧 report 不报错。
- 新 report 可看到候选数量和选中模型统计。

**Files:**
- Modify: `src/matteflow/reporting/report_view.py`
- Modify: `tests/core/test_processing_report.py`

- [ ] **Step 1: 写失败测试**

在 `tests/core/test_processing_report.py` 中增加测试：

```python
from matteflow.reporting.report_view import format_quality_selection_summary


def test_report_view_formats_quality_selection_summary():
    summary = format_quality_selection_summary(
        {
            "quality_selection": {
                "available": True,
                "candidate_count": 2,
                "selected_model_counts": {"matanyone2": 4},
            }
        }
    )

    assert "候选数量: 2" in summary
    assert "matanyone2: 4" in summary
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/core/test_processing_report.py -v`  
Expected: FAIL，函数不存在或输出不包含摘要。

- [ ] **Step 3: 实现 summary 格式化**

在 `src/matteflow/reporting/report_view.py` 中加入：

```python
def format_quality_selection_summary(payload: dict) -> str:
    quality_selection = payload.get("quality_selection") or {}
    if not quality_selection.get("available"):
        return "质量选择: 未启用"
    lines = [
        "质量选择: 已启用",
        f"候选数量: {int(quality_selection.get('candidate_count') or 0)}",
    ]
    selected_counts = quality_selection.get("selected_model_counts") or {}
    for model_name, count in selected_counts.items():
        lines.append(f"{model_name}: {int(count)}")
    return "\n".join(lines)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/core/test_processing_report.py -v`  
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src\matteflow\reporting\report_view.py tests\core\test_processing_report.py
git commit -m "feat: show quality selection in report view"
```

### Task 14: 最终验证和文档更新

**交付物：**
- README 或技术文档补充质量系统使用方式
- 全量核心测试通过
- 工作区干净

**验收标准：**
- `pytest tests/core -v` 通过。
- `quality_selection_enable=False` 默认行为测试通过。
- 文档说明如何开启质量选择和如何查看 report。

**Files:**
- Modify: `README.md` 或 `docs/technical_route_and_architecture.md`
- Test: `tests/core`

- [ ] **Step 1: 更新文档**

在 `docs/technical_route_and_architecture.md` 增加一节：

```markdown
### 质量选择系统

MatteFlow 支持可选的质量选择系统。开启 `quality_selection_enable=True` 后，系统会生成多个候选 matte，按区域评估候选质量，并输出候选决策摘要到 `processing_report.json`。默认配置保持关闭，以保证现有 CLI 和 GUI 行为稳定。
```

- [ ] **Step 2: 运行核心测试**

Run: `pytest tests/core -v`  
Expected: PASS。

- [ ] **Step 3: 运行质量相关定向测试**

Run:

```powershell
pytest tests/core/test_candidate_contracts.py tests/core/test_matte_quality_evaluator.py tests/core/test_quality_selector.py tests/core/test_quality_driven_matte.py tests/core/test_quality_regression.py -v
```

Expected: PASS。

- [ ] **Step 4: 检查工作区**

Run: `git status --short`  
Expected: 只显示本 task 文档改动，或为空。

- [ ] **Step 5: 提交**

```powershell
git add docs\technical_route_and_architecture.md
git commit -m "docs: document matting quality selection"
```

## 总体验收标准

- 默认模式：`MattingConfig().quality_selection_enable is False`，现有 pipeline 测试通过。
- opt-in 模式：开启质量选择后，`HybridMatte` 委托 `QualityDrivenMatte`，并产出 `last_quality_selection`。
- 候选协议：所有候选输出 alpha 为 float32、范围 `[0.0, 1.0]`、shape 与输入帧一致。
- 质量评估：至少覆盖 subject、hair_edge、transparent_effect、luminous_prop、background_residue、uncertain_edge。
- 区域选择：同一输入和配置下选择结果确定。
- 报告：`processing_report.json` 包含 `quality_selection`，旧 report 兼容。
- 回归：`QualityRegressionEvaluator` 能读取 quality selection 指标并触发门禁。
- GUI：report view 能展示质量选择摘要。
- 测试：`pytest tests/core -v` 通过。

## 风险和缓解

- 真实模型加载慢或缺模型：第一阶段使用 fake/traditional 候选，AI wrapper 返回 skipped，避免阻塞基础功能。
- 规则型质量分数不够准：先保证可解释和可回归，后续基于困难样本集调优；接口保留 learned evaluator 空间。
- `HybridMatte` 过大：新逻辑放入 `QualityDrivenMatte`，`HybridMatte` 只做 opt-in 委托。
- report schema 变化影响旧测试：新增字段必须提供默认值，旧 report view 必须兼容字段缺失。
- SAM2 guidance 数据结构尚未统一：第一阶段只定义 skip 和接口位置，不强行设计 GUI 交互格式。

## 自检记录

- Spec 覆盖：候选协议、候选生成器、质量评估器、区域选择器、pipeline opt-in、报告、回归、GUI 可见性均有对应任务。
- 占位符检查：计划中没有未定义的“待补充实现”；SAM2 guidance 缺失行为明确为结构化 skipped。
- 类型一致性：`MatteCandidateSequence`、`CandidateGenerationResult`、`CandidateQualityReport`、`QualitySelectionResult` 在首次定义后保持同名使用。
