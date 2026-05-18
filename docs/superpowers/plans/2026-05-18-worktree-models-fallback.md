# Worktree Models Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a worktree GUI reuse the main project `models` directory when the worktree-local `models` directory is missing or empty.

**Architecture:** Keep the fix isolated to model path resolution. `model_paths.py` should continue to expose one canonical `models_root()`, but that root should prefer the current project `models` directory and fall back to the main workspace `models` directory when running inside `.worktrees/...`. Add regression tests so `ModelChecker` and the GUI automatically inherit the corrected path without any UI-specific branching.

**Tech Stack:** Python, `pathlib`, `pytest`

---

### Task 1: Lock the fallback behavior with tests

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Write the failing fallback test**

```python
def test_models_root_falls_back_to_main_project_when_worktree_models_missing(monkeypatch):
    fake_file = Path(
        r"E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\utils\model_paths.py"
    )
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run: `pytest tests/test_model_checker_runtime.py::test_models_root_falls_back_to_main_project_when_worktree_models_missing -v`
Expected: FAIL because `models_root()` currently always points at the worktree-local `models` directory.

### Task 2: Implement fallback path resolution

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\utils\model_paths.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Add a helper that detects the main project root from a worktree path**
- [ ] **Step 2: Make `models_root()` prefer local models, then fall back to the main project `models` directory**
- [ ] **Step 3: Re-run the targeted test and confirm it passes**

### Task 3: Verify model checker behavior end-to-end

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\utils\model_paths.py` if tiny follow-up adjustment is needed
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Run `pytest tests/test_model_checker_runtime.py -q`**
- [ ] **Step 2: Run `python -m matteflow.utils.model_checker` from the worktree `src` directory**
- [ ] **Step 3: Restart the GUI and confirm its startup log lists the recovered models**
