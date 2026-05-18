# GVM ModelChecker Strict Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ModelChecker` mark `GVM` as available only when weights, CUDA, and the vendored runtime import all succeed.

**Architecture:** Keep the change isolated to `ModelChecker._check_gvm()`. Add a focused regression test that simulates "weights exist and CUDA is available but vendored runtime import fails", then make the checker return `available=False` with a precise reason so GUI model visibility follows automatically.

**Tech Stack:** Python, `pytest`, `importlib`

---

### Task 1: Lock the false-positive behavior with a failing test

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Add a test for "weights + CUDA + runtime import failure"**
- [ ] **Step 2: Run the targeted test and confirm it fails**

### Task 2: Tighten GVM availability checks

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\utils\model_checker.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Make `_check_gvm()` verify the vendored runtime import**
- [ ] **Step 2: Return `available=False` and a runtime-specific reason when import fails**
- [ ] **Step 3: Re-run the targeted test and confirm it passes**

### Task 3: Verify GUI-facing behavior

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\utils\model_checker.py` if needed
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Run `pytest tests/test_model_checker_runtime.py -q`**
- [ ] **Step 2: Run `python -m matteflow.utils.model_checker` from the worktree `src` directory**
- [ ] **Step 3: Restart the GUI and confirm `gvm` drops out of `Available models`**
