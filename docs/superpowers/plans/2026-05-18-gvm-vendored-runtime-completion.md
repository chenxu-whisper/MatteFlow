# GVM Vendored Runtime Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the vendored GVM runtime so `matteflow.vendor.gvm_core.wrapper` imports successfully and `GVM` can return to the GUI model list.

**Architecture:** Vendor the missing upstream `gvm/models` package from the known source repository, then add a focused regression test that imports the real wrapper in the current environment. After the import path is green, re-run `ModelChecker` and the GUI startup flow to confirm `GVM` is shown only when the runtime truly works.

**Tech Stack:** Python, `pytest`, vendored upstream GVM code, `diffusers`, `peft`

---

### Task 1: Lock the missing vendored runtime with a failing import test

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Add a test that imports `matteflow.vendor.gvm_core.wrapper` for real**
- [ ] **Step 2: Run the targeted test and confirm it fails on the missing `gvm.models` package**

### Task 2: Vendor the missing GVM model package

**Files:**
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\vendor\gvm_core\gvm\models\__init__.py`
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\vendor\gvm_core\gvm\models\unet_spatio_temporal_condition.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Add the upstream `gvm.models` package with MatteFlow-compatible imports**
- [ ] **Step 2: Re-run the targeted import test and confirm it passes**

### Task 3: Verify runtime restoration through the app boundary

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\src\matteflow\utils\model_checker.py` only if compatibility follow-up is needed
- Test: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\transparency-layered-fusion-v1\tests\test_model_checker_runtime.py`

- [ ] **Step 1: Run `pytest tests/test_model_checker_runtime.py -q`**
- [ ] **Step 2: Run `python -m matteflow.utils.model_checker` from the worktree `src` directory**
- [ ] **Step 3: Restart the GUI and confirm `gvm` returns to `Available models`**
