# Project Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the MatteFlow main working tree and `feature/green-screen-competitive-composer` worktree, then merge the feature branch into `main`.

**Architecture:** Use a recoverable, git-first workflow: snapshot both working trees, convert real feature changes into a mergeable state, merge into `main`, reapply preserved main work, clean generated artifacts, and validate with focused and full tests. Generated review/debug/artifact output is treated as disposable only after snapshots exist.

**Tech Stack:** Git worktrees, PowerShell 5, Python 3.10+, pytest, VS Code diagnostics, Trae file tools.

---

## File Structure

- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.gitignore` only if generated cleanup patterns are missing.
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\src\**\*.py` only through merge or conflict resolution.
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\tests\**\*.py` only through merge or conflict resolution.
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\pyproject.toml` only through merge or conflict resolution.
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\uv.lock` only through merge or conflict resolution if the lockfile is present after merge.
- Delete: generated directories matching `.review*`, `.tmp*`, `.dbg`, and `artifacts` in both target working trees.
- Create: `.superpowers/snapshots/2026-06-16-project-consolidation/` for recovery patches, status logs, and untracked file manifests.

## Task 1: Record Baseline State

**Files:**
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\main-status.txt`
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\main-diff.patch`
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\main-untracked.txt`
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\feature-status.txt`
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\feature-diff.patch`
- Create: `E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\feature-untracked.txt`

- [ ] **Step 1: Create the snapshot directory**

Run:

```powershell
New-Item -ItemType Directory -Force -Path "E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation" | Out-Null
```

Expected: command exits with code `0`.

- [ ] **Step 2: Capture main status and diff**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git status --short | Out-File -Encoding utf8 ".superpowers\snapshots\2026-06-16-project-consolidation\main-status.txt"
git diff --binary | Out-File -Encoding utf8 ".superpowers\snapshots\2026-06-16-project-consolidation\main-diff.patch"
git ls-files --others --exclude-standard | Out-File -Encoding utf8 ".superpowers\snapshots\2026-06-16-project-consolidation\main-untracked.txt"
```

Expected: all three files are created and `git diff --binary` exits with code `0`.

- [ ] **Step 3: Capture feature status and diff**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer"
git status --short | Out-File -Encoding utf8 "E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\feature-status.txt"
git diff --binary | Out-File -Encoding utf8 "E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\feature-diff.patch"
git ls-files --others --exclude-standard | Out-File -Encoding utf8 "E:\ByteDance\Projects\Code\MatteFlow\.superpowers\snapshots\2026-06-16-project-consolidation\feature-untracked.txt"
```

Expected: all three files are created and `git diff --binary` exits with code `0`.

- [ ] **Step 4: Create backup branch refs**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git branch backup/main-before-project-consolidation-20260616 HEAD
Set-Location "E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer"
git branch backup/green-screen-composer-before-project-consolidation-20260616 HEAD
```

Expected: both `git branch` commands exit with code `0`, or report that the backup branch already exists.

## Task 2: Remove Generated Artifacts After Snapshot

**Files:**
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.review_codeguard_52a0aca_recheck`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.tmp_codeguard_52a0aca`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.dbg`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review-codeguard-08f58e3`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review-codeguard-5da46aa`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review-codeguard-7cc4d8b-current`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review-codeguard-head-6f1e7d9`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review_bits_code_guard_2f84c4d`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review_code_guard_e35005a`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review_codegen_5da46aa`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review_task1_3f6e3c2`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review_task1_codegen`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review_task1_recheck`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review_task1_scope`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.review_tmp_20260611114208`
- Delete: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\artifacts`

- [ ] **Step 1: Confirm candidates still exist**

Use `LS` or `Glob` on both working trees for `.review*`, `.tmp*`, `.dbg`, and `artifacts`.

Expected: the generated directories listed in this task are present or already absent.

- [ ] **Step 2: Delete generated candidates with the file deletion tool**

Use `DeleteFile` with the existing absolute paths from this task. Do not use `Remove-Item`, `git clean`, or shell deletion.

Expected: generated directories are removed, and source, test, config, dependency, and documentation files remain untouched.

- [ ] **Step 3: Verify cleanup status**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git status --short
Set-Location "E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer"
git status --short
```

Expected: generated directories no longer appear as untracked entries.

## Task 3: Make Feature Worktree Mergeable

**Files:**
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\.gitignore`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\src\matteflow\utils\model_paths.py`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\src\matteflow\utils\output_paths.py`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\tests\web_gui\test_web_gui_output_dir.py`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\tests\web_gui\test_web_gui_queue_integration.py`
- Modify: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\uv.lock`
- Preserve or delete after review: `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\debug-gui-state-regression.md`

- [ ] **Step 1: Review feature diff**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer"
git diff -- .gitignore src/matteflow/utils/model_paths.py src/matteflow/utils/output_paths.py tests/web_gui/test_web_gui_output_dir.py tests/web_gui/test_web_gui_queue_integration.py uv.lock
```

Expected: output shows only real source, test, ignore, and lockfile changes.

- [ ] **Step 2: Decide debug note treatment**

Read `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer\debug-gui-state-regression.md`.

Expected: if it explains current source/test changes, keep it for merge; if it only documents resolved local debugging, delete it with `DeleteFile`.

- [ ] **Step 3: Run focused feature tests**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer"
python -m pytest tests/web_gui/test_web_gui_output_dir.py tests/web_gui/test_web_gui_queue_integration.py tests/core/test_model_paths.py -q
```

Expected: focused tests pass, or failures are recorded with the failing test names and traceback.

- [ ] **Step 4: Commit feature real changes**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer"
git add .gitignore src/matteflow/utils/model_paths.py src/matteflow/utils/output_paths.py tests/web_gui/test_web_gui_output_dir.py tests/web_gui/test_web_gui_queue_integration.py uv.lock
git status --short
git commit -m "chore: consolidate green screen composer worktree"
```

Expected: the commit succeeds if staged files contain changes. If there are no staged changes, record that the feature worktree already matches its HEAD for real project files.

## Task 4: Merge Feature Branch Into Main

**Files:**
- Modify through merge: `E:\ByteDance\Projects\Code\MatteFlow\src\**\*.py`
- Modify through merge: `E:\ByteDance\Projects\Code\MatteFlow\tests\**\*.py`
- Modify through merge: `E:\ByteDance\Projects\Code\MatteFlow\.gitignore`
- Modify through merge: `E:\ByteDance\Projects\Code\MatteFlow\uv.lock`
- Modify through merge: `E:\ByteDance\Projects\Code\MatteFlow\pyproject.toml`

- [ ] **Step 1: Stash main dirty state with untracked files**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git stash push -u -m "pre-project-consolidation-main-20260616"
```

Expected: stash succeeds and reports saved working directory and index state.

- [ ] **Step 2: Merge feature branch**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git merge --no-ff feature/green-screen-competitive-composer
```

Expected: merge succeeds or stops with conflict markers. If conflicts occur, proceed to the conflict resolution step.

- [ ] **Step 3: Resolve merge conflicts**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git status --short
```

Expected: conflicted files are listed with `UU`, `AA`, `DD`, `DU`, or `UD` markers.

For each conflicted source or test file:

- Read the main side and feature side.
- Preserve feature green-screen composer behavior when it does not remove main-only behavior.
- Preserve main-only changes when they are unrelated.
- Remove conflict markers.
- Stage the resolved file with `git add <path>`.

- [ ] **Step 4: Complete merge commit**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git status --short
git commit --no-edit
```

Expected: if merge did not auto-commit, the generated merge message is used. If merge already committed, `git commit --no-edit` reports no changes to commit.

## Task 5: Reapply Preserved Main Work

**Files:**
- Modify through stash application: files originally listed in `main-status.txt`

- [ ] **Step 1: Reapply main stash**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git stash list
git stash apply "stash^{/pre-project-consolidation-main-20260616}"
```

Expected: stash applies cleanly or reports conflicts.

- [ ] **Step 2: Resolve stash conflicts**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git status --short
```

Expected: conflicted files are listed if stash application conflicted.

For each conflict:

- Preserve the merge result for feature behavior.
- Reintroduce main dirty changes only when they are not generated artifacts and not obsolete design/plan deletions.
- Remove conflict markers.
- Stage resolved files with `git add <path>`.

- [ ] **Step 3: Drop stash only after verification**

Run this only after focused and full tests complete:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git stash drop "stash^{/pre-project-consolidation-main-20260616}"
```

Expected: stash is removed after its contents are either applied or intentionally excluded.

## Task 6: Validate Consolidated Main

**Files:**
- Test: `E:\ByteDance\Projects\Code\MatteFlow\tests\core\test_model_paths.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\tests\web_gui\test_web_gui_output_dir.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\tests\web_gui\test_web_gui_queue_integration.py`
- Test: `E:\ByteDance\Projects\Code\MatteFlow\tests`

- [ ] **Step 1: Run diagnostics**

Use `GetDiagnostics` for recently merged or manually resolved files.

Expected: no new Python syntax or linter diagnostics in edited files.

- [ ] **Step 2: Run focused tests**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
python -m pytest tests/web_gui/test_web_gui_output_dir.py tests/web_gui/test_web_gui_queue_integration.py tests/core/test_model_paths.py -q
```

Expected: focused tests pass. If a test fails because of a real regression, fix the implementation or test expectation before proceeding.

- [ ] **Step 3: Run full tests**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
python -m pytest tests -q
```

Expected: full test suite passes. If environment-only failures occur because model, GPU, or FFmpeg resources are unavailable, record the exact failure and whether the failing test is skipped in normal local setup.

- [ ] **Step 4: Review final git status**

Run:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git status --short
Set-Location "E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer"
git status --short
```

Expected: main contains only intentional source, test, config, dependency, documentation, and snapshot changes; feature worktree has no generated artifacts left.

## Task 7: Finalize Handoff

**Files:**
- Modify: git index only

- [ ] **Step 1: Prepare final summary**

Collect:

- merge commit hash
- feature consolidation commit hash if created
- deleted generated directories
- conflict files and chosen resolutions
- focused test result
- full test result
- remaining dirty files, if any

Expected: all items are available from command output and `git status`.

- [ ] **Step 2: Commit final non-merge cleanup if needed**

Run only when post-merge edits exist and tests have passed:

```powershell
Set-Location "E:\ByteDance\Projects\Code\MatteFlow"
git add .gitignore src tests pyproject.toml uv.lock docs
git status --short
git commit -m "chore: consolidate project state"
```

Expected: final cleanup commit succeeds, or `git status --short` shows no final non-merge changes to commit.

- [ ] **Step 3: Report completion**

Report:

- branches touched
- commits created
- files deleted
- tests run
- failures or skipped checks
- recommended next action: push, PR, or manual review

Expected: user receives a concise completion report with evidence.
