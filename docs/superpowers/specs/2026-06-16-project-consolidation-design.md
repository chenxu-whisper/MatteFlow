# Project Consolidation Design

## Goal

Consolidate the MatteFlow project by organizing both the main working tree and the
`feature/green-screen-competitive-composer` worktree, with the final goal of merging
the feature branch into `main`.

## Scope

- Target main working tree: `E:\ByteDance\Projects\Code\MatteFlow`
- Target feature worktree:
  `E:\ByteDance\Projects\Code\MatteFlow\.worktrees\green-screen-competitive-composer`
- Final branch direction: `feature/green-screen-competitive-composer` into `main`
- Allowed operation level: full consolidation after plan approval

## Current Context

- The main working tree has modified files, deleted tracked design/plan documents,
  and untracked review or diagnostic output directories.
- The feature worktree has modified source, tests, lockfile, `.gitignore`, and
  multiple untracked review, debug, and artifact directories.
- The project is a Python `src` layout application for video and image matting,
  with CLI, Gradio Web GUI, core matting modules, refinement modules, utilities,
  and pytest-based tests.

## Chosen Approach

Use a fast merge first workflow:

1. Record the exact dirty state of both working trees.
2. Create a recoverable snapshot before any destructive cleanup or merge operation.
3. Attempt to merge the feature branch into `main` as the primary path.
4. Resolve conflicts in favor of preserving verified feature behavior unless that
   would overwrite important main working tree changes.
5. Clean generated review, debug, and artifact output after merge risk is understood.
6. Validate affected tests first, then run the full test suite.

This approach prioritizes convergence speed while preserving recovery points.

## Cleanup Boundary

Candidate cleanup targets include generated or temporary project output:

- `.review*`
- `.tmp*`
- `.dbg`
- `artifacts`
- generated code review reports
- generated debug images and statistics

Review-before-keep targets include files that may represent real project changes:

- source files under `src/`
- tests under `tests/`
- `pyproject.toml`
- `requirements.txt`
- `uv.lock`
- `.gitignore`
- documentation under `docs/`
- standalone debug notes that may explain unresolved behavior

The user approved expanded cleanup, so generated artifacts may be removed after the
snapshot step. Source, test, dependency, and configuration files still require
explicit review before modification.

## Merge Rules

- Do not discard uncommitted changes without an explicit recovery path.
- Do not use destructive git commands such as hard reset or forced checkout.
- Prefer normal merge mechanics and conflict resolution over history rewriting.
- Preserve feature-branch green-screen composer behavior unless tests or direct
  inspection show it regresses main behavior.
- Preserve main branch changes when they are unrelated to the feature and are not
  conflicting.

## Validation

Validation should proceed in this order:

1. Static diagnostics for recently edited files.
2. Focused tests for changed core, utility, and Web GUI modules.
3. Full `pytest tests` run after focused tests pass or after failures are documented.
4. Final `git status --short` review for both working trees.

## Deliverables

- Consolidation execution plan.
- List of files merged, edited, deleted, or intentionally preserved.
- Conflict resolution notes, if any.
- Test and diagnostic results.
- Final recommendation for commit, PR, or additional cleanup.

## Open Risks

- Existing dirty changes in `main` may be unrelated to the feature branch.
- Existing dirty changes in the feature worktree may not all be committed feature
  work.
- Full tests may require local model, GPU, or FFmpeg resources and may need fallback
  handling if environment dependencies are unavailable.
