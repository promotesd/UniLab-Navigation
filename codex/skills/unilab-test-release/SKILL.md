---
name: unilab-test-release
description: Validate, stage, commit, and push UniLab Navigation changes safely. Use at the end of a milestone, before a Git commit, when checking generated files, or when preparing a branch for review. Enforces targeted tests, full navigation regression, smoke tests, clean diffs, and exclusion of logs/checkpoints.
metadata:
  short-description: Validate and publish one clean UniLab milestone
---

# UniLab Test and Release

## Preflight

From the repository root:

```bash
git status --short
git branch --show-current
git diff --check
```

Inspect every changed and untracked path. Remove accidental temporary files. Do not delete user data.

## Validation order

1. Ruff on changed Python paths.
2. Targeted unit tests.
3. Full `tests/envs/navigation`.
4. Relevant runtime smoke test.
5. Benchmark/evaluator smoke when the milestone changes evaluation.
6. `git diff --check`.

Do not run a long research training job merely to validate syntax.

## Generated artifacts

Never stage:

- `logs/`;
- `model_*.pt`;
- TensorBoard events;
- videos;
- datasets;
- `__pycache__`;
- `.pytest_cache`;
- temporary JSON/CSV unless explicitly designated as a checked-in fixture.

Verify with:

```bash
git status --short
git diff --cached --name-status
```

## Commit scope

One milestone per commit. The commit must contain implementation, tests, and config needed by that milestone.

Before commit:

```bash
git diff --cached --check
git diff --cached --stat
```

After commit:

```bash
git show --stat --oneline HEAD
git status --short
```

Push the active feature branch only after all required checks pass.

## Failure handling

If a test fails:

1. stop before commit;
2. diagnose whether implementation or test expectation is wrong;
3. fix the smallest root cause;
4. rerun the failed test;
5. rerun the relevant suite;
6. report the exact remaining failure if blocked.
