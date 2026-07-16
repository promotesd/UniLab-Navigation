# UniLab Navigation + SLAM Project Instructions

## Repository

- Local root: `/home/xiaodudu/robot_study/UniLab-Navigation`
- Git remote: `git@github.com:promotesd/UniLab-Navigation.git`
- Active development branch: `feat/navigation-mvp`
- Python/package manager: `uv`
- Main simulator for the navigation MVP: MuJoCo
- Primary training entrypoint: `scripts/train_rsl_rl.py`

Always resolve paths from the repository root. Never assume the current shell directory is correct.

## Project mission

Build and evaluate UniLab as a reusable robotics-learning framework that can:

1. host navigation reinforcement-learning tasks through a stable environment contract;
2. support multiple RL algorithms through adapters instead of task-specific rewrites;
3. accept SLAM/localization outputs and simulated or recorded sensor streams through explicit interfaces;
4. compare UniLab implementations with equivalent standalone implementations using a fair benchmark protocol.

Do not claim that UniLab is faster merely because it is the target framework. Treat speed as a hypothesis. A speed claim requires controlled measurements of throughput, wall-clock time, resource use, sample efficiency, and time-to-threshold.

## Current implemented state

The branch already contains:

- `DiffDrivePointGoalCfg`;
- a vectorized PointGoal environment;
- a real MuJoCo differential-drive model;
- wheel-speed control;
- registry integration for `DiffDrivePointGoal` + `mujoco`;
- RSL-RL PPO wrapper integration;
- Hydra task configuration;
- episode navigation metrics;
- PPO smoke training and TensorBoard metric validation.

Do not reimplement these components. Inspect them first and extend them.

## Current next milestone

The next incomplete milestone is M5.2: a deterministic fixed-episode evaluator comparing:

- zero policy;
- random policy;
- heuristic PointGoal controller;
- PPO checkpoints.

The evaluator must use the same initial-condition set for every controller and report episode-level metrics.

## Non-negotiable engineering rules

1. Read this file and the relevant skill before changing code.
2. Inspect existing code, tests, registration, configuration, and import paths before editing.
3. Keep one milestone per commit.
4. Add or update tests with every behavior change.
5. Run targeted tests first, then the full navigation suite.
6. Do not commit `logs/`, checkpoints, TensorBoard event files, videos, caches, datasets, or temporary benchmark output.
7. Do not modify vendored RSL-RL code when an adapter or wrapper can solve the problem.
8. Preserve official/default PPO settings in formal task config. Hardware smoke-test overrides belong on the command line.
9. Use deterministic seeds in tests and evaluation.
10. Avoid hidden fallback behavior. Fail with actionable errors when configuration or checkpoints are invalid.
11. Never use `nano`. For complete file creation use heredocs or a deterministic script; for edits use a small Python patch script or the repository editing tools.
12. Do not combine unrelated refactors with a feature milestone.

## Architecture boundaries

Keep these responsibilities separate:

```text
task config
  -> registry
  -> environment contract
  -> simulator backend
  -> algorithm wrapper
  -> training runner
  -> independent evaluator
  -> benchmark/report
```

Navigation task logic must not depend directly on a particular RL algorithm.

SLAM implementations must not be embedded directly into reward functions. Expose estimator outputs through an observation/state-provider contract.

## Standard validation

Run from the repository root:

```bash
uv run ruff check \
  src/unilab/envs/navigation \
  src/unilab/training \
  tests/envs/navigation
```

```bash
uv run pytest tests/envs/navigation -q
```

For a one-iteration PPO integration smoke test:

```bash
uv run python scripts/train_rsl_rl.py \
  task=diff_drive_point_goal/mujoco \
  algo.max_iterations=1 \
  training.no_play=true
```

A one-iteration run only validates integration. It is not evidence that learning works.

## Git workflow

Before edits:

```bash
git status --short
git branch --show-current
git log -5 --oneline
```

Before commit:

```bash
git diff --check
git status --short
```

After staging:

```bash
git diff --cached --check
git diff --cached --name-status
git diff --cached --stat
```

Use descriptive commits such as:

```text
feat(navigation): add fixed-episode policy evaluator
feat(navigation): add obstacle and collision task
feat(slam): add localization observation provider
perf(benchmark): compare UniLab and standalone throughput
```

## Codex skills

Invoke the project orchestrator for broad implementation work:

```text
$unilab-navigation-orchestrator
```

Use narrower skills when the task is specific:

- `$unilab-navigation-task`
- `$unilab-rl-adapter`
- `$unilab-slam-adapter`
- `$unilab-benchmark`
- `$unilab-test-release`
