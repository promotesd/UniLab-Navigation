# UniLab Navigation reproducibility

The navigation release is locked by `uv.lock`. Run commands from the repository
root and keep every generated checkpoint, log, manifest, and report outside the
checkout (the workflow defaults use `/tmp/unilab_navigation_repro`).

## Environment

```bash
uv sync --frozen --extra mujoco
```

Record the lock hash with every experiment. The M11 workflow writes Git state,
the `uv.lock` SHA-256, software/hardware metadata, seed/config values, command
timings, and input/output hashes to both `provenance.json` and
`provenance.csv`.

## One-command workflows

Quick release validation (Ruff, 181 navigation tests, 32 fixed episodes, a
small three-seed framework benchmark, and the artifact audit):

```bash
uv run python scripts/navigation_repro.py quick
```

Reference training:

```bash
uv run python scripts/navigation_repro.py train ppo --profile quick --seed 1
uv run python scripts/navigation_repro.py train ppo --profile formal --seed 1
uv run python scripts/navigation_repro.py train sac --profile formal --seed 1
uv run python scripts/navigation_repro.py train td3 --profile formal --seed 1
```

Fixed-episode baseline evaluation:

```bash
uv run python scripts/navigation_repro.py evaluate --profile quick
uv run python scripts/navigation_repro.py evaluate --profile formal
```

Framework-versus-standalone benchmark matrix:

```bash
uv run python scripts/navigation_repro.py benchmark --profile quick
uv run python scripts/navigation_repro.py benchmark --profile formal
```

Repository release audit:

```bash
uv run python scripts/navigation_repro.py audit
```

The fixed seeds and formal budgets are versioned in
`conf/navigation/reproducibility.json`. Learned checkpoint evaluation and the
matched PPO/SAC/TD3 aggregator remain available through
`scripts/evaluate_point_goal.py` and `scripts/compare_point_goal_algorithms.py`;
their checkpoint paths are experiment outputs and therefore are not committed.

## Quick versus formal bounds

The quick workflow is intended for a workstation or navigation CI job. On the
M11 reference machine it should finish in well under one minute, uses 32 MuJoCo
environments, and writes only a few megabytes under `/tmp`.

Formal PPO/SAC/TD3 training requires CUDA, uses 1,024 environments and 98,304
environment steps per seed, and produces checkpoints plus TensorBoard events.
Run all three seeds sequentially to avoid GPU contention. Formal fixed-episode
evaluation uses 4,096 environments and can require several gigabytes of RAM.
The formal M10 benchmark uses 256 environments, three seeds, two warmups, and
five measured repetitions per seed; it is a CPU/FP32 workload.

Runtime varies by hardware. Preserve failed seeds and cold-start effects; do not
select only favorable results. A full formal algorithm matrix is intentionally
not part of ordinary CI.

## Result contract

`provenance.json` is the canonical nested record. `provenance.csv` is a
one-row portable index with JSON-encoded seed/config/input/output columns. Both
use schema version `1`. Workflow-specific evaluation and benchmark JSON files
remain authoritative for metrics and raw repetitions.

The audit rejects tracked files under `logs/`, `runs/`, `wandb/`, or
`checkpoints/`; TensorBoard event files; PyTorch checkpoints; root-level
`run_config.json`/`run_summary.json`; missing lock/release files; and missing
ignore rules.
