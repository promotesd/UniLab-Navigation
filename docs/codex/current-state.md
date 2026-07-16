# UniLab Navigation + SLAM Current State

Last updated: 2026-07-16

## Repository state

- Active branch: `feat/navigation-mvp`
- Remote: `git@github.com:promotesd/UniLab-Navigation.git`
- M5.1 episode metrics: complete in `f068ebdd`
- M5.2 fixed-episode evaluator: complete in `cbac3705`
- Earliest incomplete milestone after this update: M5.5 trajectory analysis

Completion is based on source, tests, commit history, and bounded real MuJoCo
runs rather than roadmap labels alone.

## M5.3 fixed-episode baseline evidence

Status: complete.

Protocol:

- simulator: MuJoCo `DiffDrivePointGoal` task;
- fixed episodes: 4,096;
- manifest seed: 1001;
- random-policy seed: 2001;
- episode limit: 200 control steps (`20.0 s`, `ctrl_dt=0.1 s`);
- policies: stationary zero, seeded uniform random, heuristic PointGoal;
- autoreset: disabled by the evaluator;
- every policy used manifest SHA-256
  `2e55e0f8a5eb492c3f1230be820582eb7a08ef0a258d0aa8d542c2a71aabb0e9`.

Results:

| Policy | Success | Timeout | Final distance (mean ± std) | Progress ratio (mean ± std) | Episode length (mean ± std) |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero | 0.0000 | 1.0000 | 2.9801 ± 1.1618 | 0.0000 ± 0.0001 | 200.0000 ± 0.0000 |
| random | 0.0220 | 0.9780 | 3.6732 ± 1.7537 | -0.3543 ± 0.6945 | 198.2305 ± 13.1575 |
| heuristic | 1.0000 | 0.0000 | 0.2323 ± 0.0117 | 0.9048 ± 0.0487 | 107.7568 ± 33.0702 |

Failure counts retained with episode IDs and initial conditions in the raw
report were: zero 4,096; random 4,006; heuristic 0. The zero policy therefore
behaved as the stationary lower bound, random was a weak stochastic lower
bound, and the heuristic solved the obstacle-free task at high success.

Formal command:

```bash
uv run python scripts/evaluate_point_goal.py \
  --episodes 4096 \
  --seed 1001 \
  --random-policy-seed 2001 \
  --policies zero random heuristic \
  --manifest-output /tmp/point_goal_m53_manifest_4096.json \
  --output /tmp/point_goal_m53_baselines_4096.json
```

Deterministic replay command:

```bash
uv run python scripts/evaluate_point_goal.py \
  --manifest-input /tmp/point_goal_m53_manifest_4096.json \
  --random-policy-seed 2001 \
  --policies zero random heuristic \
  --output /tmp/point_goal_m53_baselines_4096_replay.json
```

The original and replay reports were byte-identical:

```text
a499c759180c1fc63c02a7c766cafd0424571cc2c18b633dd5bc5208732f7a36
```

The standalone manifest file SHA-256 was:

```text
4447451b4c5eb33fcf4f17dc891c03380f96e5769f487827af4235e53094f800
```

The raw JSON reports and manifest are deliberately stored under `/tmp` and are
not committed. Results are descriptive evidence for this controlled manifest;
they are not a claim that UniLab is globally faster or better.

## M5.4 formal multi-seed PPO evidence

Status: complete.

Training protocol:

- seeds: 1, 2, and 3;
- environments: 4,096;
- rollout: 24 steps per environment per iteration;
- iterations: 101 configured, producing iterations 0 through 100;
- environment steps per seed: 9,928,704;
- checkpoint interval: 25 iterations (`0`, `25`, `50`, `75`, `100`);
- actor and critic: `[512, 256, 128]` ELU MLPs from the formal config;
- simulator/task/reward/action/observation settings: identical across seeds;
- device: NVIDIA GeForce RTX 3050 Ti Laptop GPU, CUDA, float32;
- training wall time: seed 1 `178.3 s`, seed 2 `191.0 s`, seed 3 `199.3 s`.

Independent evaluation protocol:

- held-out episodes per seed/checkpoint: 4,096;
- held-out manifest seed: 1001;
- held-out manifest SHA-256:
  `2e55e0f8a5eb492c3f1230be820582eb7a08ef0a258d0aa8d542c2a71aabb0e9`;
- autoreset disabled and exactly one episode counted per manifest row;
- 15 checkpoints evaluated independently;
- final metrics come only from the fixed-episode evaluator, not rollout-window
  episode extras.

Cross-seed held-out results:

| Iteration | Success (mean ± std) | Timeout (mean ± std) | Final distance (mean ± std) | Progress ratio (mean ± std) | Episode length (mean ± std) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0317 ± 0.0062 | 0.9683 ± 0.0062 | 4.5212 ± 0.2204 | -0.7328 ± 0.0935 | 196.4690 ± 0.6758 |
| 25 | 0.9948 ± 0.0074 | 0.0052 ± 0.0074 | 0.2331 ± 0.0008 | 0.9042 ± 0.0004 | 112.3825 ± 0.9505 |
| 50 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2311 ± 0.0003 | 0.9054 ± 0.0003 | 96.9561 ± 1.4309 |
| 75 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2304 ± 0.0005 | 0.9060 ± 0.0002 | 93.5771 ± 0.2618 |
| 100 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2301 ± 0.0006 | 0.9061 ± 0.0004 | 92.9981 ± 0.7374 |

Per-seed iteration-25 success was `1.0000`, `1.0000`, and `0.9844`.
All seeds reached `1.0000` held-out success by iteration 50. This establishes
learning under this controlled obstacle-free protocol; it is not a performance
or framework-speed claim.

Formal training commands:

```bash
uv run python scripts/train_rsl_rl.py \
  task=diff_drive_point_goal/mujoco \
  algo.seed=1 algo.max_iterations=101 algo.save_interval=25 \
  training.no_play=true training.log_root=/tmp/unilab_m54_formal_logs

uv run python scripts/train_rsl_rl.py \
  task=diff_drive_point_goal/mujoco \
  algo.seed=2 algo.max_iterations=101 algo.save_interval=25 \
  training.no_play=true training.log_root=/tmp/unilab_m54_formal_logs

uv run python scripts/train_rsl_rl.py \
  task=diff_drive_point_goal/mujoco \
  algo.seed=3 algo.max_iterations=101 algo.save_interval=25 \
  training.no_play=true training.log_root=/tmp/unilab_m54_formal_logs
```

Formal evaluation command:

```bash
uv run python scripts/evaluate_point_goal_ppo_sweep.py \
  --manifest-input /tmp/point_goal_m53_manifest_4096.json \
  --expected-seeds 1 2 3 --device cuda \
  --checkpoint 1=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-45-55_mujoco/model_0.pt \
  --checkpoint 2=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-49-19_mujoco/model_0.pt \
  --checkpoint 3=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-52-55_mujoco/model_0.pt \
  --checkpoint 1=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-45-55_mujoco/model_25.pt \
  --checkpoint 2=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-49-19_mujoco/model_25.pt \
  --checkpoint 3=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-52-55_mujoco/model_25.pt \
  --checkpoint 1=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-45-55_mujoco/model_50.pt \
  --checkpoint 2=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-49-19_mujoco/model_50.pt \
  --checkpoint 3=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-52-55_mujoco/model_50.pt \
  --checkpoint 1=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-45-55_mujoco/model_75.pt \
  --checkpoint 2=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-49-19_mujoco/model_75.pt \
  --checkpoint 3=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-52-55_mujoco/model_75.pt \
  --checkpoint 1=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-45-55_mujoco/model_100.pt \
  --checkpoint 2=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-49-19_mujoco/model_100.pt \
  --checkpoint 3=/tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-52-55_mujoco/model_100.pt \
  --output /tmp/point_goal_m54_ppo_sweep_4096_final.json
```

The final raw report contains all per-episode results and every checkpoint
SHA-256. Its file SHA-256 is:

```text
522f8290a627551d30641300790a6112a4741785f5e08b9e2333fc4dad212d71
```

Checkpoints, TensorBoard events, run metadata, and the 18 MiB raw evaluation
report remain under `/tmp` and are not committed.

## Next milestone: M5.5 trajectory analysis

Required work:

- record terminal-safe trajectories for successful and failed episodes;
- compute path length and SPL where applicable;
- retain trajectory episode IDs and initial conditions;
- add machine-readable trajectory summaries and bounded visualization output;
- compare representative successful and failed behavior without cherry-picking.
