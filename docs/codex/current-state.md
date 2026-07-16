# UniLab Navigation + SLAM Current State

Last updated: 2026-07-16

## Repository state

- Active branch: `feat/navigation-mvp`
- Remote: `git@github.com:promotesd/UniLab-Navigation.git`
- M5.1 episode metrics: complete in `f068ebdd`
- M5.2 fixed-episode evaluator: complete in `cbac3705`
- Earliest incomplete milestone after this update: M7.1 SAC adapter

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

## M5.5 trajectory and path-efficiency analysis

Status: complete.

The fixed-episode evaluator now computes path length and SPL for every policy,
whether or not raw trajectories are retained. Optional trajectory recording
stores the initial state and one sample after each active step, then stops at
the first terminal step so no post-terminal motion contaminates the path.

Real analysis protocol:

- policy: PPO seed 3, iteration 25;
- held-out episodes: 128 from manifest seed 1001;
- checkpoint selection: declared in advance because its formal 4,096-episode
  success was below 100%, allowing real success and failure analysis;
- success: 127/128 (`0.9922`);
- timeout: 1/128 (`0.0078`);
- mean path length: `3.0904 ± 1.0623 m`;
- mean SPL: `0.9098 ± 0.1621`;
- mean episode length: `116.2344 ± 39.6746` steps.

Representative selection is deterministic and declared rather than visual
cherry-picking:

- successful trajectories: lower median SPL over all successes;
- failed trajectories: lower median progress ratio over all failures;
- all raw episodes remain in the source report.

For this run the selected success was episode 9 (`path=4.685 m`, `SPL=1.0`),
and the selected failure was episode 14 (`path=2.346 m`, `SPL=0.0`, progress
ratio `0.7519`). The SVG displays both complete paths, starts, and goals.

Commands:

```bash
uv run python scripts/evaluate_point_goal.py \
  --manifest-input /tmp/point_goal_m53_manifest_128.json \
  --policies ppo \
  --checkpoint /tmp/unilab_m54_formal_logs/DiffDrivePointGoal/2026-07-16_18-52-55_mujoco/model_25.pt \
  --device cuda \
  --record-trajectories \
  --output /tmp/point_goal_m55_ppo_seed3_iter25_trajectories.json

uv run python scripts/analyze_point_goal_trajectories.py \
  --input /tmp/point_goal_m55_ppo_seed3_iter25_trajectories.json \
  --output /tmp/point_goal_m55_trajectory_analysis.json \
  --svg /tmp/point_goal_m55_trajectory_analysis.svg
```

Artifact SHA-256 values:

```text
e22ab2eb7ac9267db95f19c8ed0dcc95bf191a52e5181222922d2d07106b677e  trajectory report
ad210b37cfa4cdb8212dca2e9795272eaf588909064aa58ba55b5db6275ee02f  analysis JSON
744072122a1e8bf42a418f419c3e884d1ebe5346b54f8b843eb6da8231af2f83  SVG
```

The 3.9 MiB trajectory report and generated analysis/visualization remain under
`/tmp` and are not committed.

## M6.1 static obstacle navigation

Status: complete.

Delivered:

- separate `DiffDrivePointGoalObstacles` registry/config/task variant;
- composed real MuJoCo scene with a static box centered at `(2.0, 0.0)` and
  planar half-extents `(0.35, 0.75) m`;
- vectorized axis-aligned point/clearance geometry;
- bounded vectorized rejection sampling for obstacle-clear goals;
- validation that fixed and evaluator-provided starts/goals do not overlap the
  configured obstacle clearance;
- backend-independent obstacle centers and half-extents in reset/step task info;
- dedicated Hydra task composition with training seed interpolation;
- unchanged obstacle-free `DiffDrivePointGoal` task contract.

Bounded real MuJoCo heuristic smoke:

```text
episodes: 128
success rate: 0.8828125
timeout rate: 0.1171875
```

The heuristic has no obstacle observation and M6.1 intentionally has no
collision termination yet. Its 11.7% timeout rate is therefore expected
baseline behavior, not a collision metric or obstacle-avoidance claim.

## M6.2 collision semantics

Status: complete.

Delivered:

- real robot/obstacle contact detection from a MuJoCo touch sensor attached to
  the static obstacle;
- mutually exclusive success, collision, and timeout outcomes with goal success
  taking same-step precedence over collision and collision over timeout;
- configurable contact-force threshold and collision reward penalty scoped to
  `DiffDrivePointGoalObstacles`;
- episode collision metrics emitted at the terminal transition before autoreset;
- fixed-episode JSON and console collision-rate reporting;
- collision-aware PPO sweep aggregation and trajectory summary counts;
- analytical precedence tests plus real MuJoCo forced-contact and evaluator
  integration tests.

Bounded real MuJoCo heuristic evaluator smoke (`128` fixed episodes, seed `61`,
autoreset disabled):

```text
success rate:   0.859375
collision rate: 0.140625
timeout rate:   0.0
```

The three terminal rates sum to one. The heuristic remains goal-directed and
has no obstacle observation, so this run validates collision accounting rather
than obstacle avoidance.

## M6.3 LiDAR observation

Status: complete.

Delivered:

- backend-independent, vectorized planar ray/AABB intersection provider;
- configurable fixed beam count, explicit relative angular interval, minimum and
  maximum ranges, and seeded Gaussian sensor noise;
- physical ranges clipped to the sensor interval and normalized policy ranges
  mapping the minimum to zero and maximum/no-hit to one;
- no validity mask because every configured ray has a valid finite clipped
  reading, including no-hit rays;
- `16` full-circle beams appended to the obstacle task's five PointGoal and
  command features, yielding a fixed `21`-dimensional actor/critic observation;
- raw ranges and batched beam angles in task info for inspection and adapters;
- Hydra and nested registry overrides for sensor configuration;
- analytical geometry, batching, clipping, rotation, deterministic-noise, and
  validation tests plus real MuJoCo scene/observation integration tests;
- a hot path vectorized over environments, beams, and obstacles without a
  per-environment Python loop.

Validation:

```text
Ruff: clean
complete navigation suite: 120 passed
real MuJoCo smoke: 128 environments, observation (128, 21), LiDAR (128, 16)
LiDAR finite: true
LiDAR observed range: [0.220247, 5.0] m
```

The real smoke reused the fixed seed-`61` obstacle manifest and preserved the
M6.2 terminal rates (`0.859375` success, `0.140625` collision, `0.0` timeout),
as expected because the existing goal-only heuristic does not yet consume the
new beams.

## M6.4 valid randomization

Status: complete.

Delivered:

- deterministic per-environment obstacle-center and robot-start sampling with
  bounded vectorized rejection;
- start/obstacle and goal/obstacle clearance checks for every layout; the
  current scene has exactly one obstacle, so no obstacle/obstacle pair exists;
- planar obstacle slide-joint state in every MuJoCo environment, with a real
  frame-position sensor proving sampled metadata and collision geometry match;
- fixed obstacle count, half extents, LiDAR beam count, and observation shape;
- seeded LiDAR noise enabled in the formal obstacle Hydra task;
- evaluator manifests extended with obstacle centers and half extents while
  preserving the prior obstacle-free JSON/hash format;
- exact obstacle-layout manifest replay into a different environment seed;
- a stateful LiDAR heuristic that turns toward the clearer side of a frontal
  hazard and releases after clearance;
- analytical, config, serialization, deterministic replay, and real MuJoCo
  integration coverage.

Formal fixed-manifest real MuJoCo evidence:

- episodes: `4,096`;
- layout/manifest seed: `73`;
- autoreset: disabled;
- manifest SHA-256:
  `f872f11d57797c51dd255691a669d7ba39dbc7050a8880482cbd7be8a22ae30e`.

| Policy | Success | Collision | Timeout | SPL mean |
| --- | ---: | ---: | ---: | ---: |
| Goal-only heuristic | 0.972168 | 0.027832 | 0.000000 | 0.944502 |
| LiDAR heuristic | 0.983154 | 0.002686 | 0.014160 | 0.955176 |

Validation:

```text
Ruff: clean
complete navigation suite: 123 passed
JSON: /tmp/point_goal_m64_randomized_4096.json
JSON SHA-256: 051ceebd217aa69385bdad131b59b8f5daac7ab273fbec30e1de8a0aa3acfe9c
```

The evidence JSON remains under `/tmp` and is not committed.

## Next milestone: M7.1 SAC adapter

Required work:

- preserve the PointGoal task, manifest, and evaluator contracts;
- add SAC training, checkpoint save/load, and deterministic inference;
- separate algorithm-specific configuration from task configuration;
- compare SAC to PPO using identical held-out fixed episodes and seeds;
- add unit, integration, training-smoke, and checkpoint-reload coverage.
