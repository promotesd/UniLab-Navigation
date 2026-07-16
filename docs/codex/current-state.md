# UniLab Navigation + SLAM Current State

Last updated: 2026-07-16

## Repository state

- Active branch: `feat/navigation-mvp`
- Remote: `git@github.com:promotesd/UniLab-Navigation.git`
- M5.1 episode metrics: complete in `f068ebdd`
- M5.2 fixed-episode evaluator: complete in `cbac3705`
- Earliest incomplete milestone after this update: M5.4 formal multi-seed PPO evidence

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

## Next milestone: M5.4 formal PPO evidence

Required work:

- formal PPO training seeds 1, 2, and 3;
- checkpoints every 25 iterations;
- independent evaluation of each checkpoint on one fixed held-out manifest;
- aggregate mean and standard deviation across seeds;
- no use of rollout-window episode extras as the final success rate.

If the full three-seed experiment cannot finish in one execution window, the
complete training/evaluation pipeline and bounded smoke runs must be committed,
and the exact pending commands and output locations must be recorded here
without inventing results.
