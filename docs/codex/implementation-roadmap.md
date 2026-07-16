# UniLab Navigation + SLAM Implementation Roadmap

## Definition of success

The final project must demonstrate three independently testable capabilities:

1. Navigation tasks can be added without rewriting the simulator, trainer, and evaluator.
2. RL algorithms can share the same task and evaluation contract.
3. Localization/SLAM estimates can replace ground-truth state through a stable provider interface.

A fourth claim—UniLab is faster than a standalone implementation—must be treated as an experimental result, not a required predetermined outcome.

## M5.2 Fixed-episode evaluator

Deliverables:

- `src/unilab/evaluation/point_goal.py`
- `src/unilab/envs/navigation/diff_drive/controllers.py`
- `scripts/evaluate_point_goal.py`
- unit and integration tests

Policies:

- zero;
- random;
- heuristic;
- PPO checkpoint.

Metrics:

- episode count;
- success rate;
- timeout rate;
- mean and standard deviation of initial distance;
- final distance;
- progress ratio;
- episode length;
- successful episode length;
- optional path length and SPL when path tracking is added.

Acceptance:

- identical initial conditions across policies;
- deterministic results for a fixed seed;
- no autoreset contamination;
- machine-readable JSON output;
- human-readable console table.

## M5.3 Baseline evaluation

Run zero, random, and heuristic policies on at least 4096 fixed episodes.

Acceptance:

- zero policy behaves as a stationary lower-bound check;
- random policy provides a stochastic lower bound;
- heuristic policy solves the obstacle-free task at a high success rate;
- failures are saved with seed and initial condition.

## M5.4 PPO evaluation

Train with the formal PPO config and evaluate checkpoints independently.

Minimum experiment:

- seeds: 1, 2, 3;
- checkpoints every 25 iterations;
- evaluation on a fixed held-out initial-condition set;
- report mean ± standard deviation.

Do not use rollout-window episode logs as the final success rate.

## M6 Obstacle navigation

Phases:

1. static obstacle scene;
2. collision detection;
3. collision termination and metrics;
4. collision penalty;
5. randomized obstacle layouts;
6. obstacle-aware observations.

Acceptance:

- collision metrics are episode-level;
- goal and obstacle sampling cannot overlap;
- tests cover success, timeout, and collision;
- heuristic baseline is updated.

## M6.3 LiDAR observation

Implement a backend-independent LiDAR observation provider.

Contract:

- fixed beam count;
- explicit angular range;
- range clipping;
- normalized output;
- validity mask where needed;
- deterministic sensor noise configuration;
- identical observation shape across backends.

Acceptance:

- analytical scene tests;
- MuJoCo integration tests;
- finite values under all valid scenes;
- no Python loop over environments in the hot path unless benchmarked and justified.

## M7 RL algorithm adapters

Order:

1. PPO baseline remains the reference.
2. Add SAC for continuous actions.
3. Add TD3 if its implementation adds independent value.
4. Add discrete methods only through an explicit action discretization task variant.

Every adapter must support:

- train;
- checkpoint save/load;
- inference;
- fixed-episode evaluation;
- deterministic seed;
- identical task config;
- algorithm-specific config separated from task config.

## M8 SLAM/localization adapter

Phases:

1. ground-truth pose provider;
2. noisy/dead-reckoning provider;
3. offline recorded-estimate provider;
4. online estimator plugin;
5. ROS2 bridge.

Core interface:

```text
sensor packet -> estimator/provider -> pose/covariance/status -> observation builder
```

Do not make the environment import a specific SLAM package.

## M9 ROS2 and TurtleBot 4

Deliver:

- Twist command bridge;
- odometry/localization subscription;
- LiDAR subscription;
- reset and episode-control interface;
- timestamp and frame validation;
- simulation and real-robot configuration separation.

## M10 Framework-versus-standalone benchmark

Create a minimal standalone reference that implements the same PointGoal MDP without UniLab registry/config/wrapper layers.

Control all of:

- simulator version;
- physics model;
- control timestep;
- number of environments;
- rollout length;
- network architecture;
- optimizer;
- seed;
- device;
- precision;
- episode distribution.

Measure:

- simulator steps/s;
- rollout collection time;
- policy-update time;
- end-to-end iterations/s;
- CPU utilization;
- peak RAM;
- peak GPU memory;
- wall-clock time to a fixed success-rate threshold;
- environment steps to the same threshold;
- final held-out success rate.

Report where UniLab is faster, equal, or slower. Optimize only after profiling.

## M11 Reproducibility release

Deliver:

- locked dependency state;
- one-command training;
- one-command evaluation;
- one-command benchmark matrix;
- JSON/CSV result schema;
- seed manifest;
- hardware/software metadata;
- no generated logs in Git.
