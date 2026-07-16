---
name: unilab-navigation-task
description: Implement or modify UniLab navigation tasks, differential-drive robots, observations, rewards, termination, sensors, MuJoCo integration, registry entries, Hydra task configuration, and navigation tests. Use when adding PointGoal variants, obstacles, collisions, LiDAR, maps, robot dynamics, or task metrics.
metadata:
  short-description: Build UniLab navigation tasks and sensors
---

# UniLab Navigation Task

## Inspect first

Read:

- `AGENTS.md`;
- the task config;
- the base NumPy environment;
- the concrete backend environment;
- registry decorators and package imports;
- relevant Hydra YAML;
- existing tests.

Trace the full construction path:

```text
Hydra -> registry -> config dataclass -> backend -> NpEnv -> wrapper -> runner
```

## Required separation

Keep:

- pure geometry/math in small vectorized functions;
- task configuration in dataclasses;
- backend-independent MDP logic in the base environment;
- MuJoCo-specific state/control access in the MuJoCo subclass;
- algorithm conversion in wrappers;
- evaluation outside the training loop.

Do not import RSL-RL inside a navigation environment.

## Observation changes

For every observation change:

1. define units and frame;
2. define bounds/normalization;
3. update `obs_groups_spec`;
4. update actor and critic construction assumptions;
5. update reset and step paths;
6. test shape, dtype, finite values, and semantic values;
7. run a wrapper integration test.

## Action changes

For every action change:

1. define normalized policy range;
2. define physical command mapping;
3. clip once at the task boundary;
4. validate finite values and shape;
5. document wheel/control conversion;
6. test straight, rotate, arc, and saturation behavior.

## Termination and metrics

Distinguish:

- success termination;
- collision termination;
- time-limit truncation.

Capture episode metrics before autoreset. Never compute the final metric from reset observations.

## Backend hot path

Prefer vectorized NumPy/backend calls over Python loops across environments. If a loop is unavoidable, benchmark it and state why.

## Validation

Run targeted tests for changed modules, then:

```bash
uv run ruff check \
  src/unilab/envs/navigation \
  tests/envs/navigation
```

```bash
uv run pytest tests/envs/navigation -q
```

For backend integration, run a small real MuJoCo test. Do not substitute a mock for the only integration test.
