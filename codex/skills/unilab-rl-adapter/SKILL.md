---
name: unilab-rl-adapter
description: Add or modify reinforcement-learning algorithm adapters for UniLab navigation while preserving a common task and evaluation contract. Use for PPO, SAC, TD3, policy loading, checkpointing, inference, training configs, wrappers, replay buffers, or cross-algorithm comparisons.
metadata:
  short-description: Add RL algorithms without rewriting navigation tasks
---

# UniLab RL Adapter

## Principle

The navigation environment owns the MDP. An algorithm adapter owns tensor conversion, rollout/replay interaction, optimization, checkpointing, and inference.

Do not add algorithm-specific fields to the navigation task unless they are truly task semantics.

## Required adapter capabilities

Every algorithm integration must expose:

- environment creation through the existing registry;
- deterministic seed application;
- train entrypoint;
- checkpoint save/load;
- inference policy;
- fixed-episode evaluator integration;
- algorithm-specific config;
- shape and dtype validation;
- smoke test.

## Algorithm order

1. Keep PPO as the reference on-policy implementation.
2. Add SAC for continuous differential-drive actions.
3. Add TD3 only if the implementation and benchmark provide independent value.
4. Add DQN-like methods only through a separately named discrete-action task variant.

Never compare a continuous-action method with a discretized method without reporting the action-space difference.

## Fair comparison

Hold constant:

- environment;
- initial-condition set;
- observation;
- action range;
- reward;
- episode limit;
- network capacity as closely as algorithmically sensible;
- total environment steps;
- evaluation episodes;
- seed set.

Report both environment-step efficiency and wall-clock efficiency.

## Config boundary

Use Hydra composition:

```text
task config + algorithm config + experiment overrides
```

Do not duplicate task reward values inside every algorithm config.

## Testing

Add tests for:

- adapter construction;
- action shape;
- one train/update cycle;
- checkpoint round trip;
- deterministic inference where supported;
- evaluator compatibility.

Do not use a full research training run as the only test.
