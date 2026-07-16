---
name: unilab-benchmark
description: Build fair navigation evaluation and performance benchmarks for UniLab versus standalone implementations. Use for fixed-episode evaluation, baseline controllers, checkpoint comparison, throughput profiling, sample-efficiency studies, time-to-threshold analysis, multi-seed experiments, result schemas, or claims that UniLab is faster.
metadata:
  short-description: Evaluate learning and prove performance claims fairly
---

# UniLab Benchmark

## Two benchmark layers

Keep task effectiveness separate from systems performance.

Task effectiveness:

- success rate;
- collision rate;
- timeout rate;
- final distance;
- progress ratio;
- episode length;
- path length;
- SPL;
- return.

Systems performance:

- simulator steps/s;
- rollout collection time;
- update time;
- end-to-end iteration time;
- CPU utilization;
- RAM;
- GPU utilization;
- peak GPU memory;
- wall-clock time to threshold.

## Fixed-episode evaluator

For policy comparisons:

1. generate a deterministic initial-condition manifest;
2. reuse the exact manifest for every policy;
3. disable second-episode autoreset contamination;
4. evaluate a fixed episode count;
5. save JSON with config, seed, checkpoint, Git SHA, and metrics.

Compare zero, random, heuristic, and learned policies.

Do not infer global success rate from one rollout window's episode extras.

## Framework versus standalone

The standalone reference must use the same:

- MuJoCo model and version;
- control and simulation timestep;
- environment count;
- reset distribution;
- observation/action mapping;
- reward;
- network;
- optimizer;
- rollout length;
- total steps;
- precision;
- device;
- seed.

Warm up both systems. Randomize execution order across repetitions where practical.

## Statistical protocol

Minimum:

- three training seeds;
- multiple performance repetitions;
- mean and standard deviation;
- raw result files;
- hardware/software manifest.

Use confidence intervals when enough repetitions exist.

## Claim policy

Allowed:

- “UniLab achieved X steps/s versus Y under this protocol.”
- “UniLab reached 90% success in fewer environment steps.”
- “The framework added Z% overhead in this configuration.”

Not allowed:

- “UniLab is faster” without defining workload and metric;
- selecting only favorable environment counts;
- changing network size or physics accuracy between systems;
- omitting failed seeds.

## Profiling before optimization

Profile the hot path before changing architecture. Attribute time to:

- action conversion;
- backend stepping;
- state synchronization;
- observation/reward;
- autoreset;
- tensor conversion;
- policy inference;
- optimization;
- logging.

Optimize the largest measured component first.
