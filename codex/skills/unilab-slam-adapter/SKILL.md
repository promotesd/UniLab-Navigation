---
name: unilab-slam-adapter
description: Design and implement SLAM, localization, odometry, map, sensor-stream, dataset-replay, and ROS2 adapters for UniLab navigation. Use when replacing ground-truth pose with estimated pose, adding LiDAR/IMU/odometry packets, integrating an offline trajectory, connecting a SLAM system, or defining map and frame contracts.
metadata:
  short-description: Connect SLAM and localization to UniLab navigation
---

# UniLab SLAM Adapter

## Read before editing

Read `references/slam-contract.md`.

Inspect the current navigation observation builder and backend state access. Identify every place that directly consumes ground-truth pose.

## Architecture

Use this boundary:

```text
sensor source
  -> timestamped sensor packet
  -> localization/SLAM provider
  -> estimate {pose, covariance, status, timestamp}
  -> navigation observation builder
```

The task must depend on a provider interface, not a concrete SLAM package.

## Required providers

Implement in phases:

1. ground-truth provider;
2. configurable noisy-pose/dead-reckoning provider;
3. offline recorded-estimate provider;
4. online estimator plugin;
5. ROS2 provider.

This progression permits testing the interface before importing a large SLAM dependency.

## Frame and timestamp rules

Every estimate must declare:

- parent frame;
- child frame;
- timestamp;
- pose convention;
- quaternion order;
- units;
- validity/status;
- covariance or an explicit absence.

Reject stale, non-finite, or frame-inconsistent estimates. Do not silently reinterpret frames.

## SLAM versus navigation metrics

Keep separate:

SLAM/localization:
- ATE;
- RPE;
- drift;
- update rate;
- latency;
- dropout;
- covariance calibration.

Navigation:
- success;
- collision;
- timeout;
- SPL;
- path length;
- time-to-goal.

Then analyze how estimator quality affects navigation.

## Tests

Use synthetic trajectories with analytically known transforms before real datasets.

Test:

- identity transform;
- known translation/rotation;
- timestamp ordering;
- frame mismatch;
- dropout/stale data;
- covariance shape;
- deterministic offline replay.

Do not require ROS2 for core interface unit tests.
