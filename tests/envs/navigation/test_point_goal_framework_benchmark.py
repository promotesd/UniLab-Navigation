"""Numerical and schema tests for framework-versus-standalone PointGoal."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.benchmarks import (
    StandalonePointGoalMujoco,
    StandalonePointGoalSpec,
    aggregate_point_goal_framework_benchmark,
)
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
)
from unilab.evaluation import generate_point_goal_manifest


def test_standalone_point_goal_matches_framework_real_mujoco_step_contract() -> None:
    environment_count = 8
    cfg = DiffDrivePointGoalCfg(chunk_size=2, adaptive_chunk_size=False)
    framework = DiffDrivePointGoalMujocoEnv(cfg=cfg, num_envs=environment_count)
    framework.set_autoreset(False)
    standalone = StandalonePointGoalMujoco(
        StandalonePointGoalSpec(chunk_size=2),
        environment_count,
    )
    manifest = generate_point_goal_manifest(
        episode_count=environment_count,
        seed=1001,
        min_goal_distance=cfg.min_goal_distance,
        max_goal_distance=cfg.max_goal_distance,
    )
    framework_state = framework.reset_to_initial_conditions(
        manifest.robot_states,
        manifest.goals,
    )
    standalone_observation = standalone.reset_to_initial_conditions(
        manifest.robot_states,
        manifest.goals,
    )
    np.testing.assert_allclose(
        framework_state.obs["obs"],
        standalone_observation,
        atol=1.0e-6,
    )

    rng = np.random.default_rng(44)
    for _ in range(6):
        actions = rng.uniform(-1.0, 1.0, size=(environment_count, 2)).astype(np.float32)
        framework_state = framework.step(actions)
        observation, reward, terminated, truncated = standalone.step(actions)
        np.testing.assert_allclose(framework_state.obs["obs"], observation, atol=1.0e-5)
        np.testing.assert_allclose(framework_state.reward, reward, atol=1.0e-5)
        np.testing.assert_array_equal(framework_state.terminated, terminated)
        np.testing.assert_array_equal(framework_state.truncated, truncated)
        np.testing.assert_allclose(
            framework_state.info["robot_state"],
            standalone.robot_states,
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            framework_state.info["wheel_commands"],
            standalone.wheel_commands,
            atol=1.0e-6,
        )
    framework.close()
    standalone.close()


def benchmark_run(path: str, seed: int, repetition: int, scale: float) -> dict:
    return {
        "path": path,
        "seed": seed,
        "repetition": repetition,
        "simulator_seconds": 1.0 * scale,
        "simulator_steps_per_second": 100.0 / scale,
        "training_iteration_seconds": 2.0 * scale,
        "training_steps_per_second": 50.0 / scale,
        "reward_checksum": float(seed),
    }


def test_framework_benchmark_aggregate_requires_pairs_and_reports_overhead() -> None:
    runs = [
        benchmark_run(path, seed, repetition, 2.0 if path == "framework" else 1.0)
        for seed in (1, 2, 3)
        for repetition in (0, 1)
        for path in ("framework", "standalone")
    ]
    aggregate = aggregate_point_goal_framework_benchmark(runs)
    assert aggregate["paths"]["framework"]["simulator_seconds"]["mean"] == 2.0
    assert aggregate["paths"]["standalone"]["simulator_seconds"]["mean"] == 1.0
    assert aggregate["framework_overhead_percent"]["simulator_time_percent"] == 100.0
    assert (
        aggregate["framework_overhead_percent"]["training_iteration_time_percent"]
        == 100.0
    )
    assert (
        aggregate["contract_checks"]["maximum_paired_reward_checksum_difference"]
        == 0.0
    )
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_point_goal_framework_benchmark(runs[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_point_goal_framework_benchmark([*runs, runs[0]])
