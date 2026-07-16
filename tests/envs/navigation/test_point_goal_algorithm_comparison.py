"""Tests for controlled PointGoal PPO/SAC/TD3 comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from unilab.evaluation.algorithm_comparison import (
    AlgorithmRunSpec,
    aggregate_point_goal_algorithm_runs,
    format_point_goal_algorithm_comparison,
    parse_algorithm_run_spec,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_run(
    root: Path,
    algorithm: str,
    seed: int,
    *,
    steps: int = 98_304,
    manifest_hash: str = "fixed-manifest",
) -> AlgorithmRunSpec:
    run_dir = root / f"{algorithm}_{seed}"
    checkpoint = run_dir / "model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"{algorithm}-{seed}".encode())
    summary = run_dir / "run_summary.json"
    evaluation = run_dir / "evaluation.json"
    _write_json(
        summary,
        {
            "status": "completed",
            "algo": algorithm,
            "effective_seed": seed,
            "total_env_steps": steps,
            "last_checkpoint": str(checkpoint),
            "training_wall_time_sec": 10.0 + seed,
        },
    )
    _write_json(
        run_dir / "run_config.json",
        {
            "run": {
                "algo": algorithm,
                "task": "DiffDrivePointGoal",
                "sim_backend": "mujoco",
                "device": "cuda",
                "effective_seed": seed,
                "hardware": {"gpu_name": "test-gpu"},
                "git": {"commit": "test-sha", "branch": "test", "dirty": False},
            },
            "config": {
                "training": {"use_amp": False},
                "algo": {"num_envs": 1024},
                "env": {"adaptive_chunk_size": True, "seed": seed},
                "reward": {
                    "progress_scale": 2.0,
                    "success_bonus": 10.0,
                    "time_penalty": 0.01,
                },
            },
        },
    )
    success = 0.1 * seed
    distribution = {"mean": 1.0 + seed, "std": 0.1}
    metrics = {
        "episode_count": 128,
        "success_rate": success,
        "collision_rate": 0.0,
        "timeout_rate": 1.0 - success,
        "initial_distance": distribution,
        "final_distance": distribution,
        "progress_ratio": distribution,
        "episode_length": distribution,
        "successful_episode_length": distribution,
        "path_length": distribution,
        "spl": {"mean": success / 2.0, "std": 0.1},
    }
    _write_json(
        evaluation,
        {
            "config": {
                "task": "DiffDrivePointGoal",
                "sim_backend": "mujoco",
                "episodes": 128,
                "seed": 7301,
                "max_episode_steps": 200,
                "ctrl_dt": 0.1,
                "goal_tolerance": 0.25,
            },
            "evaluation": {"device": "cpu"},
            "checkpoints": {algorithm: str(checkpoint)},
            "manifest": {"sha256": manifest_hash},
            "policies": {
                algorithm: {
                    "manifest_sha256": manifest_hash,
                    "checkpoint_sha256": hashlib.sha256(
                        f"{algorithm}-{seed}".encode()
                    ).hexdigest(),
                    "metrics": metrics,
                }
            },
        },
    )
    return AlgorithmRunSpec(algorithm, seed, summary, evaluation)


def test_parse_algorithm_run_spec() -> None:
    spec = parse_algorithm_run_spec("sac:2:/tmp/summary.json:/tmp/evaluation.json")
    assert spec.algorithm == "sac"
    assert spec.seed == 2
    assert spec.training_summary == Path("/tmp/summary.json")


def test_aggregate_enforces_matrix_budget_and_common_protocol(tmp_path) -> None:
    specs = [
        _make_run(tmp_path, algorithm, seed)
        for algorithm in ("ppo", "sac", "td3")
        for seed in (1, 2, 3)
    ]
    report = aggregate_point_goal_algorithm_runs(
        specs,
        expected_seeds=[1, 2, 3],
        environment_step_budget=98_304,
    )
    assert report["protocol"]["manifest_sha256"] == "fixed-manifest"
    assert report["protocol"]["training_precision"] == "float32"
    assert report["algorithms"]["ppo"]["metrics"]["success_rate"][
        "mean"
    ] == pytest.approx(0.2)
    assert report["algorithms"]["ppo"]["sample_efficiency"][
        "success_rate_at_budget"
    ]["seed_count"] == 3
    assert "env_steps" in format_point_goal_algorithm_comparison(report)


def test_aggregate_rejects_unequal_budget_or_manifest(tmp_path) -> None:
    specs = [
        _make_run(
            tmp_path,
            algorithm,
            seed,
            steps=90_000 if (algorithm, seed) == ("td3", 3) else 98_304,
        )
        for algorithm in ("ppo", "sac", "td3")
        for seed in (1, 2, 3)
    ]
    with pytest.raises(ValueError, match="expected exactly"):
        aggregate_point_goal_algorithm_runs(
            specs,
            expected_seeds=[1, 2, 3],
            environment_step_budget=98_304,
        )

    specs = [
        _make_run(
            tmp_path / "manifest",
            algorithm,
            seed,
            manifest_hash="different" if (algorithm, seed) == ("sac", 2) else "fixed",
        )
        for algorithm in ("ppo", "sac", "td3")
        for seed in (1, 2, 3)
    ]
    with pytest.raises(ValueError, match="protocol differs"):
        aggregate_point_goal_algorithm_runs(
            specs,
            expected_seeds=[1, 2, 3],
            environment_step_budget=98_304,
        )


def test_aggregate_rejects_missing_seed(tmp_path) -> None:
    specs = [
        _make_run(tmp_path, algorithm, seed)
        for algorithm in ("ppo", "sac", "td3")
        for seed in (1, 2, 3)
        if (algorithm, seed) != ("sac", 2)
    ]
    with pytest.raises(ValueError, match="matrix mismatch"):
        aggregate_point_goal_algorithm_runs(
            specs,
            expected_seeds=[1, 2, 3],
            environment_step_budget=98_304,
        )
