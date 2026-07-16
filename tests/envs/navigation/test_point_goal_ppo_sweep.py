"""Tests for cross-seed PointGoal PPO checkpoint aggregation."""

from pathlib import Path

import pytest
from scripts import evaluate_point_goal_ppo_sweep as sweep_script

from unilab.evaluation.point_goal_ppo import load_point_goal_ppo_config
from unilab.evaluation.ppo_sweep import (
    aggregate_ppo_checkpoint_results,
    format_ppo_sweep_summary,
    parse_ppo_checkpoint_spec,
)


def result(seed: int, iteration: int, success: float) -> dict:
    distribution = {"mean": 1.0 + seed, "std": 0.1}
    return {
        "seed": seed,
        "iteration": iteration,
        "metrics": {
            "success_rate": success,
            "timeout_rate": 1.0 - success,
            "collision_rate": 0.0,
            "initial_distance": distribution,
            "final_distance": distribution,
            "progress_ratio": distribution,
            "episode_length": distribution,
            "successful_episode_length": distribution,
            "path_length": distribution,
            "spl": distribution,
        },
    }


def test_parse_checkpoint_spec_extracts_seed_and_iteration(tmp_path) -> None:
    spec = parse_ppo_checkpoint_spec(f"3={tmp_path / 'model_75.pt'}")
    assert spec.seed == 3
    assert spec.iteration == 75
    assert spec.path == Path(tmp_path / "model_75.pt").resolve()


def test_checkpoint_hash_is_recordable(tmp_path) -> None:
    checkpoint = tmp_path / "model_25.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert (
        sweep_script._file_sha256(checkpoint)
        == "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )


def test_formal_point_goal_config_links_task_seed_and_checkpoint_cadence() -> None:
    root_dir = Path(__file__).parents[3]
    cfg = load_point_goal_ppo_config(root_dir)
    assert cfg.algo.save_interval == 25
    assert cfg.env.seed == 1
    cfg.algo.seed = 7
    assert cfg.env.seed == 7


@pytest.mark.parametrize("value", ["bad", "x=/tmp/model_1.pt", "1=/tmp/latest.pt"])
def test_parse_checkpoint_spec_rejects_ambiguous_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_ppo_checkpoint_spec(value)


def test_aggregate_requires_all_seeds_and_reports_mean_std() -> None:
    aggregate = aggregate_ppo_checkpoint_results(
        [result(1, 25, 0.25), result(2, 25, 0.5), result(3, 25, 1.0)],
        expected_seeds=[1, 2, 3],
    )
    iteration = aggregate["iterations"]["25"]
    assert iteration["seeds"] == [1, 2, 3]
    assert iteration["metrics"]["success_rate"]["mean"] == pytest.approx(7.0 / 12.0)
    assert iteration["metrics"]["success_rate"]["std"] > 0.0
    assert "25" in format_ppo_sweep_summary(aggregate)


def test_aggregate_rejects_missing_or_duplicate_seed_results() -> None:
    with pytest.raises(ValueError, match="expected"):
        aggregate_ppo_checkpoint_results(
            [result(1, 25, 0.5), result(2, 25, 0.5)],
            expected_seeds=[1, 2, 3],
        )
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_ppo_checkpoint_results(
            [result(1, 25, 0.5), result(1, 25, 0.5)],
            expected_seeds=[1],
        )
