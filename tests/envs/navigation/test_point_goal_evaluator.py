"""Unit tests for deterministic fixed-episode PointGoal evaluation."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalEnv,
    HeuristicPointGoalPolicy,
    RandomPointGoalPolicy,
    ZeroPointGoalPolicy,
)
from unilab.evaluation.point_goal import (
    PointGoalManifest,
    evaluate_point_goal_policies,
    evaluate_point_goal_policy,
    format_point_goal_summary,
    generate_point_goal_manifest,
    read_point_goal_manifest,
    write_point_goal_manifest,
    write_point_goal_report,
)


def make_env(*, num_envs: int = 3, max_episode_seconds: float = 0.3):
    cfg = DiffDrivePointGoalCfg(
        seed=5,
        min_goal_distance=1.0,
        max_goal_distance=2.0,
        max_episode_seconds=max_episode_seconds,
    )
    backend = MagicMock()
    backend.step.return_value = None
    return DiffDrivePointGoalEnv(cfg=cfg, backend=backend, num_envs=num_envs)


def test_manifest_is_deterministic_serializable_and_tamper_evident() -> None:
    first = generate_point_goal_manifest(
        episode_count=4, seed=7, min_goal_distance=1.0, max_goal_distance=2.0
    )
    second = generate_point_goal_manifest(
        episode_count=4, seed=7, min_goal_distance=1.0, max_goal_distance=2.0
    )
    assert first.to_dict() == second.to_dict()
    assert PointGoalManifest.from_dict(first.to_dict()).to_dict() == first.to_dict()

    tampered = first.to_dict()
    tampered["initial_conditions"][0]["goal_position"][0] += 0.5
    with pytest.raises(ValueError, match="sha256"):
        PointGoalManifest.from_dict(tampered)


def test_manifest_round_trip_supports_standalone_and_embedded_report(tmp_path) -> None:
    manifest = generate_point_goal_manifest(
        episode_count=4, seed=9, min_goal_distance=1.0, max_goal_distance=2.0
    )
    standalone = write_point_goal_manifest(manifest, tmp_path / "manifest.json")
    assert read_point_goal_manifest(standalone).to_dict() == manifest.to_dict()

    report = tmp_path / "report.json"
    report.write_text(json.dumps({"manifest": manifest.to_dict()}))
    assert read_point_goal_manifest(report).to_dict() == manifest.to_dict()


def test_manifest_rejects_mismatched_count_and_episode_ids() -> None:
    payload = generate_point_goal_manifest(
        episode_count=2, seed=2, min_goal_distance=1.0, max_goal_distance=2.0
    ).to_dict()
    payload.pop("sha256")
    payload["episode_count"] = 3
    with pytest.raises(ValueError, match="episode_count"):
        PointGoalManifest.from_dict(payload)

    payload["episode_count"] = 2
    payload["initial_conditions"][1]["episode_id"] = 4
    with pytest.raises(ValueError, match="episode_id"):
        PointGoalManifest.from_dict(payload)


def test_explicit_reset_installs_exact_conditions_and_clears_episode_state() -> None:
    env = make_env(num_envs=2)
    states = np.array([[0.0, 0.0, 0.25], [0.0, 0.0, -0.5]], dtype=np.float32)
    goals = np.array([[1.5, 0.0], [0.0, -1.25]], dtype=np.float32)
    state = env.reset_to_initial_conditions(states, goals)
    np.testing.assert_array_equal(env.robot_states, states)
    np.testing.assert_array_equal(env.goals, goals)
    np.testing.assert_array_equal(state.info["steps"], np.zeros(2, dtype=np.uint32))
    assert not np.any(state.terminated | state.truncated)


def test_evaluator_disables_autoreset_and_counts_each_slot_once() -> None:
    env = make_env(num_envs=2, max_episode_seconds=0.2)
    manifest = PointGoalManifest(
        seed=1,
        robot_states=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]], dtype=np.float32),
        goals=np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
    )
    result = evaluate_point_goal_policy(
        env, ZeroPointGoalPolicy(), manifest, policy_name="zero"
    )
    assert env._autoreset is False
    assert result["metrics"]["episode_count"] == 2
    assert result["metrics"]["timeout_rate"] == 1.0
    assert len(result["episodes"]) == 2
    assert [episode["episode_length"] for episode in result["episodes"]] == [2, 2]


def test_all_policy_factories_share_one_manifest_and_emit_requested_metrics() -> None:
    manifest = generate_point_goal_manifest(
        episode_count=3, seed=3, min_goal_distance=1.0, max_goal_distance=2.0
    )
    report = evaluate_point_goal_policies(
        lambda: make_env(num_envs=3),
        {
            "zero": lambda env: ZeroPointGoalPolicy(),
            "random": lambda env: RandomPointGoalPolicy(seed=4),
            "heuristic": lambda env: HeuristicPointGoalPolicy(),
        },
        manifest,
    )
    manifest_hash = report["manifest"]["sha256"]
    expected_metrics = {
        "episode_count",
        "success_rate",
        "timeout_rate",
        "initial_distance",
        "final_distance",
        "progress_ratio",
        "episode_length",
        "successful_episode_length",
        "path_length",
        "spl",
    }
    for result in report["policies"].values():
        assert result["manifest_sha256"] == manifest_hash
        assert set(result["metrics"]) == expected_metrics
    assert report["policies"]["zero"]["metrics"]["successful_episode_length"] == {
        "mean": None,
        "std": None,
    }


def test_fixed_seed_evaluation_is_reproducible() -> None:
    manifest = generate_point_goal_manifest(
        episode_count=3, seed=8, min_goal_distance=1.0, max_goal_distance=2.0
    )

    def run():
        return evaluate_point_goal_policies(
            lambda: make_env(num_envs=3),
            {"random": lambda env: RandomPointGoalPolicy(seed=13)},
            manifest,
        )

    assert run() == run()


def test_successful_episode_length_filters_to_successes() -> None:
    env = make_env(num_envs=1, max_episode_seconds=2.0)
    manifest = PointGoalManifest(
        seed=1,
        robot_states=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        goals=np.array([[1.0, 0.0]], dtype=np.float32),
    )
    metrics = evaluate_point_goal_policy(
        env, HeuristicPointGoalPolicy(), manifest, policy_name="heuristic"
    )["metrics"]
    assert metrics["success_rate"] == 1.0
    assert metrics["timeout_rate"] == 0.0
    assert metrics["successful_episode_length"]["mean"] == metrics["episode_length"]["mean"]
    assert metrics["path_length"]["mean"] == pytest.approx(0.75)
    assert metrics["spl"]["mean"] == pytest.approx(1.0)


def test_trajectory_recording_stops_at_each_first_terminal_step() -> None:
    env = make_env(num_envs=2, max_episode_seconds=2.0)
    manifest = PointGoalManifest(
        seed=1,
        robot_states=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]], dtype=np.float32),
        goals=np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
    )
    result = evaluate_point_goal_policy(
        env,
        HeuristicPointGoalPolicy(),
        manifest,
        policy_name="heuristic",
        record_trajectories=True,
    )
    for episode in result["episodes"]:
        trajectory = episode["trajectory"]
        assert trajectory[0]["step"] == 0
        assert trajectory[-1]["step"] == episode["episode_length"]
        assert len(trajectory) == episode["episode_length"] + 1


def test_report_has_console_summary_and_strict_json(tmp_path) -> None:
    manifest = generate_point_goal_manifest(
        episode_count=1, seed=1, min_goal_distance=1.0, max_goal_distance=2.0
    )
    report = evaluate_point_goal_policies(
        lambda: make_env(num_envs=1),
        {"zero": lambda env: ZeroPointGoalPolicy()},
        manifest,
    )
    summary = format_point_goal_summary(report)
    assert "success_len" in summary
    assert "zero" in summary
    output = write_point_goal_report(report, tmp_path / "result.json")
    assert '"success_rate"' in output.read_text()


def test_evaluator_rejects_invalid_policy_action_shape() -> None:
    manifest = generate_point_goal_manifest(
        episode_count=2, seed=1, min_goal_distance=1.0, max_goal_distance=2.0
    )
    with pytest.raises(ValueError, match="policy actions"):
        evaluate_point_goal_policy(
            make_env(num_envs=2),
            lambda obs: np.zeros((2, 3)),
            manifest,
            policy_name="bad",
        )
