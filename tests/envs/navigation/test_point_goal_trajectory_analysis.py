"""Tests for PointGoal trajectory summaries and SVG rendering."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalEnv,
    HeuristicPointGoalPolicy,
    ZeroPointGoalPolicy,
)
from unilab.evaluation.point_goal import (
    PointGoalManifest,
    evaluate_point_goal_policies,
)
from unilab.evaluation.trajectory_analysis import (
    analyze_point_goal_trajectories,
    write_point_goal_trajectory_svg,
)


def make_env(*, num_envs: int, max_episode_seconds: float) -> DiffDrivePointGoalEnv:
    backend = MagicMock()
    backend.step.return_value = None
    return DiffDrivePointGoalEnv(
        cfg=DiffDrivePointGoalCfg(
            seed=4,
            min_goal_distance=1.0,
            max_goal_distance=2.0,
            max_episode_seconds=max_episode_seconds,
        ),
        backend=backend,
        num_envs=num_envs,
    )


def trajectory_report() -> dict:
    manifest = PointGoalManifest(
        seed=4,
        robot_states=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float32),
        goals=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    return evaluate_point_goal_policies(
        lambda: make_env(num_envs=2, max_episode_seconds=2.0),
        {
            "zero": lambda env: ZeroPointGoalPolicy(),
            "heuristic": lambda env: HeuristicPointGoalPolicy(),
        },
        manifest,
        record_trajectories=True,
    )


def test_analysis_uses_declared_non_cherry_picked_selection_rules() -> None:
    analysis = analyze_point_goal_trajectories(trajectory_report())
    assert "lower medians" in analysis["selection_policy"]
    assert analysis["policies"]["zero"]["failure_count"] == 2
    assert analysis["policies"]["zero"]["collision_count"] == 0
    assert analysis["policies"]["zero"]["timeout_count"] == 2
    assert analysis["policies"]["heuristic"]["success_count"] == 2
    representatives = {
        (entry["policy"], entry["outcome"]): entry for entry in analysis["representatives"]
    }
    assert representatives[("zero", "failure")]["selection_metric"] == "progress_ratio"
    assert representatives[("heuristic", "success")]["selection_metric"] == "spl"


def test_analysis_rejects_reports_without_recorded_trajectories() -> None:
    report = trajectory_report()
    del report["policies"]["zero"]["episodes"][0]["trajectory"]
    with pytest.raises(ValueError, match="does not contain trajectories"):
        analyze_point_goal_trajectories(report)


def test_svg_contains_selected_paths_and_goal_markers(tmp_path) -> None:
    report = trajectory_report()
    analysis = analyze_point_goal_trajectories(report)
    output = write_point_goal_trajectory_svg(report, analysis, tmp_path / "trajectories.svg")
    svg = output.read_text()
    assert svg.startswith("<svg")
    assert "polyline" in svg
    assert "heuristic" in svg
    assert "zero" in svg
