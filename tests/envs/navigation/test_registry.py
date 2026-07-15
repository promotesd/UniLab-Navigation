"""Registry tests for navigation tasks."""

import pytest

import unilab.base.registry as registry
from unilab.envs.navigation.diff_drive import DiffDrivePointGoalCfg


def test_diff_drive_point_goal_config_is_registered() -> None:
    registry.ensure_registries()

    registered_envs = registry.list_registered_envs()

    assert "DiffDrivePointGoal" in registered_envs
    assert (
        registered_envs["DiffDrivePointGoal"]["config_class"]
        == "DiffDrivePointGoalCfg"
    )

    # The MuJoCo environment implementation will be added in the next milestone.
    assert registered_envs["DiffDrivePointGoal"]["available_backends"] == []


def test_diff_drive_point_goal_timing() -> None:
    cfg = DiffDrivePointGoalCfg()

    assert cfg.sim_substeps == 10
    assert cfg.max_episode_steps == 200


def test_diff_drive_point_goal_rejects_invalid_distance() -> None:
    cfg = DiffDrivePointGoalCfg(
        goal_tolerance=1.0,
        max_goal_distance=0.5,
    )

    with pytest.raises(
        ValueError,
        match="max_goal_distance must be greater than goal_tolerance",
    ):
        cfg.validate()
