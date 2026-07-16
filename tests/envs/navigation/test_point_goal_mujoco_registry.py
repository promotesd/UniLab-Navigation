"""Registry integration tests for the MuJoCo PointGoal task."""

from unilab.base import registry
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalMujocoEnv,
)


def test_registry_creates_mujoco_point_goal_environment() -> None:
    """Create the complete task through UniLab's public registry."""
    registry.ensure_registries()

    registered = registry.list_registered_envs()

    assert "DiffDrivePointGoal" in registered

    assert registered["DiffDrivePointGoal"][
        "available_backends"
    ] == ["mujoco"]

    env = registry.make(
        "DiffDrivePointGoal",
        sim_backend="mujoco",
        num_envs=2,
    )

    assert isinstance(
        env,
        DiffDrivePointGoalMujocoEnv,
    )

    assert env.num_envs == 2

    state = env.init_state()

    assert state.obs["obs"].shape == (2, 5)
    assert state.obs["critic"].shape == (2, 5)
    assert state.reward.shape == (2,)
    assert state.terminated.shape == (2,)
    assert state.truncated.shape == (2,)
