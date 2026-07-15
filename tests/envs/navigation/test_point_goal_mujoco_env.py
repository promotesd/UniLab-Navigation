"""Integration tests for the MuJoCo PointGoal environment."""

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
)

pytest.importorskip("mujoco")


def make_env(
    *,
    num_envs: int = 2,
    seed: int = 7,
) -> DiffDrivePointGoalMujocoEnv:
    """Create a small real MuJoCo navigation environment."""
    cfg = DiffDrivePointGoalCfg(seed=seed)

    return DiffDrivePointGoalMujocoEnv(
        cfg=cfg,
        num_envs=num_envs,
        backend_type="mujoco",
    )


def test_mujoco_env_initializes_expected_spaces() -> None:
    env = make_env(num_envs=2)

    state = env.init_state()

    assert env.num_envs == 2
    assert env.action_space.shape == (2,)
    assert state.obs["obs"].shape == (2, 5)
    assert state.obs["critic"].shape == (2, 5)

    assert env.wheel_commands.shape == (2, 2)
    assert env.robot_states.shape == (2, 3)
    assert env.goals.shape == (2, 2)


def test_reset_synchronizes_planar_state_from_backend() -> None:
    env = make_env(num_envs=4)

    env.init_state()

    backend_position = np.asarray(
        env._backend.get_base_pos()
    )
    backend_quaternion = np.asarray(
        env._backend.get_base_quat()
    )

    assert backend_position.shape == (4, 3)
    assert backend_quaternion.shape == (4, 4)

    np.testing.assert_allclose(
        env.robot_states[:, 0:2],
        backend_position[:, 0:2],
        atol=1.0e-6,
    )

    assert np.all(
        env.previous_distance
        >= env.cfg.min_goal_distance
    )
    assert np.all(
        env.previous_distance
        <= env.cfg.max_goal_distance
    )


def test_apply_action_returns_wheel_speed_targets() -> None:
    env = make_env(num_envs=2)
    state = env.init_state()

    wheel_commands = env.apply_action(
        np.array(
            [
                [1.0, 0.0],
                [-1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        state,
    )

    # First action:
    # normalized linear action 1 -> v = 0.5 m/s
    # angular action 0 -> w = 0 rad/s
    # wheel speed = 0.5 / 0.08 = 6.25 rad/s
    np.testing.assert_allclose(
        wheel_commands[0],
        np.array([6.25, 6.25]),
        atol=1.0e-6,
    )

    # Second action:
    # normalized linear action -1 -> v = 0
    # angular action 1 -> w = 1.5 rad/s
    # left/right = [-3, 3] rad/s
    np.testing.assert_allclose(
        wheel_commands[1],
        np.array([-3.0, 3.0]),
        atol=1.0e-6,
    )


def test_step_reads_motion_from_real_mujoco_backend() -> None:
    env = make_env(num_envs=1)
    env.init_state()

    indices = np.array([0], dtype=np.int32)

    qpos = np.tile(
        env._home_qpos,
        (1, 1),
    )
    qvel = np.tile(
        env._home_qvel,
        (1, 1),
    )

    # Start at origin, facing world +x.
    qpos[0, 0:2] = 0.0
    qpos[0, 3:7] = np.array(
        [1.0, 0.0, 0.0, 0.0]
    )

    env._backend.set_state(
        indices,
        qpos,
        qvel,
    )

    env._sync_robot_states_from_backend()

    env.goals[:] = np.array(
        [[2.0, 0.0]]
    )

    env.previous_distance[:] = 2.0

    initial_x = float(
        env.robot_states[0, 0]
    )

    # Run several policy control intervals so actuator response
    # and contact dynamics have time to move the robot.
    state = None

    for _ in range(20):
        state = env.step(
            np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            )
        )

    assert state is not None

    final_x = float(
        env.robot_states[0, 0]
    )

    backend_x = float(
        env._backend.get_base_pos()[0, 0]
    )

    assert final_x > initial_x + 0.05

    assert final_x == pytest.approx(
        backend_x,
        abs=1.0e-6,
    )

    assert state.info[
        "distance_to_goal"
    ][0] < 2.0

    assert np.all(
        np.isfinite(env.robot_states)
    )


def test_real_mujoco_step_uses_wheel_commands_as_ctrl() -> None:
    env = make_env(num_envs=1)
    env.init_state()

    state = env.step(
        np.array(
            [[1.0, 0.0]],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(
        env.wheel_commands,
        np.array([[6.25, 6.25]]),
        atol=1.0e-6,
    )

    assert state.info[
        "wheel_commands"
    ].shape == (1, 2)


def test_partial_reset_only_changes_selected_environments() -> None:
    env = make_env(
        num_envs=3,
        seed=11,
    )
    env.init_state()

    goals_before = env.goals.copy()

    reset_indices = np.array(
        [1],
        dtype=np.int32,
    )

    observation, info = env.reset(
        reset_indices
    )

    assert observation["obs"].shape == (1, 5)
    assert observation["critic"].shape == (1, 5)

    np.testing.assert_allclose(
        env.goals[0],
        goals_before[0],
    )

    np.testing.assert_allclose(
        env.goals[2],
        goals_before[2],
    )

    assert not np.allclose(
        env.goals[1],
        goals_before[1],
    )

    assert info[
        "distance_to_goal"
    ].shape == (1,)


def test_mujoco_env_emits_success_episode_log() -> None:
    """A reached goal must produce one metric value per episode."""
    env = make_env(num_envs=2)
    state = env.init_state()

    # Place both goals exactly at the current robot positions.
    env.goals[:] = env.robot_states[:, 0:2]

    env.initial_distance[:] = np.array(
        [2.0, 3.0],
        dtype=np.float32,
    )

    env.previous_distance[:] = env.initial_distance

    state.info["steps"][:] = np.array(
        [9, 19],
        dtype=np.uint32,
    )

    updated_state = env.update_state(state)

    assert np.all(updated_state.terminated)
    assert "log" in updated_state.info

    log = updated_state.info["log"]

    np.testing.assert_allclose(
        log["Navigation/success_rate"],
        np.array([1.0, 1.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/timeout_rate"],
        np.array([0.0, 0.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/final_distance"],
        np.array([0.0, 0.0]),
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        log["Navigation/episode_length"],
        np.array([10.0, 20.0]),
    )


def test_mujoco_env_emits_timeout_episode_log() -> None:
    """An unfinished episode at the limit must be logged as timeout."""
    env = make_env(num_envs=2)
    state = env.init_state()

    env.goals[:] = (
        env.robot_states[:, 0:2]
        + np.array(
            [
                [2.0, 0.0],
                [3.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    env.initial_distance[:] = np.array(
        [2.0, 3.0],
        dtype=np.float32,
    )

    env.previous_distance[:] = env.initial_distance

    assert env.cfg.max_episode_steps is not None

    state.info["steps"][:] = (
        env.cfg.max_episode_steps - 1
    )

    updated_state = env.update_state(state)

    assert not np.any(updated_state.terminated)
    assert "log" in updated_state.info

    log = updated_state.info["log"]

    np.testing.assert_allclose(
        log["Navigation/success_rate"],
        np.array([0.0, 0.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/timeout_rate"],
        np.array([1.0, 1.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/episode_length"],
        np.array([200.0, 200.0]),
    )

