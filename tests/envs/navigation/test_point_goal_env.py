"""Contract tests for the kinematic PointGoal NpEnv."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalEnv,
)


def make_env(
    num_envs: int = 2,
) -> tuple[DiffDrivePointGoalEnv, MagicMock]:
    cfg = DiffDrivePointGoalCfg(seed=7)

    backend = MagicMock()
    backend.backend_type = "mujoco"
    backend.step.return_value = None

    env = DiffDrivePointGoalEnv(
        cfg=cfg,
        backend=backend,
        num_envs=num_envs,
    )

    return env, backend


def test_initial_state_has_expected_observation_shapes() -> None:
    env, _ = make_env(num_envs=4)

    state = env.init_state()

    assert state.obs["obs"].shape == (4, 5)
    assert state.obs["critic"].shape == (4, 5)
    assert env.action_space.shape == (2,)


def test_reset_samples_goals_inside_configured_range() -> None:
    env, _ = make_env(num_envs=16)

    env.init_state()

    distances = np.linalg.norm(env.goals, axis=1)

    assert np.all(
        distances >= env.cfg.min_goal_distance
    )
    assert np.all(
        distances <= env.cfg.max_goal_distance
    )


def test_apply_action_scales_normalized_commands() -> None:
    env, _ = make_env(num_envs=2)
    state = env.init_state()

    controls = env.apply_action(
        np.array(
            [
                [-1.0, -1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
        state,
    )

    expected = np.array(
        [
            [0.0, -1.5],
            [0.5, 1.5],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        controls,
        expected,
        atol=1.0e-6,
    )


def test_step_moves_robot_toward_goal_and_rewards_progress() -> None:
    env, backend = make_env(num_envs=1)
    env.init_state()

    env.robot_states[:] = np.array(
        [[0.0, 0.0, 0.0]]
    )
    env.goals[:] = np.array(
        [[1.0, 0.0]]
    )
    env.previous_distance[:] = 1.0

    state = env.step(
        np.array([[1.0, 0.0]], dtype=np.float32)
    )

    # v = 0.5 m/s and ctrl_dt = 0.1 s, so x increases by 0.05 m.
    assert env.robot_states[0, 0] == pytest.approx(0.05)
    assert state.info["distance_to_goal"][0] == pytest.approx(0.95)

    # progress reward:
    # 2.0 * (1.0 - 0.95) - 0.01 = 0.09
    assert state.reward[0] == pytest.approx(0.09)

    backend.step.assert_called_once()

    _, called_substeps = backend.step.call_args.args

    # ctrl_dt / sim_dt = 0.10 / 0.01 = 10
    assert called_substeps == 10


def test_goal_success_sets_terminated_and_preserves_final_observation() -> None:
    env, _ = make_env(num_envs=1)
    env.init_state()

    env.robot_states[:] = np.array(
        [[0.0, 0.0, 0.0]]
    )
    env.goals[:] = np.array(
        [[0.1, 0.0]]
    )
    env.previous_distance[:] = 0.1

    state = env.step(
        np.array([[-1.0, 0.0]], dtype=np.float32)
    )

    assert state.terminated[0]
    assert state.final_observation is not None
    assert state.info["_final_observation"][0]


def test_invalid_action_shape_is_rejected() -> None:
    env, _ = make_env(num_envs=2)
    state = env.init_state()

    with pytest.raises(
        ValueError,
        match="actions must have shape",
    ):
        env.apply_action(
            np.zeros((2, 3)),
            state,
        )
