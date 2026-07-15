"""Tests for PointGoal episode-level navigation metrics."""

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive.episode_metrics import (
    build_point_goal_episode_log,
)


def test_returns_none_when_no_episode_completed() -> None:
    """Do not log incomplete episodes."""
    log = build_point_goal_episode_log(
        initial_distance=np.array(
            [2.0, 3.0],
            dtype=np.float32,
        ),
        final_distance=np.array(
            [1.5, 2.5],
            dtype=np.float32,
        ),
        reached_goal=np.array(
            [False, False],
        ),
        episode_steps=np.array(
            [20, 40],
            dtype=np.int32,
        ),
        max_episode_steps=200,
    )

    assert log is None


def test_logs_successful_episode() -> None:
    """A robot reaching the goal must be counted as success."""
    log = build_point_goal_episode_log(
        initial_distance=np.array(
            [2.0, 4.0],
            dtype=np.float32,
        ),
        final_distance=np.array(
            [0.2, 3.0],
            dtype=np.float32,
        ),
        reached_goal=np.array(
            [True, False],
        ),
        episode_steps=np.array(
            [30, 50],
            dtype=np.int32,
        ),
        max_episode_steps=200,
    )

    assert log is not None

    np.testing.assert_allclose(
        log["Navigation/success_rate"],
        np.array([1.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/timeout_rate"],
        np.array([0.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/initial_distance"],
        np.array([2.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/final_distance"],
        np.array([0.2]),
    )

    np.testing.assert_allclose(
        log["Navigation/progress_ratio"],
        np.array([0.9]),
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        log["Navigation/episode_length"],
        np.array([30.0]),
    )


def test_logs_timed_out_episode() -> None:
    """An unfinished episode at the step limit must be a timeout."""
    log = build_point_goal_episode_log(
        initial_distance=np.array(
            [5.0],
            dtype=np.float32,
        ),
        final_distance=np.array(
            [3.0],
            dtype=np.float32,
        ),
        reached_goal=np.array(
            [False],
        ),
        episode_steps=np.array(
            [200],
            dtype=np.int32,
        ),
        max_episode_steps=200,
    )

    assert log is not None

    np.testing.assert_allclose(
        log["Navigation/success_rate"],
        np.array([0.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/timeout_rate"],
        np.array([1.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/progress_ratio"],
        np.array([0.4]),
        atol=1.0e-6,
    )


def test_success_on_final_step_is_not_timeout() -> None:
    """Reaching the goal on the final allowed step remains a success."""
    log = build_point_goal_episode_log(
        initial_distance=np.array(
            [1.5],
            dtype=np.float32,
        ),
        final_distance=np.array(
            [0.1],
            dtype=np.float32,
        ),
        reached_goal=np.array(
            [True],
        ),
        episode_steps=np.array(
            [200],
            dtype=np.int32,
        ),
        max_episode_steps=200,
    )

    assert log is not None

    np.testing.assert_allclose(
        log["Navigation/success_rate"],
        np.array([1.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/timeout_rate"],
        np.array([0.0]),
    )


def test_logs_multiple_completed_episodes_only() -> None:
    """Only completed environments should appear in the result."""
    log = build_point_goal_episode_log(
        initial_distance=np.array(
            [2.0, 4.0, 5.0],
            dtype=np.float32,
        ),
        final_distance=np.array(
            [0.2, 3.0, 2.5],
            dtype=np.float32,
        ),
        reached_goal=np.array(
            [True, False, False],
        ),
        episode_steps=np.array(
            [25, 100, 200],
            dtype=np.int32,
        ),
        max_episode_steps=200,
    )

    assert log is not None

    # Environment 0 succeeded.
    # Environment 1 is still running and must be excluded.
    # Environment 2 timed out.
    np.testing.assert_allclose(
        log["Navigation/success_rate"],
        np.array([1.0, 0.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/timeout_rate"],
        np.array([0.0, 1.0]),
    )

    np.testing.assert_allclose(
        log["Navigation/episode_length"],
        np.array([25.0, 200.0]),
    )


def test_rejects_mismatched_shapes() -> None:
    """All episode arrays must describe the same environments."""
    with pytest.raises(
        ValueError,
        match="final_distance must have shape",
    ):
        build_point_goal_episode_log(
            initial_distance=np.array(
                [2.0, 3.0],
                dtype=np.float32,
            ),
            final_distance=np.array(
                [1.0],
                dtype=np.float32,
            ),
            reached_goal=np.array(
                [False, False],
            ),
            episode_steps=np.array(
                [10, 20],
                dtype=np.int32,
            ),
            max_episode_steps=200,
        )
