"""Tests for PointGoal navigation task mathematics."""

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import (
    build_point_goal_observation,
    compute_point_goal_metrics,
    compute_point_goal_reward,
    is_goal_reached,
)


def test_goal_ahead_has_zero_relative_bearing() -> None:
    state = np.array([0.0, 0.0, 0.0])
    goal = np.array([2.0, 0.0])

    distance, bearing = compute_point_goal_metrics(state, goal)

    assert distance == pytest.approx(2.0)
    assert bearing == pytest.approx(0.0)


def test_goal_bearing_is_relative_to_robot_heading() -> None:
    state = np.array([0.0, 0.0, np.pi / 2.0])
    goal = np.array([1.0, 0.0])

    distance, bearing = compute_point_goal_metrics(state, goal)

    assert distance == pytest.approx(1.0)
    assert bearing == pytest.approx(-np.pi / 2.0)


def test_relative_bearing_wraps_across_pi_boundary() -> None:
    robot_heading = np.pi - 0.1
    goal_heading = -np.pi + 0.1

    state = np.array([0.0, 0.0, robot_heading])
    goal = np.array(
        [
            np.cos(goal_heading),
            np.sin(goal_heading),
        ]
    )

    _, bearing = compute_point_goal_metrics(state, goal)

    assert bearing == pytest.approx(0.2)


def test_observation_contains_distance_sine_and_cosine() -> None:
    state = np.array([0.0, 0.0, 0.0])
    goal = np.array([3.0, 4.0])

    observation = build_point_goal_observation(
        state,
        goal,
        max_goal_distance=5.0,
    )

    np.testing.assert_allclose(
        observation,
        np.array([1.0, 0.8, 0.6]),
        atol=1.0e-6,
    )


def test_observation_supports_batched_environments() -> None:
    states = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 2.0],
        ]
    )
    goals = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    observations = build_point_goal_observation(
        states,
        goals,
        max_goal_distance=2.0,
    )

    expected = np.array(
        [
            [0.5, 0.0, 1.0],
            [0.5, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(observations, expected, atol=1.0e-6)


def test_goal_reached_uses_tolerance() -> None:
    distances = np.array([0.1, 0.25, 0.3])

    reached = is_goal_reached(
        distances,
        goal_tolerance=0.25,
    )

    np.testing.assert_array_equal(
        reached,
        np.array([True, True, False]),
    )


def test_reward_is_positive_when_robot_moves_toward_goal() -> None:
    reward = compute_point_goal_reward(
        previous_distance=np.array([2.0]),
        current_distance=np.array([1.5]),
        reached_goal=np.array([False]),
        progress_scale=2.0,
        success_bonus=10.0,
        time_penalty=0.1,
    )

    np.testing.assert_allclose(reward, np.array([0.9]))


def test_reward_adds_success_bonus() -> None:
    reward = compute_point_goal_reward(
        previous_distance=np.array([0.3]),
        current_distance=np.array([0.2]),
        reached_goal=np.array([True]),
        progress_scale=2.0,
        success_bonus=10.0,
        time_penalty=0.1,
    )

    np.testing.assert_allclose(reward, np.array([10.1]))


def test_invalid_state_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="states must have shape"):
        compute_point_goal_metrics(
            np.zeros((4, 2)),
            np.zeros((4, 2)),
        )
