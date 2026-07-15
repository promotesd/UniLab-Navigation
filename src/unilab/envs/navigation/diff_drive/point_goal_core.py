"""Pure NumPy math for differential-drive PointGoal navigation."""

from __future__ import annotations

import numpy as np

from .kinematics import wrap_angle


def compute_point_goal_metrics(
    states: np.ndarray,
    goals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute goal distance and goal bearing in the robot frame.

    State layout:
        states[..., 0] = robot x position
        states[..., 1] = robot y position
        states[..., 2] = robot heading

    Goal layout:
        goals[..., 0] = goal x position
        goals[..., 1] = goal y position

    Returns:
        distance: Euclidean distance from robot to goal.
        bearing: Goal angle relative to robot heading in [-pi, pi).
    """
    state_array = np.asarray(states)
    goal_array = np.asarray(goals)

    if state_array.ndim == 0 or state_array.shape[-1] != 3:
        raise ValueError("states must have shape (..., 3)")

    if goal_array.ndim == 0 or goal_array.shape[-1] != 2:
        raise ValueError("goals must have shape (..., 2)")

    if state_array.shape[:-1] != goal_array.shape[:-1]:
        raise ValueError("states and goals must have matching batch dimensions")

    dtype = np.result_type(
        state_array.dtype,
        goal_array.dtype,
        np.float32,
    )

    state_array = state_array.astype(dtype, copy=False)
    goal_array = goal_array.astype(dtype, copy=False)

    delta = goal_array - state_array[..., :2]

    distance = np.linalg.norm(delta, axis=-1)

    goal_heading_world = np.arctan2(
        delta[..., 1],
        delta[..., 0],
    )

    relative_bearing = np.asarray(
        wrap_angle(goal_heading_world - state_array[..., 2]),
        dtype=dtype,
    )

    return distance, relative_bearing


def build_point_goal_observation(
    states: np.ndarray,
    goals: np.ndarray,
    max_goal_distance: float,
) -> np.ndarray:
    """Build a bounded PointGoal policy observation.

    Observation layout:
        observation[..., 0] = normalized goal distance
        observation[..., 1] = sin(relative goal bearing)
        observation[..., 2] = cos(relative goal bearing)
    """
    if max_goal_distance <= 0.0:
        raise ValueError("max_goal_distance must be positive")

    distance, bearing = compute_point_goal_metrics(states, goals)

    normalized_distance = np.clip(
        distance / max_goal_distance,
        0.0,
        1.0,
    )

    return np.stack(
        [
            normalized_distance,
            np.sin(bearing),
            np.cos(bearing),
        ],
        axis=-1,
    )


def is_goal_reached(
    distance: np.ndarray,
    goal_tolerance: float,
) -> np.ndarray:
    """Return a boolean mask indicating which robots reached their goals."""
    if goal_tolerance <= 0.0:
        raise ValueError("goal_tolerance must be positive")

    distance_array = np.asarray(distance)

    return distance_array <= goal_tolerance


def compute_point_goal_reward(
    previous_distance: np.ndarray,
    current_distance: np.ndarray,
    reached_goal: np.ndarray,
    *,
    progress_scale: float = 1.0,
    success_bonus: float = 10.0,
    time_penalty: float = 0.01,
) -> np.ndarray:
    """Compute progress-based PointGoal navigation reward."""
    if progress_scale < 0.0:
        raise ValueError("progress_scale must be non-negative")

    if success_bonus < 0.0:
        raise ValueError("success_bonus must be non-negative")

    if time_penalty < 0.0:
        raise ValueError("time_penalty must be non-negative")

    previous_array = np.asarray(previous_distance)
    current_array = np.asarray(current_distance)
    reached_array = np.asarray(reached_goal, dtype=bool)

    if previous_array.shape != current_array.shape:
        raise ValueError("previous_distance and current_distance must have matching shapes")

    if reached_array.shape != current_array.shape:
        raise ValueError("reached_goal must match the distance shape")

    dtype = np.result_type(
        previous_array.dtype,
        current_array.dtype,
        np.float32,
    )

    progress = previous_array.astype(dtype) - current_array.astype(dtype)

    return (
        progress_scale * progress
        + success_bonus * reached_array.astype(dtype)
        - time_penalty
    )
