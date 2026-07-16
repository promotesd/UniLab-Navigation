"""Planar kinematics for differential-drive navigation."""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray:
    """Wrap angles to the interval [-pi, pi)."""
    angle_array = np.asarray(angle)
    return (angle_array + np.pi) % (2.0 * np.pi) - np.pi


def differential_drive_step(
    states: np.ndarray,
    actions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance differential-drive states by one constant-control time step.

    State layout:
        states[..., 0] = x position in meters
        states[..., 1] = y position in meters
        states[..., 2] = heading angle in radians

    Action layout:
        actions[..., 0] = linear velocity in meters per second
        actions[..., 1] = angular velocity in radians per second

    Args:
        states: Array with final dimension 3.
        actions: Array with final dimension 2.
        dt: Positive control interval in seconds.

    Returns:
        Updated states with the same batch shape as ``states``.
    """
    state_array = np.asarray(states)
    action_array = np.asarray(actions)

    if state_array.ndim == 0 or state_array.shape[-1] != 3:
        raise ValueError("states must have shape (..., 3)")

    if action_array.ndim == 0 or action_array.shape[-1] != 2:
        raise ValueError("actions must have shape (..., 2)")

    if state_array.shape[:-1] != action_array.shape[:-1]:
        raise ValueError("states and actions must have matching batch dimensions")

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    dtype = np.result_type(
        state_array.dtype,
        action_array.dtype,
        np.float32,
    )

    state_array = state_array.astype(dtype, copy=False)
    action_array = action_array.astype(dtype, copy=False)

    x = state_array[..., 0]
    y = state_array[..., 1]
    heading = state_array[..., 2]

    linear_velocity = action_array[..., 0]
    angular_velocity = action_array[..., 1]

    heading_change = angular_velocity * dt

    # Near-zero angular velocity is treated as straight-line motion.
    moving_straight = np.abs(angular_velocity) < 1.0e-6

    x_straight = x + linear_velocity * np.cos(heading) * dt
    y_straight = y + linear_velocity * np.sin(heading) * dt

    # Avoid division by zero. Values at straight-motion locations are not used.
    safe_angular_velocity = np.where(
        moving_straight,
        np.ones_like(angular_velocity),
        angular_velocity,
    )

    turning_radius = linear_velocity / safe_angular_velocity
    next_heading_unwrapped = heading + heading_change

    x_turn = x + turning_radius * (
        np.sin(next_heading_unwrapped) - np.sin(heading)
    )
    y_turn = y - turning_radius * (
        np.cos(next_heading_unwrapped) - np.cos(heading)
    )

    next_x = np.where(moving_straight, x_straight, x_turn)
    next_y = np.where(moving_straight, y_straight, y_turn)
    next_heading = wrap_angle(next_heading_unwrapped)

    return np.stack(
        [next_x, next_y, next_heading],
        axis=-1,
    )
