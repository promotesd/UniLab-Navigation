"""Wheel-control mathematics for differential-drive robots."""

from __future__ import annotations

import numpy as np


def twist_to_wheel_speeds(
    commands: np.ndarray,
    *,
    wheel_radius: float,
    wheel_track: float,
    max_wheel_speed: float | None = None,
) -> np.ndarray:
    """Convert body velocity commands into left and right wheel speeds.

    Command layout:
        commands[..., 0] = linear velocity in meters per second
        commands[..., 1] = angular velocity in radians per second

    Output layout:
        wheel_speeds[..., 0] = left wheel angular velocity
        wheel_speeds[..., 1] = right wheel angular velocity

    Args:
        commands: Body velocity commands with shape (..., 2).
        wheel_radius: Positive wheel radius in meters.
        wheel_track: Positive distance between left and right wheels.
        max_wheel_speed: Optional symmetric wheel-speed limit in rad/s.

    Returns:
        Left and right wheel angular velocities with shape (..., 2).
    """
    command_array = np.asarray(commands)

    if command_array.ndim == 0 or command_array.shape[-1] != 2:
        raise ValueError("commands must have shape (..., 2)")

    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive")

    if wheel_track <= 0.0:
        raise ValueError("wheel_track must be positive")

    if max_wheel_speed is not None and max_wheel_speed <= 0.0:
        raise ValueError("max_wheel_speed must be positive")

    if not np.all(np.isfinite(command_array)):
        raise ValueError("commands must contain only finite values")

    dtype = np.result_type(
        command_array.dtype,
        np.float32,
    )

    command_array = command_array.astype(
        dtype,
        copy=False,
    )

    linear_velocity = command_array[..., 0]
    angular_velocity = command_array[..., 1]

    half_track = 0.5 * wheel_track

    left_wheel_speed = (
        linear_velocity - half_track * angular_velocity
    ) / wheel_radius

    right_wheel_speed = (
        linear_velocity + half_track * angular_velocity
    ) / wheel_radius

    wheel_speeds = np.stack(
        [
            left_wheel_speed,
            right_wheel_speed,
        ],
        axis=-1,
    )

    if max_wheel_speed is not None:
        wheel_speeds = np.clip(
            wheel_speeds,
            -max_wheel_speed,
            max_wheel_speed,
        )

    return wheel_speeds


def wheel_speeds_to_twist(
    wheel_speeds: np.ndarray,
    *,
    wheel_radius: float,
    wheel_track: float,
) -> np.ndarray:
    """Convert measured left/right wheel speeds into body linear/angular velocity."""
    speed_array = np.asarray(wheel_speeds)
    if speed_array.ndim == 0 or speed_array.shape[-1] != 2:
        raise ValueError("wheel_speeds must have shape (..., 2)")
    if wheel_radius <= 0.0:
        raise ValueError("wheel_radius must be positive")
    if wheel_track <= 0.0:
        raise ValueError("wheel_track must be positive")
    if not np.all(np.isfinite(speed_array)):
        raise ValueError("wheel_speeds must contain only finite values")
    dtype = np.result_type(speed_array.dtype, np.float32)
    speed_array = speed_array.astype(dtype, copy=False)
    left = speed_array[..., 0]
    right = speed_array[..., 1]
    linear_velocity = 0.5 * wheel_radius * (left + right)
    angular_velocity = wheel_radius * (right - left) / wheel_track
    return np.stack((linear_velocity, angular_velocity), axis=-1)
