"""Tests for differential-drive wheel control conversion."""

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import twist_to_wheel_speeds


def test_straight_command_produces_equal_wheel_speeds() -> None:
    wheel_speeds = twist_to_wheel_speeds(
        np.array([0.5, 0.0]),
        wheel_radius=0.08,
        wheel_track=0.32,
    )

    np.testing.assert_allclose(
        wheel_speeds,
        np.array([6.25, 6.25]),
        atol=1.0e-6,
    )


def test_positive_rotation_drives_wheels_in_opposite_directions() -> None:
    wheel_speeds = twist_to_wheel_speeds(
        np.array([0.0, 1.0]),
        wheel_radius=0.08,
        wheel_track=0.32,
    )

    np.testing.assert_allclose(
        wheel_speeds,
        np.array([-2.0, 2.0]),
        atol=1.0e-6,
    )


def test_arc_command_produces_different_forward_wheel_speeds() -> None:
    wheel_speeds = twist_to_wheel_speeds(
        np.array([0.4, 1.0]),
        wheel_radius=0.08,
        wheel_track=0.32,
    )

    np.testing.assert_allclose(
        wheel_speeds,
        np.array([3.0, 7.0]),
        atol=1.0e-6,
    )


def test_wheel_conversion_supports_batched_commands() -> None:
    commands = np.array(
        [
            [0.5, 0.0],
            [0.0, 1.0],
            [0.4, 1.0],
        ]
    )

    wheel_speeds = twist_to_wheel_speeds(
        commands,
        wheel_radius=0.08,
        wheel_track=0.32,
    )

    expected = np.array(
        [
            [6.25, 6.25],
            [-2.0, 2.0],
            [3.0, 7.0],
        ]
    )

    np.testing.assert_allclose(
        wheel_speeds,
        expected,
        atol=1.0e-6,
    )


def test_wheel_speeds_are_clipped_to_actuator_limit() -> None:
    wheel_speeds = twist_to_wheel_speeds(
        np.array([10.0, 0.0]),
        wheel_radius=0.08,
        wheel_track=0.32,
        max_wheel_speed=20.0,
    )

    np.testing.assert_allclose(
        wheel_speeds,
        np.array([20.0, 20.0]),
        atol=1.0e-6,
    )


def test_invalid_wheel_radius_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="wheel_radius must be positive",
    ):
        twist_to_wheel_speeds(
            np.zeros(2),
            wheel_radius=0.0,
            wheel_track=0.32,
        )


def test_invalid_command_shape_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="commands must have shape",
    ):
        twist_to_wheel_speeds(
            np.zeros(3),
            wheel_radius=0.08,
            wheel_track=0.32,
        )
