"""Tests for differential-drive planar kinematics."""

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import (
    differential_drive_step,
    wrap_angle,
)


def test_straight_motion_along_world_x() -> None:
    state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    action = np.array([1.0, 0.0], dtype=np.float32)

    next_state = differential_drive_step(state, action, dt=2.0)

    np.testing.assert_allclose(
        next_state,
        np.array([2.0, 0.0, 0.0], dtype=np.float32),
        atol=1.0e-6,
    )


def test_straight_motion_respects_heading() -> None:
    state = np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float64)
    action = np.array([1.0, 0.0], dtype=np.float64)

    next_state = differential_drive_step(state, action, dt=1.0)

    np.testing.assert_allclose(
        next_state,
        np.array([0.0, 1.0, np.pi / 2.0]),
        atol=1.0e-6,
    )


def test_pure_rotation_does_not_change_position() -> None:
    state = np.array([2.0, -1.0, 0.0], dtype=np.float64)
    action = np.array([0.0, 1.0], dtype=np.float64)

    next_state = differential_drive_step(
        state,
        action,
        dt=np.pi / 2.0,
    )

    np.testing.assert_allclose(
        next_state,
        np.array([2.0, -1.0, np.pi / 2.0]),
        atol=1.0e-6,
    )


def test_constant_turn_follows_quarter_circle() -> None:
    state = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    action = np.array([1.0, 1.0], dtype=np.float64)

    next_state = differential_drive_step(
        state,
        action,
        dt=np.pi / 2.0,
    )

    np.testing.assert_allclose(
        next_state,
        np.array([1.0, 1.0, np.pi / 2.0]),
        atol=1.0e-6,
    )


def test_batched_motion_updates_each_environment() -> None:
    states = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 2.0],
        ],
        dtype=np.float64,
    )
    actions = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    next_states = differential_drive_step(states, actions, dt=1.0)

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, np.pi / 2.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(next_states, expected, atol=1.0e-6)


def test_wrap_angle_returns_principal_angle() -> None:
    angles = np.array(
        [
            0.0,
            2.0 * np.pi,
            3.0 * np.pi / 2.0,
        ]
    )

    wrapped = wrap_angle(angles)

    np.testing.assert_allclose(
        wrapped,
        np.array([0.0, 0.0, -np.pi / 2.0]),
        atol=1.0e-6,
    )


def test_invalid_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="states must have shape"):
        differential_drive_step(
            np.zeros((4, 2)),
            np.zeros((4, 2)),
            dt=0.1,
        )

    with pytest.raises(ValueError, match="actions must have shape"):
        differential_drive_step(
            np.zeros((4, 3)),
            np.zeros((4, 3)),
            dt=0.1,
        )


def test_non_positive_dt_is_rejected() -> None:
    with pytest.raises(ValueError, match="dt must be positive"):
        differential_drive_step(
            np.zeros(3),
            np.zeros(2),
            dt=0.0,
        )
