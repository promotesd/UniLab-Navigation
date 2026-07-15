"""Contract tests for the differential-drive MuJoCo model."""

from pathlib import Path

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    twist_to_wheel_speeds,
    wrap_angle,
)

mujoco = pytest.importorskip("mujoco")


def load_model():
    cfg = DiffDrivePointGoalCfg()

    assert cfg.scene is not None

    model_path = Path(cfg.scene.model_file)

    assert model_path.is_file()

    return mujoco.MjModel.from_xml_path(str(model_path))


def test_diff_drive_model_compiles() -> None:
    model = load_model()

    # Free joint: nq=7, nv=6.
    # Two wheel hinge joints add two qpos and two qvel entries.
    assert model.nq == 9
    assert model.nv == 8
    assert model.nu == 2

    cfg = DiffDrivePointGoalCfg()

    expected_ctrl_range = np.array(
        [
            [-cfg.max_wheel_speed, cfg.max_wheel_speed],
            [-cfg.max_wheel_speed, cfg.max_wheel_speed],
        ]
    )

    np.testing.assert_allclose(
        model.actuator_ctrlrange,
        expected_ctrl_range,
        atol=1.0e-6,
    )

    np.testing.assert_array_equal(
        model.actuator_forcelimited,
        np.array([1, 1]),
    )

    np.testing.assert_allclose(
        model.actuator_forcerange,
        np.array(
            [
                [-0.5, 0.5],
                [-0.5, 0.5],
            ]
        ),
        atol=1.0e-6,
    )


@pytest.mark.parametrize(
    ("object_type", "name"),
    [
        (mujoco.mjtObj.mjOBJ_BODY, "base_link"),
        (mujoco.mjtObj.mjOBJ_BODY, "left_wheel"),
        (mujoco.mjtObj.mjOBJ_BODY, "right_wheel"),
        (mujoco.mjtObj.mjOBJ_JOINT, "base_free_joint"),
        (mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint"),
        (mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint"),
        (
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "left_wheel_actuator",
        ),
        (
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_wheel_actuator",
        ),
    ],
)
def test_required_model_names_exist(
    object_type,
    name: str,
) -> None:
    model = load_model()

    object_id = mujoco.mj_name2id(
        model,
        object_type,
        name,
    )

    assert object_id >= 0


def test_model_steps_without_non_finite_state() -> None:
    model = load_model()
    data = mujoco.MjData(model)

    for _ in range(200):
        mujoco.mj_step(model, data)

    assert np.all(np.isfinite(data.qpos))
    assert np.all(np.isfinite(data.qvel))
    assert data.qpos[2] > 0.0


def quaternion_to_yaw(quaternion: np.ndarray) -> float:
    """Extract world-frame yaw from a MuJoCo wxyz quaternion."""
    w, x, y, z = quaternion

    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def simulate_velocity_command(
    linear_velocity: float,
    angular_velocity: float,
    *,
    drive_steps: int = 200,
) -> tuple[np.ndarray, float]:
    """Run one isolated MuJoCo wheel-control experiment."""
    cfg = DiffDrivePointGoalCfg()
    model = load_model()
    data = mujoco.MjData(model)

    # Let gravity and contact constraints settle before measuring motion.
    for _ in range(100):
        mujoco.mj_step(model, data)

    initial_position = data.qpos[:3].copy()
    initial_yaw = quaternion_to_yaw(data.qpos[3:7])

    wheel_speeds = twist_to_wheel_speeds(
        np.array(
            [
                linear_velocity,
                angular_velocity,
            ]
        ),
        wheel_radius=cfg.wheel_radius,
        wheel_track=cfg.wheel_track,
        max_wheel_speed=cfg.max_wheel_speed,
    )

    data.ctrl[:] = wheel_speeds

    for _ in range(drive_steps):
        mujoco.mj_step(model, data)

    position_change = data.qpos[:3] - initial_position

    final_yaw = quaternion_to_yaw(data.qpos[3:7])
    yaw_change = float(
        wrap_angle(final_yaw - initial_yaw)
    )

    return position_change, yaw_change


def test_equal_positive_wheel_speeds_move_robot_forward() -> None:
    position_change, yaw_change = simulate_velocity_command(
        linear_velocity=0.5,
        angular_velocity=0.0,
    )

    assert position_change[0] > 0.20
    assert abs(position_change[1]) < 0.15
    assert abs(yaw_change) < 0.20


def test_opposite_wheel_speeds_rotate_robot_counterclockwise() -> None:
    position_change, yaw_change = simulate_velocity_command(
        linear_velocity=0.0,
        angular_velocity=1.0,
    )

    assert yaw_change > 0.20
    assert np.linalg.norm(position_change[:2]) < 0.30
