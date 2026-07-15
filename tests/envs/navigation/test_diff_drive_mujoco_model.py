"""Contract tests for the differential-drive MuJoCo model."""

from pathlib import Path

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import DiffDrivePointGoalCfg

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
