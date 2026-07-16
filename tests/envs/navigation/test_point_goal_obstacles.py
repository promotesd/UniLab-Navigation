"""Unit and real MuJoCo tests for the static-obstacle PointGoal task."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalObstaclesCfg,
    DiffDrivePointGoalObstaclesMujocoEnv,
    points_clear_static_obstacles,
)

mujoco = pytest.importorskip("mujoco")


def test_static_obstacle_clearance_geometry_is_vectorized() -> None:
    clear = points_clear_static_obstacles(
        np.array([[0.0, 0.0], [2.0, 0.0], [2.71, 0.0], [2.0, 1.11]]),
        np.array([[2.0, 0.0]]),
        np.array([[0.35, 0.75]]),
        clearance=0.35,
    )
    np.testing.assert_array_equal(clear, np.array([True, False, True, True]))


def test_obstacle_config_rejects_start_clearance_overlap() -> None:
    cfg = DiffDrivePointGoalObstaclesCfg(
        obstacle_centers=((0.1, 0.0),),
        obstacle_half_extents=((0.2, 0.2),),
        obstacle_clearance=0.35,
    )
    with pytest.raises(ValueError, match="fixed robot start overlaps"):
        cfg.validate()


def test_registry_creates_separately_named_obstacle_task() -> None:
    registry.ensure_registries()
    registered = registry.list_registered_envs()
    assert registered["DiffDrivePointGoalObstacles"]["config_class"] == (
        "DiffDrivePointGoalObstaclesCfg"
    )
    env = registry.make(
        "DiffDrivePointGoalObstacles", sim_backend="mujoco", num_envs=4
    )
    assert isinstance(env, DiffDrivePointGoalObstaclesMujocoEnv)
    env.close()


def test_hydra_composes_static_obstacle_training_task() -> None:
    config_dir = Path(__file__).parents[3] / "conf" / "ppo"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=["task=diff_drive_point_goal_obstacles/mujoco", "algo.seed=23"],
        )
    assert cfg.training.task_name == "DiffDrivePointGoalObstacles"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.seed == 23


def test_real_mujoco_scene_contains_matching_static_obstacle() -> None:
    cfg = DiffDrivePointGoalObstaclesCfg()
    model = mujoco.MjModel.from_xml_path(str(Path(cfg.scene.model_file)))
    obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_obstacle")
    assert obstacle_id >= 0
    np.testing.assert_allclose(model.geom_pos[obstacle_id, :2], np.array([2.0, 0.0]))
    np.testing.assert_allclose(model.geom_size[obstacle_id, :2], np.array([0.35, 0.75]))


def test_real_mujoco_reset_samples_obstacle_clear_goals_and_exposes_layout() -> None:
    cfg = DiffDrivePointGoalObstaclesCfg(seed=17)
    env = DiffDrivePointGoalObstaclesMujocoEnv(cfg=cfg, num_envs=128)
    state = env.init_state()
    clear = points_clear_static_obstacles(
        env.goals,
        np.asarray(cfg.obstacle_centers),
        np.asarray(cfg.obstacle_half_extents),
        clearance=cfg.obstacle_clearance,
    )
    assert np.all(clear)
    assert state.info["obstacle_centers"].shape == (128, 1, 2)
    assert state.info["obstacle_half_extents"].shape == (128, 1, 2)
    next_state = env.step(np.tile(np.array([[-1.0, 0.0]]), (128, 1)))
    np.testing.assert_array_equal(
        next_state.info["obstacle_centers"], state.info["obstacle_centers"]
    )
    env.close()


def test_explicit_obstacle_task_manifest_rejects_overlap() -> None:
    env = DiffDrivePointGoalObstaclesMujocoEnv(
        cfg=DiffDrivePointGoalObstaclesCfg(), num_envs=1
    )
    with pytest.raises(ValueError, match="goals overlap obstacle"):
        env.reset_to_initial_conditions(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[2.0, 0.0]], dtype=np.float32),
        )
    env.close()
