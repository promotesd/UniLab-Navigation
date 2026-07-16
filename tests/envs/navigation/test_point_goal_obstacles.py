"""Unit and real MuJoCo tests for the static-obstacle PointGoal task."""

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from unilab.base import registry
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalObstaclesCfg,
    DiffDrivePointGoalObstaclesMujocoEnv,
    points_clear_batched_obstacles,
    points_clear_static_obstacles,
)
from unilab.evaluation.point_goal import PointGoalManifest, evaluate_point_goal_policy

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


def test_registry_applies_nested_lidar_override_and_observation_shape() -> None:
    registry.ensure_registries()
    env = registry.make(
        "DiffDrivePointGoalObstacles",
        sim_backend="mujoco",
        env_cfg_override={"lidar": {"beam_count": 8, "noise_seed": 19}},
        num_envs=2,
    )
    assert env.cfg.lidar.beam_count == 8
    assert env.cfg.lidar.noise_seed == 19
    assert env.obs_groups_spec == {"obs": 13, "critic": 13}
    state = env.init_state()
    assert state.obs["obs"].shape == (2, 13)
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
    assert cfg.env.randomize_layout
    assert cfg.env.lidar.beam_count == 16
    assert cfg.env.lidar.noise_seed == 23
    assert cfg.env.lidar.noise_std == 0.01


def test_real_mujoco_scene_contains_matching_static_obstacle() -> None:
    cfg = DiffDrivePointGoalObstaclesCfg()
    model = mujoco.MjModel.from_xml_path(str(Path(cfg.scene.model_file)))
    obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "static_obstacle")
    assert obstacle_id >= 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.geom_xpos[obstacle_id, :2], np.array([2.0, 0.0]))
    np.testing.assert_allclose(model.geom_size[obstacle_id, :2], np.array([0.35, 0.75]))
    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "static_obstacle_contact"
    )
    assert sensor_id >= 0


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


def test_randomized_layout_replays_and_matches_real_mujoco_geometry() -> None:
    cfg = DiffDrivePointGoalObstaclesCfg(randomize_layout=True, seed=29)
    first = DiffDrivePointGoalObstaclesMujocoEnv(cfg=cfg, num_envs=64)
    second = DiffDrivePointGoalObstaclesMujocoEnv(
        cfg=DiffDrivePointGoalObstaclesCfg(randomize_layout=True, seed=29),
        num_envs=64,
    )
    first_state = first.init_state()
    second_state = second.init_state()
    np.testing.assert_array_equal(first.robot_states, second.robot_states)
    np.testing.assert_array_equal(first.goals, second.goals)
    np.testing.assert_array_equal(first.obstacle_centers, second.obstacle_centers)
    assert np.unique(first.obstacle_centers[:, 0], axis=0).shape[0] > 32
    assert np.all(
        points_clear_batched_obstacles(
            first.robot_states[:, :2],
            first.obstacle_centers,
            first.obstacle_half_extents,
            clearance=cfg.obstacle_clearance,
        )
    )
    assert np.all(
        points_clear_batched_obstacles(
            first.goals,
            first.obstacle_centers,
            first.obstacle_half_extents,
            clearance=cfg.obstacle_clearance,
        )
    )
    np.testing.assert_allclose(
        first_state.info["obstacle_physics_position"][:, :2],
        first.obstacle_centers[:, 0],
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        second_state.info["obstacle_physics_position"][:, :2],
        second.obstacle_centers[:, 0],
        atol=1.0e-6,
    )
    stationary = np.tile(np.array([[-1.0, 0.0]], dtype=np.float32), (64, 1))
    for _ in range(5):
        first_state = first.step(stationary)
    np.testing.assert_allclose(
        first_state.info["obstacle_physics_position"][:, :2],
        first.obstacle_centers[:, 0],
        atol=1.0e-5,
    )
    first.close()
    second.close()


def test_fixed_manifest_round_trips_and_replays_randomized_obstacle_layouts() -> None:
    source = DiffDrivePointGoalObstaclesMujocoEnv(
        cfg=DiffDrivePointGoalObstaclesCfg(
            randomize_layout=True,
            seed=41,
            max_episode_seconds=0.2,
        ),
        num_envs=8,
    )
    source.init_state()
    manifest = PointGoalManifest(
        seed=41,
        robot_states=source.robot_states.copy(),
        goals=source.goals.copy(),
        obstacle_centers=source.obstacle_centers.copy(),
        obstacle_half_extents=source.obstacle_half_extents.copy(),
    )
    round_trip = PointGoalManifest.from_dict(manifest.to_dict())
    assert round_trip.to_dict() == manifest.to_dict()
    source.close()

    replay = DiffDrivePointGoalObstaclesMujocoEnv(
        cfg=DiffDrivePointGoalObstaclesCfg(
            randomize_layout=True,
            seed=999,
            max_episode_seconds=0.2,
        ),
        num_envs=8,
    )
    result = evaluate_point_goal_policy(
        replay,
        lambda observations: np.tile(
            np.array([[-1.0, 0.0]], dtype=np.float32),
            (len(observations["obs"]), 1),
        ),
        round_trip,
        policy_name="zero",
    )
    np.testing.assert_array_equal(replay.obstacle_centers, manifest.obstacle_centers)
    np.testing.assert_allclose(
        replay.state.info["obstacle_physics_position"][:, :2],
        manifest.obstacle_centers[:, 0],
        atol=1.0e-6,
    )
    assert result["manifest_sha256"] == manifest.to_dict()["sha256"]
    assert result["metrics"]["timeout_rate"] == 1.0
    replay.close()


def test_real_mujoco_obstacle_task_exposes_normalized_fixed_beam_lidar() -> None:
    cfg = DiffDrivePointGoalObstaclesCfg()
    env = DiffDrivePointGoalObstaclesMujocoEnv(cfg=cfg, num_envs=1)
    state = env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[4.0, 0.0]], dtype=np.float32),
    )
    forward_beam = cfg.lidar.beam_count // 2
    assert state.obs["obs"].shape == (1, 5 + cfg.lidar.beam_count)
    assert env.obs_groups_spec == {
        "obs": 5 + cfg.lidar.beam_count,
        "critic": 5 + cfg.lidar.beam_count,
    }
    assert state.info["lidar_ranges"].shape == (1, cfg.lidar.beam_count)
    assert np.all(np.isfinite(state.obs["obs"]))
    np.testing.assert_allclose(
        state.info["lidar_ranges"][0, forward_beam],
        2.0 - 0.35,
        atol=1.0e-5,
    )
    expected_normalized = (2.0 - 0.35 - cfg.lidar.min_range) / (
        cfg.lidar.max_range - cfg.lidar.min_range
    )
    np.testing.assert_allclose(
        state.obs["obs"][0, 5 + forward_beam],
        expected_normalized,
        atol=1.0e-5,
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


def test_real_mujoco_collision_terminates_penalizes_and_logs_before_autoreset() -> None:
    cfg = DiffDrivePointGoalObstaclesCfg(collision_penalty=5.0)
    env = DiffDrivePointGoalObstaclesMujocoEnv(cfg=cfg, num_envs=1)
    env.set_autoreset(False)
    env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[4.0, 0.0]], dtype=np.float32),
    )
    state = env.state
    assert state is not None
    for _ in range(100):
        state = env.step(np.array([[1.0, 0.0]], dtype=np.float32))
        if state.terminated[0]:
            break
    assert state.terminated[0]
    assert state.info["collision"][0]
    assert not state.info["goal_reached"][0]
    assert state.reward[0] < -4.0
    assert state.info["log"]["Navigation/collision_rate"][0] == 1.0
    assert state.info["log"]["Navigation/success_rate"][0] == 0.0
    env.close()


def test_fixed_episode_evaluator_reports_collision_separately() -> None:
    manifest = PointGoalManifest(
        seed=8,
        robot_states=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        goals=np.array([[4.0, 0.0]], dtype=np.float32),
    )
    env = DiffDrivePointGoalObstaclesMujocoEnv(
        cfg=DiffDrivePointGoalObstaclesCfg(), num_envs=1
    )
    result = evaluate_point_goal_policy(
        env,
        lambda observations: np.array([[1.0, 0.0]], dtype=np.float32),
        manifest,
        policy_name="forward",
        record_trajectories=True,
    )
    assert result["metrics"]["success_rate"] == 0.0
    assert result["metrics"]["collision_rate"] == 1.0
    assert result["metrics"]["timeout_rate"] == 0.0
    assert result["metrics"]["spl"]["mean"] == 0.0
    assert result["episodes"][0]["trajectory"][-1]["step"] < 200
    env.close()
