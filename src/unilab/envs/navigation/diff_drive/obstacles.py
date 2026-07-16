"""Static-obstacle geometry and PointGoal task variant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype

from .lidar import PlanarLidarCfg, PlanarLidarProvider
from .point_goal_cfg import DiffDrivePointGoalCfg
from .point_goal_env import DiffDrivePointGoalEnv
from .point_goal_mujoco_env import DiffDrivePointGoalMujocoEnv


def points_clear_static_obstacles(
    points: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_half_extents: np.ndarray,
    *,
    clearance: float,
) -> np.ndarray:
    """Return whether planar points clear every axis-aligned obstacle."""
    point_array = np.asarray(points)
    centers = np.asarray(obstacle_centers)
    half_extents = np.asarray(obstacle_half_extents)
    if point_array.ndim != 2 or point_array.shape[1] != 2:
        raise ValueError("points must have shape (count, 2)")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("obstacle_centers must have shape (obstacles, 2)")
    if half_extents.shape != centers.shape:
        raise ValueError("obstacle_half_extents must match obstacle_centers")
    if clearance < 0.0:
        raise ValueError("clearance must be non-negative")
    if np.any(half_extents <= 0.0):
        raise ValueError("obstacle_half_extents must be positive")
    if (
        not np.all(np.isfinite(point_array))
        or not np.all(np.isfinite(centers))
        or not np.all(np.isfinite(half_extents))
    ):
        raise ValueError("points and obstacle geometry must be finite")
    expanded = half_extents + clearance
    inside = np.all(np.abs(point_array[:, None, :] - centers[None, :, :]) <= expanded, axis=2)
    return ~np.any(inside, axis=1)


@registry.envcfg("DiffDrivePointGoalObstacles")
@dataclass
class DiffDrivePointGoalObstaclesCfg(DiffDrivePointGoalCfg):
    """PointGoal task with one fixed axis-aligned obstacle layout."""

    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(
                ASSETS_ROOT_PATH / "robots" / "diff_drive" / "scene_obstacle.xml"
            )
        )
    )
    obstacle_centers: tuple[tuple[float, float], ...] = ((2.0, 0.0),)
    obstacle_half_extents: tuple[tuple[float, float], ...] = ((0.35, 0.75),)
    obstacle_clearance: float = 0.35
    goal_sample_max_attempts: int = 128
    collision_force_threshold: float = 1.0e-6
    collision_penalty: float = 5.0
    lidar: PlanarLidarCfg = field(default_factory=PlanarLidarCfg)

    def validate(self) -> None:
        super().validate()
        centers = np.asarray(self.obstacle_centers, dtype=float)
        half_extents = np.asarray(self.obstacle_half_extents, dtype=float)
        start_clear = points_clear_static_obstacles(
            np.zeros((1, 2)), centers, half_extents, clearance=self.obstacle_clearance
        )
        if not start_clear[0]:
            raise ValueError("the fixed robot start overlaps the configured obstacle clearance")
        if self.goal_sample_max_attempts <= 0:
            raise ValueError("goal_sample_max_attempts must be positive")
        if self.collision_force_threshold < 0.0:
            raise ValueError("collision_force_threshold must be non-negative")
        if self.collision_penalty < 0.0:
            raise ValueError("collision_penalty must be non-negative")
        self.lidar.validate()


@registry.env("DiffDrivePointGoalObstacles", sim_backend="mujoco")
class DiffDrivePointGoalObstaclesMujocoEnv(DiffDrivePointGoalMujocoEnv):
    """Real MuJoCo PointGoal task with a fixed static obstacle layout."""

    COLLISION_SENSOR_NAME = "static_obstacle_contact"

    def __init__(
        self,
        cfg: DiffDrivePointGoalObstaclesCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        self.OBSERVATION_DIM = (
            DiffDrivePointGoalEnv.OBSERVATION_DIM + cfg.lidar.beam_count
        )
        super().__init__(cfg=cfg, num_envs=num_envs, backend_type=backend_type)
        dtype = get_global_dtype()
        centers = np.asarray(cfg.obstacle_centers, dtype=dtype)
        half_extents = np.asarray(cfg.obstacle_half_extents, dtype=dtype)
        self.obstacle_centers = np.broadcast_to(
            centers, (num_envs,) + centers.shape
        ).copy()
        self.obstacle_half_extents = np.broadcast_to(
            half_extents, (num_envs,) + half_extents.shape
        ).copy()
        self._lidar = PlanarLidarProvider(cfg.lidar)
        self.lidar_ranges = np.full(
            (num_envs, cfg.lidar.beam_count), cfg.lidar.max_range, dtype=dtype
        )
        self.lidar_observation = np.ones(
            (num_envs, cfg.lidar.beam_count), dtype=dtype
        )

    def _sample_goal_positions(self, origins: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        assert isinstance(cfg, DiffDrivePointGoalObstaclesCfg)
        goals = np.zeros_like(origins)
        pending = np.ones(len(origins), dtype=bool)
        centers = np.asarray(cfg.obstacle_centers, dtype=origins.dtype)
        half_extents = np.asarray(cfg.obstacle_half_extents, dtype=origins.dtype)
        for _ in range(cfg.goal_sample_max_attempts):
            pending_indices = np.flatnonzero(pending)
            if len(pending_indices) == 0:
                return goals
            distances = self._rng.uniform(
                cfg.min_goal_distance, cfg.max_goal_distance, size=len(pending_indices)
            )
            angles = self._rng.uniform(-np.pi, np.pi, size=len(pending_indices))
            candidates = origins[pending_indices] + np.stack(
                [distances * np.cos(angles), distances * np.sin(angles)], axis=1
            )
            valid = points_clear_static_obstacles(
                candidates,
                centers,
                half_extents,
                clearance=cfg.obstacle_clearance,
            )
            accepted = pending_indices[valid]
            goals[accepted] = candidates[valid]
            pending[accepted] = False
        raise RuntimeError(
            f"could not sample {int(np.count_nonzero(pending))} valid obstacle-clear goals "
            f"within {cfg.goal_sample_max_attempts} attempts"
        )

    def _build_task_info(self, env_indices: np.ndarray | None = None) -> dict[str, Any]:
        if env_indices is None:
            centers = self.obstacle_centers
            half_extents = self.obstacle_half_extents
        else:
            centers = self.obstacle_centers[env_indices]
            half_extents = self.obstacle_half_extents[env_indices]
        beam_angles = np.broadcast_to(
            self._lidar.beam_angles,
            (len(centers), self._lidar.cfg.beam_count),
        )
        return {
            "obstacle_centers": centers.copy(),
            "obstacle_half_extents": half_extents.copy(),
            "lidar_ranges": (
                self.lidar_ranges.copy()
                if env_indices is None
                else self.lidar_ranges[env_indices].copy()
            ),
            "lidar_beam_angles": beam_angles.copy(),
        }

    def _build_observation(self, env_indices: np.ndarray | None = None) -> np.ndarray:
        point_goal = super()._build_observation(env_indices)
        if env_indices is None:
            states = self.robot_states
            centers = self.obstacle_centers
            half_extents = self.obstacle_half_extents
        else:
            states = self.robot_states[env_indices]
            centers = self.obstacle_centers[env_indices]
            half_extents = self.obstacle_half_extents[env_indices]
        ranges, normalized = self._lidar.scan(states, centers, half_extents)
        if env_indices is None:
            self.lidar_ranges[:] = ranges
            self.lidar_observation[:] = normalized
        else:
            self.lidar_ranges[env_indices] = ranges
            self.lidar_observation[env_indices] = normalized
        return np.concatenate((point_goal, normalized), axis=1)

    def _compute_collision_mask(self) -> np.ndarray:
        cfg = self.cfg
        assert isinstance(cfg, DiffDrivePointGoalObstaclesCfg)
        contact_force = np.asarray(
            self._backend.get_sensor_data(self.COLLISION_SENSOR_NAME)
        ).reshape(self.num_envs, -1)
        return np.any(contact_force > cfg.collision_force_threshold, axis=1)

    def _collision_penalty(self, collision: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        assert isinstance(cfg, DiffDrivePointGoalObstaclesCfg)
        return collision.astype(get_global_dtype()) * cfg.collision_penalty

    def _validate_initial_conditions(
        self,
        robot_states: np.ndarray,
        goals: np.ndarray,
    ) -> None:
        cfg = self.cfg
        assert isinstance(cfg, DiffDrivePointGoalObstaclesCfg)
        centers = np.asarray(cfg.obstacle_centers)
        half_extents = np.asarray(cfg.obstacle_half_extents)
        if not np.all(
            points_clear_static_obstacles(
                robot_states[:, :2],
                centers,
                half_extents,
                clearance=cfg.obstacle_clearance,
            )
        ):
            raise ValueError("initial robot states overlap obstacle clearance")
        if not np.all(
            points_clear_static_obstacles(
                goals,
                centers,
                half_extents,
                clearance=cfg.obstacle_clearance,
            )
        ):
            raise ValueError("initial goals overlap obstacle clearance")
