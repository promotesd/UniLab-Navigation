"""Minimal PointGoal MDP over the shared MuJoCo backend, without Env wrappers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base.backend import create_backend
from unilab.base.scene import SceneCfg
from unilab.dtype_config import get_global_dtype


@dataclass(frozen=True)
class StandalonePointGoalSpec:
    """All physics and MDP values controlled by the benchmark protocol."""

    model_file: str = str(
        ASSETS_ROOT_PATH / "robots" / "diff_drive" / "scene.xml"
    )
    sim_dt: float = 0.01
    ctrl_dt: float = 0.1
    max_episode_steps: int = 200
    goal_tolerance: float = 0.25
    max_goal_distance: float = 5.0
    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 1.5
    wheel_radius: float = 0.08
    wheel_track: float = 0.32
    max_wheel_speed: float = 20.0
    progress_scale: float = 2.0
    success_bonus: float = 10.0
    time_penalty: float = 0.01
    chunk_size: int | None = None

    def validate(self) -> None:
        positive = (
            self.sim_dt,
            self.ctrl_dt,
            self.max_episode_steps,
            self.goal_tolerance,
            self.max_goal_distance,
            self.max_linear_velocity,
            self.max_angular_velocity,
            self.wheel_radius,
            self.wheel_track,
            self.max_wheel_speed,
        )
        if not np.all(np.isfinite(positive)) or np.any(np.asarray(positive) <= 0.0):
            raise ValueError("standalone PointGoal positive parameters are invalid")
        ratio = self.ctrl_dt / self.sim_dt
        if not np.isclose(ratio, round(ratio), atol=1.0e-9):
            raise ValueError("standalone ctrl_dt must be an integer multiple of sim_dt")
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ValueError("standalone chunk_size must be positive")


class StandalonePointGoalMujoco:
    """Standalone NumPy MDP loop sharing only UniLab's MuJoCo engine adapter."""

    observation_dim = 5
    action_dim = 2

    def __init__(self, spec: StandalonePointGoalSpec, environment_count: int) -> None:
        spec.validate()
        if environment_count <= 0:
            raise ValueError("standalone environment_count must be positive")
        self.spec = spec
        self.environment_count = environment_count
        self.sim_substeps = int(round(spec.ctrl_dt / spec.sim_dt))
        self.backend = create_backend(
            "mujoco",
            SceneCfg(model_file=spec.model_file),
            environment_count,
            spec.sim_dt,
            base_name="base_link",
            chunk_size=spec.chunk_size,
            adaptive_chunk_size=False,
            bench_nsteps=self.sim_substeps,
        )
        self.backend.materialize()
        if self.backend.num_actuators != self.action_dim:
            raise ValueError("standalone differential drive requires two actuators")
        dtype = get_global_dtype()
        self.home_qpos = np.asarray(self.backend.get_keyframe_qpos("home"), dtype=dtype)
        self.home_qvel = np.asarray(self.backend.get_init_qvel(), dtype=dtype)
        self.robot_states = np.zeros((environment_count, 3), dtype=dtype)
        self.goals = np.zeros((environment_count, 2), dtype=dtype)
        self.velocity_commands = np.zeros((environment_count, 2), dtype=dtype)
        self.wheel_commands = np.zeros((environment_count, 2), dtype=dtype)
        self.previous_distance = np.zeros(environment_count, dtype=dtype)
        self.steps = np.zeros(environment_count, dtype=np.int64)

    def reset_to_initial_conditions(
        self,
        robot_states: np.ndarray,
        goals: np.ndarray,
    ) -> np.ndarray:
        states = np.asarray(robot_states, dtype=get_global_dtype())
        goal_array = np.asarray(goals, dtype=get_global_dtype())
        if states.shape != (self.environment_count, 3):
            raise ValueError("standalone robot states have the wrong shape")
        if goal_array.shape != (self.environment_count, 2):
            raise ValueError("standalone goals have the wrong shape")
        qpos = np.tile(self.home_qpos, (self.environment_count, 1))
        qvel = np.tile(self.home_qvel, (self.environment_count, 1))
        qpos[:, :2] = states[:, :2]
        half_yaw = 0.5 * states[:, 2]
        qpos[:, 3] = np.cos(half_yaw)
        qpos[:, 4:6] = 0.0
        qpos[:, 6] = np.sin(half_yaw)
        indices = np.arange(self.environment_count, dtype=np.int32)
        self.backend.set_state(indices, qpos, qvel)
        self._sync_pose()
        self.goals[:] = goal_array
        self.velocity_commands.fill(0.0)
        self.wheel_commands.fill(0.0)
        self.steps.fill(0)
        distance, _ = self._metrics()
        self.previous_distance[:] = distance
        return self._observation()

    def step(
        self,
        normalized_actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        actions = np.asarray(normalized_actions, dtype=get_global_dtype())
        if actions.shape != (self.environment_count, 2):
            raise ValueError("standalone actions have the wrong shape")
        if not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1.0):
            raise ValueError("standalone actions must be finite and normalized")
        self.velocity_commands[:, 0] = (
            0.5 * (actions[:, 0] + 1.0) * self.spec.max_linear_velocity
        )
        self.velocity_commands[:, 1] = (
            actions[:, 1] * self.spec.max_angular_velocity
        )
        half_track = 0.5 * self.spec.wheel_track
        self.wheel_commands[:, 0] = (
            self.velocity_commands[:, 0]
            - half_track * self.velocity_commands[:, 1]
        ) / self.spec.wheel_radius
        self.wheel_commands[:, 1] = (
            self.velocity_commands[:, 0]
            + half_track * self.velocity_commands[:, 1]
        ) / self.spec.wheel_radius
        np.clip(
            self.wheel_commands,
            -self.spec.max_wheel_speed,
            self.spec.max_wheel_speed,
            out=self.wheel_commands,
        )
        old_distance = self.previous_distance.copy()
        self.backend.step(self.wheel_commands, self.sim_substeps)
        self._sync_pose()
        distance, _ = self._metrics()
        reached = distance <= self.spec.goal_tolerance
        reward = (
            self.spec.progress_scale * (old_distance - distance)
            + self.spec.success_bonus * reached.astype(get_global_dtype())
            - self.spec.time_penalty
        )
        self.previous_distance[:] = distance
        self.steps += 1
        truncated = (self.steps >= self.spec.max_episode_steps) & ~reached
        return self._observation(), reward, reached, truncated

    def _sync_pose(self) -> None:
        position = np.asarray(self.backend.get_base_pos(), dtype=get_global_dtype())
        quaternion = np.asarray(self.backend.get_base_quat(), dtype=get_global_dtype())
        w, x, y, z = np.moveaxis(quaternion, -1, 0)
        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        self.robot_states[:, :2] = position[:, :2]
        self.robot_states[:, 2] = yaw

    def _metrics(self) -> tuple[np.ndarray, np.ndarray]:
        delta = self.goals - self.robot_states[:, :2]
        distance = np.linalg.norm(delta, axis=1)
        goal_yaw = np.arctan2(delta[:, 1], delta[:, 0])
        bearing = (goal_yaw - self.robot_states[:, 2] + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        return distance, bearing

    def _observation(self) -> np.ndarray:
        distance, bearing = self._metrics()
        return np.stack(
            (
                np.clip(distance / self.spec.max_goal_distance, 0.0, 1.0),
                np.sin(bearing),
                np.cos(bearing),
                self.velocity_commands[:, 0] / self.spec.max_linear_velocity,
                self.velocity_commands[:, 1] / self.spec.max_angular_velocity,
            ),
            axis=1,
        ).astype(get_global_dtype(), copy=False)

    def close(self) -> None:
        cleanup = getattr(self.backend, "cleanup_scene_assets", None)
        if callable(cleanup):
            cleanup()
