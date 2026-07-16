"""MuJoCo-backed differential-drive PointGoal environment."""

from __future__ import annotations

from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.backend import create_backend, env_backend_kwargs
from unilab.base.np_env import NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.utils.rotation import np_yaw_from_quat, np_yaw_to_quat

from .point_goal_cfg import DiffDrivePointGoalCfg
from .point_goal_core import (
    compute_point_goal_metrics,
    compute_point_goal_reward,
    is_goal_reached,
)
from .point_goal_env import DiffDrivePointGoalEnv
from .wheel_control import twist_to_wheel_speeds


@registry.env("DiffDrivePointGoal", sim_backend="mujoco")
class DiffDrivePointGoalMujocoEnv(DiffDrivePointGoalEnv):
    """PointGoal navigation using the real MuJoCo differential-drive model."""

    def __init__(
        self,
        cfg: DiffDrivePointGoalCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
    ) -> None:
        if cfg.scene is None:
            raise ValueError("DiffDrivePointGoalCfg.scene must be configured")

        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name="base_link",
            **env_backend_kwargs(cfg),
        )

        super().__init__(
            cfg=cfg,
            backend=backend,
            num_envs=num_envs,
        )

        # MuJoCo's vectorized pool must exist before reset(), set_state()
        # or step() can use the backend.
        self._backend.materialize()

        if self._backend.num_actuators != self.ACTION_DIM:
            raise ValueError(
                "diff-drive MuJoCo model must have exactly "
                f"{self.ACTION_DIM} actuators, "
                f"got {self._backend.num_actuators}"
            )

        dtype = get_global_dtype()

        self._home_qpos = np.asarray(
            self._backend.get_keyframe_qpos("home"),
            dtype=dtype,
        )

        self._home_qvel = np.asarray(
            self._backend.get_init_qvel(),
            dtype=dtype,
        )

        self.wheel_commands = np.zeros(
            (num_envs, self.ACTION_DIM),
            dtype=dtype,
        )

    def apply_action(
        self,
        actions: np.ndarray,
        state: NpEnvState,
    ) -> np.ndarray:
        """Convert policy actions into MuJoCo wheel-speed targets."""
        velocity_commands = super().apply_action(
            actions,
            state,
        )

        wheel_commands = twist_to_wheel_speeds(
            velocity_commands,
            wheel_radius=self._cfg.wheel_radius,
            wheel_track=self._cfg.wheel_track,
            max_wheel_speed=self._cfg.max_wheel_speed,
        )

        self.wheel_commands[:] = wheel_commands

        return self.wheel_commands

    def update_state(
        self,
        state: NpEnvState,
    ) -> NpEnvState:
        """Read the real MuJoCo pose and compute PointGoal MDP outputs."""
        old_distance = self.previous_distance.copy()

        self._sync_robot_states_from_backend()

        current_distance, _ = compute_point_goal_metrics(
            self.robot_states,
            self.goals,
        )

        reached_goal = is_goal_reached(
            current_distance,
            self._cfg.goal_tolerance,
        )

        reward = compute_point_goal_reward(
            previous_distance=old_distance,
            current_distance=current_distance,
            reached_goal=reached_goal,
            progress_scale=self._cfg.reward_config.progress_scale,
            success_bonus=self._cfg.reward_config.success_bonus,
            time_penalty=self._cfg.reward_config.time_penalty,
        )

        self.previous_distance[:] = current_distance

        observation = self._build_observation()

        self._update_episode_log(
            state=state,
            current_distance=current_distance,
            reached_goal=reached_goal,
        )

        state.info["distance_to_goal"] = current_distance.copy()
        state.info["goal_reached"] = reached_goal.copy()
        state.info["robot_state"] = self.robot_states.copy()
        state.info["goal_position"] = self.goals.copy()
        state.info["wheel_commands"] = self.wheel_commands.copy()
        state.info.update(self._build_task_info())

        return state.replace(
            obs={
                "obs": observation,
                "critic": observation.copy(),
            },
            reward=reward,
            terminated=reached_goal,
            truncated=state.truncated,
        )

    def reset(
        self,
        env_indices: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset selected MuJoCo robots and sample new goals."""
        indices = np.asarray(
            env_indices,
            dtype=np.int32,
        )

        count = len(indices)

        if count == 0:
            empty_observation = np.zeros(
                (0, self.OBSERVATION_DIM),
                dtype=get_global_dtype(),
            )

            return {
                "obs": empty_observation,
                "critic": empty_observation.copy(),
            }, {}

        qpos = np.tile(
            self._home_qpos,
            (count, 1),
        )

        qvel = np.tile(
            self._home_qvel,
            (count, 1),
        )

        initial_yaw = self._rng.uniform(
            -np.pi,
            np.pi,
            size=count,
        )

        # Free-joint qpos layout:
        # [x, y, z, qw, qx, qy, qz, left_wheel, right_wheel]
        qpos[:, 0] = 0.0
        qpos[:, 1] = 0.0
        qpos[:, 3:7] = np_yaw_to_quat(
            initial_yaw
        )

        self._backend.set_state(
            indices,
            qpos,
            qvel,
        )

        self._sync_robot_states_from_backend(
            indices
        )

        self.goals[indices] = self._sample_goal_positions(self.robot_states[indices, :2])

        self.normalized_actions[indices] = 0.0
        self.velocity_commands[indices] = 0.0
        self.wheel_commands[indices] = 0.0

        distance, _ = compute_point_goal_metrics(
            self.robot_states[indices],
            self.goals[indices],
        )

        self.previous_distance[indices] = distance
        self.initial_distance[indices] = distance

        observation = self._build_observation(
            indices
        )

        info = {
            "distance_to_goal": distance.copy(),
            "goal_reached": np.zeros(
                count,
                dtype=bool,
            ),
            "robot_state": self.robot_states[
                indices
            ].copy(),
            "goal_position": self.goals[
                indices
            ].copy(),
            "wheel_commands": self.wheel_commands[
                indices
            ].copy(),
            **self._build_task_info(indices),
        }

        return {
            "obs": observation,
            "critic": observation.copy(),
        }, info

    def _sync_robot_states_from_backend(
        self,
        env_indices: np.ndarray | None = None,
    ) -> None:
        """Copy MuJoCo base position and yaw into planar robot states."""
        base_position = np.asarray(
            self._backend.get_base_pos(),
            dtype=get_global_dtype(),
        )

        base_quaternion = np.asarray(
            self._backend.get_base_quat(),
            dtype=get_global_dtype(),
        )

        base_yaw = np_yaw_from_quat(
            base_quaternion
        )

        if env_indices is None:
            self.robot_states[:, 0:2] = (
                base_position[:, 0:2]
            )
            self.robot_states[:, 2] = base_yaw
            return

        self.robot_states[env_indices, 0:2] = (
            base_position[env_indices, 0:2]
        )

        self.robot_states[env_indices, 2] = (
            base_yaw[env_indices]
        )

    def _set_initial_conditions(
        self,
        robot_states: np.ndarray,
        goals: np.ndarray,
    ) -> None:
        """Install explicit evaluator starts in the real MuJoCo backend."""
        qpos = np.tile(self._home_qpos, (self.num_envs, 1))
        qvel = np.tile(self._home_qvel, (self.num_envs, 1))
        qpos[:, 0:2] = robot_states[:, 0:2]
        qpos[:, 3:7] = np_yaw_to_quat(robot_states[:, 2])
        indices = np.arange(self.num_envs, dtype=np.int32)
        self._backend.set_state(indices, qpos, qvel)
        self._sync_robot_states_from_backend()
        self.goals[:] = goals
        self.wheel_commands.fill(0.0)
