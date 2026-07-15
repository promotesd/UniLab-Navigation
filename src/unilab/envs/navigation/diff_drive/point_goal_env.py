"""UniLab NpEnv contract for kinematic PointGoal navigation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.dtype_config import get_global_dtype

from .kinematics import differential_drive_step
from .point_goal_cfg import DiffDrivePointGoalCfg
from .point_goal_core import (
    build_point_goal_observation,
    compute_point_goal_metrics,
    compute_point_goal_reward,
    is_goal_reached,
)


class DiffDrivePointGoalEnv(NpEnv):
    """Vectorized PointGoal navigation environment contract.

    This class currently uses an injected backend so its UniLab environment
    behavior can be tested independently from the future MuJoCo robot model.
    """

    OBSERVATION_DIM = 5
    ACTION_DIM = 2

    def __init__(
        self,
        cfg: DiffDrivePointGoalCfg,
        backend: SimBackend,
        num_envs: int = 1,
    ) -> None:
        cfg.validate()
        super().__init__(cfg, backend, num_envs)

        self._cfg: DiffDrivePointGoalCfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        dtype = get_global_dtype()

        # [x, y, heading]
        self.robot_states = np.zeros(
            (num_envs, 3),
            dtype=dtype,
        )

        # [goal_x, goal_y]
        self.goals = np.zeros(
            (num_envs, 2),
            dtype=dtype,
        )

        # Normalized actions produced by the policy.
        self.normalized_actions = np.zeros(
            (num_envs, self.ACTION_DIM),
            dtype=dtype,
        )

        # Physical commands [linear_velocity, angular_velocity].
        self.velocity_commands = np.zeros(
            (num_envs, self.ACTION_DIM),
            dtype=dtype,
        )

        self.previous_distance = np.zeros(
            (num_envs,),
            dtype=dtype,
        )

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """Dimensions trusted by the PPO wrapper and neural networks."""
        return {
            "obs": self.OBSERVATION_DIM,
            "critic": self.OBSERVATION_DIM,
        }

    @property
    def action_space(self) -> gym.Space:
        """Normalized policy action space."""
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.ACTION_DIM,),
            dtype=np.float32,
        )

    def apply_action(
        self,
        actions: np.ndarray,
        state: NpEnvState,
    ) -> np.ndarray:
        """Convert normalized policy actions into physical velocity commands."""
        del state

        action_array = np.asarray(
            actions,
            dtype=get_global_dtype(),
        )

        expected_shape = (self.num_envs, self.ACTION_DIM)

        if action_array.shape != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, "
                f"got {action_array.shape}"
            )

        if not np.all(np.isfinite(action_array)):
            raise ValueError("actions must contain only finite values")

        np.clip(
            action_array,
            -1.0,
            1.0,
            out=self.normalized_actions,
        )

        # Linear action:
        #   -1 -> 0 m/s
        #   +1 -> max_linear_velocity
        self.velocity_commands[:, 0] = (
            0.5
            * (self.normalized_actions[:, 0] + 1.0)
            * self._cfg.max_linear_velocity
        )

        # Angular action:
        #   -1 -> -max_angular_velocity
        #   +1 -> +max_angular_velocity
        self.velocity_commands[:, 1] = (
            self.normalized_actions[:, 1]
            * self._cfg.max_angular_velocity
        )

        return self.velocity_commands

    def update_state(
        self,
        state: NpEnvState,
    ) -> NpEnvState:
        """Advance navigation state and compute the MDP outputs."""
        old_distance = self.previous_distance.copy()

        self.robot_states[:] = differential_drive_step(
            self.robot_states,
            self.velocity_commands,
            dt=self._cfg.ctrl_dt,
        )

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
            progress_scale=self._cfg.progress_reward_scale,
            success_bonus=self._cfg.success_bonus,
            time_penalty=self._cfg.time_penalty,
        )

        self.previous_distance[:] = current_distance

        observation = self._build_observation()

        state.info["distance_to_goal"] = current_distance.copy()
        state.info["goal_reached"] = reached_goal.copy()
        state.info["robot_state"] = self.robot_states.copy()
        state.info["goal_position"] = self.goals.copy()

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
        """Reset selected robots and sample new PointGoal targets."""
        indices = np.asarray(
            env_indices,
            dtype=np.int32,
        )

        count = len(indices)

        if count == 0:
            empty_obs = np.zeros(
                (0, self.OBSERVATION_DIM),
                dtype=get_global_dtype(),
            )
            return {
                "obs": empty_obs,
                "critic": empty_obs.copy(),
            }, {}

        # Version 0 always starts each robot at the world origin.
        self.robot_states[indices, 0:2] = 0.0

        # Random initial heading.
        self.robot_states[indices, 2] = self._rng.uniform(
            -np.pi,
            np.pi,
            size=count,
        )

        goal_distance = self._rng.uniform(
            self._cfg.min_goal_distance,
            self._cfg.max_goal_distance,
            size=count,
        )

        goal_angle = self._rng.uniform(
            -np.pi,
            np.pi,
            size=count,
        )

        self.goals[indices, 0] = (
            goal_distance * np.cos(goal_angle)
        )
        self.goals[indices, 1] = (
            goal_distance * np.sin(goal_angle)
        )

        self.normalized_actions[indices] = 0.0
        self.velocity_commands[indices] = 0.0

        distance, _ = compute_point_goal_metrics(
            self.robot_states[indices],
            self.goals[indices],
        )

        self.previous_distance[indices] = distance

        observation = self._build_observation(indices)

        info = {
            "distance_to_goal": distance.copy(),
            "goal_reached": np.zeros(count, dtype=bool),
            "robot_state": self.robot_states[indices].copy(),
            "goal_position": self.goals[indices].copy(),
        }

        return {
            "obs": observation,
            "critic": observation.copy(),
        }, info

    def _build_observation(
        self,
        env_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the five-dimensional bounded policy observation."""
        if env_indices is None:
            states = self.robot_states
            goals = self.goals
            commands = self.velocity_commands
        else:
            states = self.robot_states[env_indices]
            goals = self.goals[env_indices]
            commands = self.velocity_commands[env_indices]

        point_goal_obs = build_point_goal_observation(
            states,
            goals,
            max_goal_distance=self._cfg.max_goal_distance,
        )

        normalized_linear_velocity = (
            commands[:, 0] / self._cfg.max_linear_velocity
        )

        normalized_angular_velocity = (
            commands[:, 1] / self._cfg.max_angular_velocity
        )

        observation = np.concatenate(
            [
                point_goal_obs,
                normalized_linear_velocity[:, None],
                normalized_angular_velocity[:, None],
            ],
            axis=1,
        )

        return observation.astype(
            get_global_dtype(),
            copy=False,
        )
