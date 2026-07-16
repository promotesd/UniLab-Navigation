"""UniLab NpEnv contract for kinematic PointGoal navigation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from unilab.base.backend import SimBackend
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.dtype_config import get_global_dtype
from unilab.envs.navigation.localization import (
    GroundTruthPosePacket,
    LocalizationPacket,
    PoseProvider,
    WheelOdometryPacket,
    create_pose_provider,
)

from .episode_metrics import build_point_goal_episode_log
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
        pose_provider: PoseProvider | None = None,
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
        self.localization_time_s = np.zeros(num_envs, dtype=np.float64)
        self.pose_provider = pose_provider or create_pose_provider(cfg.localization)
        all_indices = np.arange(num_envs, dtype=np.int32)
        self.pose_estimate = self.pose_provider.reset(
            LocalizationPacket(
                ground_truth=GroundTruthPosePacket(
                    timestamp_s=self.localization_time_s,
                    pose=self.robot_states,
                    child_frame=cfg.localization.child_frame,
                ),
                wheel_odometry=WheelOdometryPacket(
                    timestamp_s=self.localization_time_s,
                    dt_s=np.zeros(num_envs),
                    linear_velocity=np.zeros(num_envs),
                    angular_velocity=np.zeros(num_envs),
                    frame_id=cfg.localization.child_frame,
                ),
            ),
            all_indices,
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

        # Distance sampled at the beginning of each episode.
        # This remains unchanged until that environment is reset.
        self.initial_distance = np.zeros(
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
        self.localization_time_s += self._cfg.ctrl_dt
        self.pose_estimate = self.pose_provider.update(
            self._build_localization_packet(self._cfg.ctrl_dt),
        )

        current_distance, _ = compute_point_goal_metrics(
            self.robot_states,
            self.goals,
        )

        reached_goal = is_goal_reached(
            current_distance,
            self._cfg.goal_tolerance,
        )
        collision = self._compute_collision_mask() & ~reached_goal

        reward = compute_point_goal_reward(
            previous_distance=old_distance,
            current_distance=current_distance,
            reached_goal=reached_goal,
            progress_scale=self._cfg.reward_config.progress_scale,
            success_bonus=self._cfg.reward_config.success_bonus,
            time_penalty=self._cfg.reward_config.time_penalty,
        )
        reward -= self._collision_penalty(collision)

        self.previous_distance[:] = current_distance

        observation = self._build_observation()

        self._update_episode_log(
            state=state,
            current_distance=current_distance,
            reached_goal=reached_goal,
            collision=collision,
        )

        state.info["distance_to_goal"] = current_distance.copy()
        state.info["goal_reached"] = reached_goal.copy()
        state.info["collision"] = collision.copy()
        state.info["robot_state"] = self.robot_states.copy()
        state.info["goal_position"] = self.goals.copy()
        state.info.update(self._build_task_info())
        state.info.update(self._build_localization_info())

        return state.replace(
            obs={
                "obs": observation,
                "critic": observation.copy(),
            },
            reward=reward,
            terminated=reached_goal | collision,
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

        self.goals[indices] = self._sample_goal_positions(
            self.robot_states[indices, :2],
            env_indices=indices,
        )
        self.localization_time_s[indices] = 0.0
        self.pose_estimate = self.pose_provider.reset(
            self._build_localization_packet(0.0),
            indices,
        )

        self.normalized_actions[indices] = 0.0
        self.velocity_commands[indices] = 0.0

        distance, _ = compute_point_goal_metrics(
            self.robot_states[indices],
            self.goals[indices],
        )

        self.previous_distance[indices] = distance
        self.initial_distance[indices] = distance

        observation = self._build_observation(indices)

        info = {
            "distance_to_goal": distance.copy(),
            "goal_reached": np.zeros(count, dtype=bool),
            "collision": np.zeros(count, dtype=bool),
            "robot_state": self.robot_states[indices].copy(),
            "goal_position": self.goals[indices].copy(),
            **self._build_task_info(indices),
            **self._build_localization_info(indices),
        }

        return {
            "obs": observation,
            "critic": observation.copy(),
        }, info

    def reset_to_initial_conditions(
        self,
        robot_states: np.ndarray,
        goals: np.ndarray,
    ) -> NpEnvState:
        """Reset the full vectorized environment to explicit episode starts.

        This evaluator-facing reset bypasses random sampling so multiple
        policies can run exactly the same initial-condition manifest.
        """
        states = np.asarray(robot_states, dtype=get_global_dtype())
        goal_array = np.asarray(goals, dtype=get_global_dtype())
        expected_state_shape = (self.num_envs, 3)
        expected_goal_shape = (self.num_envs, 2)

        if states.shape != expected_state_shape:
            raise ValueError(
                f"robot_states must have shape {expected_state_shape}, got {states.shape}"
            )
        if goal_array.shape != expected_goal_shape:
            raise ValueError(f"goals must have shape {expected_goal_shape}, got {goal_array.shape}")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(goal_array)):
            raise ValueError("initial conditions must contain only finite values")

        distances, _ = compute_point_goal_metrics(states, goal_array)
        if np.any(distances < self._cfg.min_goal_distance) or np.any(
            distances > self._cfg.max_goal_distance
        ):
            raise ValueError(
                "initial goal distances must be inside the configured "
                "[min_goal_distance, max_goal_distance] range"
            )
        self._validate_initial_conditions(states, goal_array)

        if self._state is None:
            self.init_state()
        assert self._state is not None

        self._set_initial_conditions(states, goal_array)
        self.localization_time_s.fill(0.0)
        all_indices = np.arange(self.num_envs, dtype=np.int32)
        self.pose_estimate = self.pose_provider.reset(
            self._build_localization_packet(0.0),
            all_indices,
        )
        distances, _ = compute_point_goal_metrics(self.robot_states, self.goals)
        self.normalized_actions.fill(0.0)
        self.velocity_commands.fill(0.0)
        self.previous_distance[:] = distances
        self.initial_distance[:] = distances

        observation = self._build_observation()
        info: dict[str, Any] = {
            "steps": np.zeros(self.num_envs, dtype=np.uint32),
            "distance_to_goal": distances.copy(),
            "goal_reached": np.zeros(self.num_envs, dtype=bool),
            "collision": np.zeros(self.num_envs, dtype=bool),
            "robot_state": self.robot_states.copy(),
            "goal_position": self.goals.copy(),
            **self._build_task_info(),
            **self._build_localization_info(),
        }
        self._state = self._state.replace(
            obs={"obs": observation, "critic": observation.copy()},
            reward=np.zeros(self.num_envs, dtype=get_global_dtype()),
            terminated=np.zeros(self.num_envs, dtype=bool),
            truncated=np.zeros(self.num_envs, dtype=bool),
            info=info,
            final_observation=None,
        )
        return self._state

    def _set_initial_conditions(
        self,
        robot_states: np.ndarray,
        goals: np.ndarray,
    ) -> None:
        """Install validated conditions in the backend-independent state."""
        self.robot_states[:] = robot_states
        self.goals[:] = goals

    def _validate_initial_conditions(
        self,
        robot_states: np.ndarray,
        goals: np.ndarray,
    ) -> None:
        """Validate variant-specific explicit evaluator starts."""
        del robot_states, goals

    def _sample_goal_positions(
        self,
        origins: np.ndarray,
        *,
        env_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample goals relative to planar origins using the task distribution."""
        del env_indices
        distances = self._rng.uniform(
            self._cfg.min_goal_distance,
            self._cfg.max_goal_distance,
            size=len(origins),
        )
        angles = self._rng.uniform(-np.pi, np.pi, size=len(origins))
        return origins + np.stack(
            [distances * np.cos(angles), distances * np.sin(angles)], axis=1
        )

    def _build_task_info(self, env_indices: np.ndarray | None = None) -> dict[str, Any]:
        """Return variant-specific backend-independent task state."""
        del env_indices
        return {}

    def _build_localization_info(
        self,
        env_indices: np.ndarray | None = None,
    ) -> dict[str, Any]:
        estimate = self.pose_estimate
        indices = slice(None) if env_indices is None else env_indices
        count = self.num_envs if env_indices is None else len(env_indices)
        return {
            "localization_pose": estimate.pose[indices].copy(),
            "localization_covariance": estimate.covariance[indices].copy(),
            "localization_valid": estimate.valid[indices].copy(),
            "localization_status": estimate.status[indices].copy(),
            "localization_timestamp_s": estimate.timestamp_s[indices].copy(),
            "localization_parent_frame": np.full(count, estimate.parent_frame),
            "localization_child_frame": np.full(count, estimate.child_frame),
        }

    def _build_localization_packet(self, dt_s: float) -> LocalizationPacket:
        return LocalizationPacket(
            ground_truth=GroundTruthPosePacket(
                timestamp_s=self.localization_time_s,
                pose=self.robot_states,
                child_frame=self._cfg.localization.child_frame,
            ),
            wheel_odometry=WheelOdometryPacket(
                timestamp_s=self.localization_time_s,
                dt_s=np.full(self.num_envs, dt_s, dtype=np.float64),
                linear_velocity=self.velocity_commands[:, 0],
                angular_velocity=self.velocity_commands[:, 1],
                frame_id=self._cfg.localization.child_frame,
            ),
        )

    def _compute_collision_mask(self) -> np.ndarray:
        """Return task collision terminals; obstacle-free tasks have none."""
        return np.zeros(self.num_envs, dtype=bool)

    def close(self) -> None:
        """Release localization-plugin and backend-owned resources."""
        close_provider = getattr(self.pose_provider, "close", None)
        if callable(close_provider):
            close_provider()
        super().close()

    def _collision_penalty(self, collision: np.ndarray) -> np.ndarray:
        """Return a per-environment collision reward penalty."""
        del collision
        return np.zeros(self.num_envs, dtype=get_global_dtype())

    def _update_episode_log(
        self,
        *,
        state: NpEnvState,
        current_distance: np.ndarray,
        reached_goal: np.ndarray,
        collision: np.ndarray | None = None,
    ) -> None:
        """Publish metrics for episodes completed on this step."""
        max_episode_steps = self._cfg.max_episode_steps

        if max_episode_steps is None:
            state.info.pop("log", None)
            return

        # NpEnv increments state.info["steps"] after update_state().
        # Therefore +1 represents the action that has just completed.
        episode_steps = np.asarray(
            state.info["steps"],
        ) + 1

        episode_log = build_point_goal_episode_log(
            initial_distance=self.initial_distance,
            final_distance=current_distance,
            reached_goal=reached_goal,
            collision=collision,
            episode_steps=episode_steps,
            max_episode_steps=int(max_episode_steps),
        )

        if episode_log is None:
            # Prevent metrics from a previous terminal step from being
            # emitted again on the following environment step.
            state.info.pop("log", None)
        else:
            state.info["log"] = episode_log

    def _build_observation(
        self,
        env_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the five-dimensional bounded policy observation."""
        if env_indices is None:
            states = self.pose_estimate.pose
            goals = self.goals
            commands = self.velocity_commands
        else:
            states = self.pose_estimate.pose[env_indices]
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
