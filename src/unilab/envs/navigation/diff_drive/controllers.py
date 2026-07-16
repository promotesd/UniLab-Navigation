"""Baseline controllers for differential-drive PointGoal evaluation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .lidar import PlanarLidarCfg

STATIONARY_ACTION = np.array([-1.0, 0.0], dtype=np.float32)


class PointGoalPolicy(Protocol):
    """Common NumPy policy contract used by the independent evaluator."""

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        """Return normalized actions with shape ``(num_envs, 2)``."""


class ZeroPointGoalPolicy:
    """Stationary lower-bound policy.

    PointGoal's first normalized action maps ``-1`` to zero linear velocity,
    so the physical zero command is ``[-1, 0]`` rather than ``[0, 0]``.
    """

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        num_envs = len(observations["obs"])
        return np.broadcast_to(STATIONARY_ACTION, (num_envs, 2)).copy()


class RandomPointGoalPolicy:
    """Seeded uniform-random normalized-action policy."""

    def __init__(self, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._rng = np.random.default_rng(seed)

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        num_envs = len(observations["obs"])
        return self._rng.uniform(-1.0, 1.0, size=(num_envs, 2)).astype(np.float32)


class HeuristicPointGoalPolicy:
    """Proportional PointGoal controller using the egocentric goal bearing."""

    def __init__(self, angular_gain: float = 1.5) -> None:
        if angular_gain <= 0.0:
            raise ValueError("angular_gain must be positive")
        self.angular_gain = float(angular_gain)

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        actor_obs = np.asarray(observations["obs"])
        if actor_obs.ndim != 2 or actor_obs.shape[1] < 3:
            raise ValueError("PointGoal observations must have shape (num_envs, at least 3)")

        bearing = np.arctan2(actor_obs[:, 1], actor_obs[:, 2])
        forward_fraction = np.clip(np.cos(bearing), 0.0, 1.0)
        linear_action = 2.0 * forward_fraction - 1.0
        angular_action = np.clip(self.angular_gain * bearing, -1.0, 1.0)
        return np.stack([linear_action, angular_action], axis=1).astype(np.float32)


class LidarHeuristicPointGoalPolicy:
    """Reactive PointGoal controller that turns around frontal LiDAR hazards."""

    def __init__(
        self,
        lidar_cfg: PlanarLidarCfg,
        *,
        angular_gain: float = 1.5,
        avoid_distance: float = 0.6,
        release_distance: float = 0.8,
        minimum_avoid_steps: int = 1,
    ) -> None:
        lidar_cfg.validate()
        if angular_gain <= 0.0:
            raise ValueError("angular_gain must be positive")
        if avoid_distance <= lidar_cfg.min_range:
            raise ValueError("avoid_distance must exceed the LiDAR minimum range")
        if release_distance <= avoid_distance:
            raise ValueError("release_distance must exceed avoid_distance")
        if minimum_avoid_steps <= 0:
            raise ValueError("minimum_avoid_steps must be positive")
        self.lidar_cfg = lidar_cfg
        self.angular_gain = float(angular_gain)
        self.avoid_distance = float(avoid_distance)
        self.release_distance = float(release_distance)
        self.minimum_avoid_steps = int(minimum_avoid_steps)
        self._beam_angles = np.linspace(
            lidar_cfg.angle_min,
            lidar_cfg.angle_max,
            lidar_cfg.beam_count,
            endpoint=False,
        )
        self._turn_direction = np.empty(0, dtype=np.float32)
        self._avoid_steps = np.empty(0, dtype=np.int32)

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        actor_obs = np.asarray(observations["obs"])
        expected_width = 5 + self.lidar_cfg.beam_count
        if actor_obs.ndim != 2 or actor_obs.shape[1] != expected_width:
            raise ValueError(
                f"LiDAR PointGoal observations must have shape "
                f"(num_envs, {expected_width})"
            )
        count = len(actor_obs)
        if self._turn_direction.shape != (count,):
            self._turn_direction = np.zeros(count, dtype=np.float32)
            self._avoid_steps = np.zeros(count, dtype=np.int32)

        bearing = np.arctan2(actor_obs[:, 1], actor_obs[:, 2])
        normalized_ranges = np.clip(actor_obs[:, 5:], 0.0, 1.0)
        ranges = self.lidar_cfg.min_range + normalized_ranges * (
            self.lidar_cfg.max_range - self.lidar_cfg.min_range
        )
        front = np.abs(self._beam_angles) <= np.pi / 4.0
        front_distance = np.min(ranges[:, front], axis=1)
        newly_blocked = (front_distance < self.avoid_distance) & (
            self._turn_direction == 0.0
        )
        if np.any(newly_blocked):
            left_clearance = np.mean(ranges[:, self._beam_angles > 0.0], axis=1)
            right_clearance = np.mean(ranges[:, self._beam_angles < 0.0], axis=1)
            preferred_turn = np.where(left_clearance >= right_clearance, 1.0, -1.0)
            self._turn_direction[newly_blocked] = preferred_turn[newly_blocked]
            self._avoid_steps[newly_blocked] = 0
        avoiding = self._turn_direction != 0.0
        self._avoid_steps[avoiding] += 1
        released = (
            (front_distance >= self.release_distance)
            & avoiding
            & (self._avoid_steps >= self.minimum_avoid_steps)
        )
        self._turn_direction[released] = 0.0
        self._avoid_steps[released] = 0

        avoiding = self._turn_direction != 0.0
        forward_fraction = np.clip(np.cos(bearing), 0.0, 1.0)
        linear_action = 2.0 * forward_fraction - 1.0
        linear_action[avoiding] = 0.0
        angular_action = np.clip(self.angular_gain * bearing, -1.0, 1.0)
        angular_action[avoiding] = self._turn_direction[avoiding]
        return np.stack((linear_action, angular_action), axis=1).astype(np.float32)
