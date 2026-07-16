"""Baseline controllers for differential-drive PointGoal evaluation."""

from __future__ import annotations

from typing import Protocol

import numpy as np

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
