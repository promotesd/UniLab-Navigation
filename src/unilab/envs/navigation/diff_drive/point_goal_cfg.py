"""Configuration for differential-drive PointGoal navigation."""

from dataclasses import dataclass

from unilab.base import registry
from unilab.base.base import EnvCfg


@registry.envcfg("DiffDrivePointGoal")
@dataclass
class DiffDrivePointGoalCfg(EnvCfg):
    """Configuration shared by differential-drive PointGoal environments."""

    # Reproducible random reset sampling.
    seed: int = 1

    # Physics and policy timing.
    sim_dt: float = 0.01
    ctrl_dt: float = 0.10
    max_episode_seconds: float = 20.0

    # Goal sampling and success condition.
    goal_tolerance: float = 0.25
    min_goal_distance: float = 1.0
    max_goal_distance: float = 5.0

    # Physical command limits.
    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 1.5

    # Reward parameters.
    progress_reward_scale: float = 2.0
    success_bonus: float = 10.0
    time_penalty: float = 0.01

    def validate(self) -> None:
        """Reject invalid task configurations before environment creation."""
        super().validate()

        if self.seed < 0:
            raise ValueError("seed must be non-negative")

        if self.sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive")

        if self.ctrl_dt <= 0.0:
            raise ValueError("ctrl_dt must be positive")

        if self.max_episode_seconds <= 0.0:
            raise ValueError("max_episode_seconds must be positive")

        if self.goal_tolerance <= 0.0:
            raise ValueError("goal_tolerance must be positive")

        if self.min_goal_distance <= self.goal_tolerance:
            raise ValueError(
                "min_goal_distance must be greater than goal_tolerance"
            )

        if self.max_goal_distance <= self.min_goal_distance:
            raise ValueError(
                "max_goal_distance must be greater than min_goal_distance"
            )

        if self.max_linear_velocity <= 0.0:
            raise ValueError("max_linear_velocity must be positive")

        if self.max_angular_velocity <= 0.0:
            raise ValueError("max_angular_velocity must be positive")

        if self.progress_reward_scale < 0.0:
            raise ValueError("progress_reward_scale must be non-negative")

        if self.success_bonus < 0.0:
            raise ValueError("success_bonus must be non-negative")

        if self.time_penalty < 0.0:
            raise ValueError("time_penalty must be non-negative")
