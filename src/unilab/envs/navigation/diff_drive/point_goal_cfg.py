"""Configuration for differential-drive PointGoal navigation."""

from dataclasses import dataclass

from unilab.base import registry
from unilab.base.base import EnvCfg


@registry.envcfg("DiffDrivePointGoal")
@dataclass
class DiffDrivePointGoalCfg(EnvCfg):
    """Configuration shared by differential-drive PointGoal environments."""

    # Physics simulation advances every 0.01 seconds.
    sim_dt: float = 0.01

    # The reinforcement-learning policy chooses a new action every 0.10 seconds.
    ctrl_dt: float = 0.10

    # One episode lasts at most 20 seconds.
    max_episode_seconds: float = 20.0

    # The robot succeeds inside this distance from the goal.
    goal_tolerance: float = 0.25

    # Initial goals will be sampled inside this distance.
    max_goal_distance: float = 5.0

    # Physical velocity limits used when mapping normalized RL actions.
    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 1.5

    def validate(self) -> None:
        """Reject invalid task configurations before creating the environment."""
        super().validate()

        if self.sim_dt <= 0.0:
            raise ValueError("sim_dt must be positive")

        if self.ctrl_dt <= 0.0:
            raise ValueError("ctrl_dt must be positive")

        if self.max_episode_seconds <= 0.0:
            raise ValueError("max_episode_seconds must be positive")

        if self.goal_tolerance <= 0.0:
            raise ValueError("goal_tolerance must be positive")

        if self.max_goal_distance <= self.goal_tolerance:
            raise ValueError("max_goal_distance must be greater than goal_tolerance")

        if self.max_linear_velocity <= 0.0:
            raise ValueError("max_linear_velocity must be positive")

        if self.max_angular_velocity <= 0.0:
            raise ValueError("max_angular_velocity must be positive")
