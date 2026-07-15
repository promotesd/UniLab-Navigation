"""RSL-RL wrapper integration tests for PointGoal navigation."""

import numpy as np
import torch

from unilab.base import registry
from unilab.training.rsl_rl import RslRlVecEnvWrapper


def make_wrapped_env(
    *,
    num_envs: int = 4,
) -> RslRlVecEnvWrapper:
    """Create PointGoal through the same path used by PPO training."""
    registry.ensure_registries()

    env = registry.make(
        "DiffDrivePointGoal",
        sim_backend="mujoco",
        num_envs=num_envs,
    )

    return RslRlVecEnvWrapper(
        env,
        device="cpu",
    )


def test_wrapper_exposes_ppo_dimensions() -> None:
    """Verify the dimensions used to construct actor and critic networks."""
    wrapped_env = make_wrapped_env(
        num_envs=4,
    )

    assert wrapped_env.num_envs == 4

    assert wrapped_env.num_obs == 5
    assert wrapped_env.num_privileged_obs == 5
    assert wrapped_env.num_actions == 2

    # 20 seconds / 0.1 second control interval = 200 steps.
    assert wrapped_env.max_episode_length == 200


def test_wrapper_reset_returns_rsl_rl_tensordict() -> None:
    """Verify reset output matches the RSL-RL observation contract."""
    wrapped_env = make_wrapped_env(
        num_envs=4,
    )

    observations, info = wrapped_env.reset()

    assert observations.batch_size == torch.Size([4])

    assert observations["actor"].shape == (4, 5)
    assert observations["policy"].shape == (4, 5)
    assert observations["critic"].shape == (4, 5)

    assert observations["actor"].device.type == "cpu"
    assert observations["actor"].dtype == torch.float32

    assert isinstance(info, dict)

    assert torch.all(
        torch.isfinite(observations["actor"])
    )


def test_wrapper_step_accepts_torch_policy_actions() -> None:
    """Run the complete Torch action to MuJoCo state transition."""
    wrapped_env = make_wrapped_env(
        num_envs=4,
    )

    wrapped_env.reset()

    actions = torch.zeros(
        (4, 2),
        dtype=torch.float32,
    )

    observations, rewards, dones, info = (
        wrapped_env.step(actions)
    )

    assert observations["actor"].shape == (4, 5)
    assert observations["critic"].shape == (4, 5)

    assert rewards.shape == (4,)
    assert dones.shape == (4,)

    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.bool

    assert torch.all(torch.isfinite(rewards))
    assert torch.all(
        torch.isfinite(observations["actor"])
    )

    assert isinstance(info, dict)


def test_wrapper_runs_multiple_physics_steps() -> None:
    """Ensure repeated PPO-style steps keep all outputs finite."""
    wrapped_env = make_wrapped_env(
        num_envs=8,
    )

    observations, _ = wrapped_env.reset()

    for _ in range(20):
        actions = torch.empty(
            (8, 2),
            dtype=torch.float32,
        ).uniform_(-1.0, 1.0)

        observations, rewards, dones, _ = (
            wrapped_env.step(actions)
        )

        assert observations["actor"].shape == (8, 5)
        assert rewards.shape == (8,)
        assert dones.shape == (8,)

        assert torch.all(
            torch.isfinite(observations["actor"])
        )

        assert torch.all(
            torch.isfinite(rewards)
        )

    robot_states = wrapped_env.env.robot_states

    assert robot_states.shape == (8, 3)
    assert np.all(np.isfinite(robot_states))
