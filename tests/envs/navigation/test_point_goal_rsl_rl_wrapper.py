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


def test_wrapper_converts_episode_log_to_torch() -> None:
    """Episode metrics must reach RSL-RL as Torch tensors."""
    wrapped_env = make_wrapped_env(
        num_envs=4,
    )

    wrapped_env.reset()

    base_env = wrapped_env.env

    # Force every environment to reach its goal on this step.
    base_env.goals[:] = base_env.robot_states[:, 0:2]
    base_env.initial_distance[:] = 1.0
    base_env.previous_distance[:] = 1.0

    # Linear action -1 maps to zero forward speed.
    actions = torch.tensor(
        [
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    _, _, dones, info = wrapped_env.step(actions)

    assert torch.all(dones)
    assert "log" in info

    log = info["log"]

    assert isinstance(log, dict)

    success_rate = log[
        "Navigation/success_rate"
    ]

    final_distance = log[
        "Navigation/final_distance"
    ]

    assert isinstance(
        success_rate,
        torch.Tensor,
    )

    assert isinstance(
        final_distance,
        torch.Tensor,
    )

    assert success_rate.shape == (4,)
    assert final_distance.shape == (4,)

    assert success_rate.dtype == torch.float32
    assert final_distance.dtype == torch.float32

    assert torch.allclose(
        success_rate,
        torch.ones(4),
    )

    # A real MuJoCo step can introduce a very small displacement
    # through contact solving and numerical integration. Navigation
    # success therefore means being inside the configured tolerance,
    # not being exactly zero metres from the goal.
    assert torch.all(
        final_distance
        <= base_env.cfg.goal_tolerance
    )

    assert torch.all(
        torch.isfinite(final_distance)
    )

