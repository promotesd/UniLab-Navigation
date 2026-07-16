"""Tests for the PointGoal FastTD3 adapter."""

from pathlib import Path

import numpy as np
import pytest
import torch

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.fast_td3.learner import TD3Actor
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
)
from unilab.evaluation.point_goal_td3 import (
    build_point_goal_td3_policy_factory,
    load_point_goal_td3_config,
)

ROOT_DIR = Path(__file__).parents[3]


def test_formal_td3_config_uses_same_task_and_seed_contract() -> None:
    cfg = load_point_goal_td3_config(ROOT_DIR)
    assert cfg.algo.algo == "td3"
    assert cfg.training.task_name == "DiffDrivePointGoal"
    assert cfg.env.seed == cfg.algo.seed
    assert cfg.algo.save_interval == 25
    assert cfg.algo.obs_normalization is True


def test_real_mujoco_td3_checkpoint_loads_deterministic_policy(tmp_path) -> None:
    cfg = load_point_goal_td3_config(ROOT_DIR)
    env = DiffDrivePointGoalMujocoEnv(cfg=DiffDrivePointGoalCfg(), num_envs=2)
    source = TD3Actor(
        obs_dim=5,
        n_act=2,
        num_envs=7,
        init_scale=float(cfg.algo.algo_params.init_scale),
        hidden_dim=int(cfg.algo.actor_hidden_dim),
        log_std_min=float(cfg.algo.algo_params.log_std_min),
        log_std_max=float(cfg.algo.algo_params.log_std_max),
        device=torch.device("cpu"),
    )
    normalizer = EmpiricalNormalization(shape=5, device="cpu")
    checkpoint = tmp_path / "model_0.pt"
    torch.save(
        {
            "actor": source.state_dict(),
            "obs_normalizer": normalizer.state_dict(),
        },
        checkpoint,
    )
    policy = build_point_goal_td3_policy_factory(
        cfg,
        checkpoint=checkpoint,
        device="cpu",
    )(env)
    observations = env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi]], dtype=np.float32),
        np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
    ).obs
    first = policy(observations)
    second = policy(observations)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 2)
    env.close()


def test_td3_checkpoint_requires_actor_key(tmp_path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"qnet": {}}, checkpoint)
    with pytest.raises(ValueError, match="must contain actor"):
        build_point_goal_td3_policy_factory(
            load_point_goal_td3_config(ROOT_DIR),
            checkpoint=checkpoint,
            device="cpu",
        )
