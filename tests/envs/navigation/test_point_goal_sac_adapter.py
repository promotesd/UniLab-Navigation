"""Tests for the PointGoal FastSAC adapter."""

from pathlib import Path

import numpy as np
import pytest
import torch

from unilab.algos.torch.fast_sac.learner import SACActor
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
)
from unilab.evaluation.point_goal_sac import (
    build_point_goal_sac_policy_factory,
    load_point_goal_sac_config,
)

ROOT_DIR = Path(__file__).parents[3]


def test_formal_sac_config_uses_same_task_and_seed_contract() -> None:
    cfg = load_point_goal_sac_config(ROOT_DIR)
    assert cfg.algo.algo == "sac"
    assert cfg.training.task_name == "DiffDrivePointGoal"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.seed == cfg.algo.seed
    assert cfg.algo.save_interval == 25
    assert cfg.algo.use_symmetry is False


def test_real_mujoco_sac_checkpoint_loads_deterministic_policy(tmp_path) -> None:
    cfg = load_point_goal_sac_config(ROOT_DIR)
    env = DiffDrivePointGoalMujocoEnv(cfg=DiffDrivePointGoalCfg(), num_envs=2)
    source = SACActor(
        obs_dim=5,
        action_dim=2,
        hidden_dim=int(cfg.algo.actor_hidden_dim),
        use_layer_norm=bool(cfg.algo.use_layer_norm),
        device="cpu",
    )
    checkpoint = tmp_path / "model_0.pt"
    torch.save({"actor": source.state_dict(), "obs_normalizer": None}, checkpoint)
    policy = build_point_goal_sac_policy_factory(
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
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)
    env.close()


def test_sac_checkpoint_requires_actor_key(tmp_path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    torch.save({"qnet": {}}, checkpoint)
    with pytest.raises(ValueError, match="must contain actor"):
        build_point_goal_sac_policy_factory(
            load_point_goal_sac_config(ROOT_DIR),
            checkpoint=checkpoint,
            device="cpu",
        )
