"""Real MuJoCo integration tests for fixed-episode PointGoal evaluation."""

import numpy as np
import pytest
import torch
from rsl_rl.runners import OnPolicyRunner
from scripts import evaluate_point_goal as evaluate_script

from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
    HeuristicPointGoalPolicy,
    ZeroPointGoalPolicy,
)
from unilab.evaluation.point_goal import (
    PointGoalManifest,
    evaluate_point_goal_policies,
)
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg

pytest.importorskip("mujoco")


def make_env() -> DiffDrivePointGoalMujocoEnv:
    cfg = DiffDrivePointGoalCfg(
        seed=11,
        min_goal_distance=1.0,
        max_goal_distance=2.0,
        max_episode_seconds=4.0,
    )
    return DiffDrivePointGoalMujocoEnv(cfg=cfg, num_envs=2, backend_type="mujoco")


def test_real_mujoco_reset_applies_manifest_pose_and_goal() -> None:
    env = make_env()
    states = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float32)
    goals = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    state = env.reset_to_initial_conditions(states, goals)
    np.testing.assert_allclose(env.robot_states, states, atol=1.0e-6)
    np.testing.assert_array_equal(env.goals, goals)
    np.testing.assert_allclose(state.info["distance_to_goal"], np.ones(2), atol=1.0e-6)
    env.close()


def test_real_mujoco_fixed_evaluator_compares_same_episodes() -> None:
    manifest = PointGoalManifest(
        seed=12,
        robot_states=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float32),
        goals=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    report = evaluate_point_goal_policies(
        make_env,
        {
            "zero": lambda env: ZeroPointGoalPolicy(),
            "heuristic": lambda env: HeuristicPointGoalPolicy(),
        },
        manifest,
    )
    zero = report["policies"]["zero"]["metrics"]
    heuristic = report["policies"]["heuristic"]["metrics"]
    assert zero["episode_count"] == heuristic["episode_count"] == 2
    assert zero["timeout_rate"] == 1.0
    assert heuristic["progress_ratio"]["mean"] > zero["progress_ratio"]["mean"]
    assert report["policies"]["zero"]["manifest_sha256"] == report["policies"][
        "heuristic"
    ]["manifest_sha256"]


def test_real_mujoco_ppo_checkpoint_factory_produces_deterministic_actions(tmp_path) -> None:
    cfg = evaluate_script._load_config()
    train_cfg = normalize_ppo_train_cfg(evaluate_script._algo_config_dict(cfg))
    algorithm_cfg = train_cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        algorithm_cfg["enable_compile"] = False
    train_cfg.setdefault("runner", {})["logger"] = "none"

    source_env = make_env()
    source_wrapper = RslRlVecEnvWrapper(source_env, device="cpu")
    source_runner = OnPolicyRunner(source_wrapper, train_cfg, log_dir=None, device="cpu")
    checkpoint = tmp_path / "model_0.pt"
    torch.save(
        {
            "actor_state_dict": source_runner.alg.actor.state_dict(),
            "critic_state_dict": source_runner.alg.critic.state_dict(),
            "optimizer_state_dict": source_runner.alg.optimizer.state_dict(),
            "iter": 0,
            "infos": {},
        },
        checkpoint,
    )
    source_env.close()

    eval_env = make_env()
    policy = evaluate_script._ppo_policy_factory(
        cfg, checkpoint=checkpoint, device="cpu"
    )(eval_env)
    states = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float32)
    goals = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    observations = eval_env.reset_to_initial_conditions(states, goals).obs
    first = policy(observations)
    second = policy(observations)
    assert first.shape == (2, 2)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)
    eval_env.close()
