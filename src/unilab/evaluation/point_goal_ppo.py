"""RSL-RL PPO integration for independent PointGoal evaluation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.evaluation.point_goal import PpoPointGoalPolicy
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg


def load_point_goal_ppo_config(root_dir: str | Path) -> DictConfig:
    """Compose the formal PointGoal PPO task configuration."""
    config_dir = Path(root_dir) / "conf" / "ppo"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(
            config_name="config",
            overrides=["task=diff_drive_point_goal/mujoco"],
        )


def ppo_algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    """Resolve the Hydra algorithm section to a detached dictionary."""
    value = OmegaConf.to_container(cfg.algo, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("cfg.algo must resolve to a dictionary")
    return deepcopy(cast(dict[str, Any], value))


def build_point_goal_ppo_policy_factory(
    cfg: DictConfig,
    *,
    checkpoint: str | Path,
    device: str,
):
    """Build a deterministic evaluator policy factory for one PPO checkpoint."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint_path}")
    checkpoint_keys = set(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True).keys()
    )
    if "actor_state_dict" not in checkpoint_keys:
        raise ValueError(
            "PPO checkpoint must contain actor_state_dict; "
            f"found keys {sorted(checkpoint_keys)}"
        )

    def factory(env: Any) -> PpoPointGoalPolicy:
        try:
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as exc:  # pragma: no cover - dependency installation error
            raise RuntimeError("rsl-rl-lib is required to evaluate PPO checkpoints") from exc

        wrapped_env = RslRlVecEnvWrapper(env, device=device)
        train_cfg = normalize_ppo_train_cfg(ppo_algo_config_dict(cfg))
        algorithm_cfg = train_cfg.get("algorithm")
        if isinstance(algorithm_cfg, dict):
            algorithm_cfg["enable_compile"] = False
        train_cfg.setdefault("runner", {})["logger"] = "none"
        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
        runner.load(
            str(checkpoint_path),
            load_cfg={
                "actor": True,
                "critic": False,
                "optimizer": False,
                "iteration": False,
                "rnd": False,
            },
            map_location=device,
        )
        inference_policy = runner.get_inference_policy(device=device)

        def infer(observations: Any) -> Any:
            with torch.inference_mode():
                return inference_policy(observations)

        return PpoPointGoalPolicy(
            infer,
            lambda observations: wrapped_env.observations_to_tensordict(observations),
        )

    return factory
