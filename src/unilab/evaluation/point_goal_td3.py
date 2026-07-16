"""FastTD3 checkpoint adapter for fixed-episode PointGoal evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.fast_td3.learner import TD3Actor
from unilab.base.observations import get_obs_dims, split_obs_dict


def load_point_goal_td3_config(root_dir: str | Path) -> DictConfig:
    """Compose the formal PointGoal TD3 task configuration."""
    config_dir = Path(root_dir) / "conf" / "offpolicy"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(
            config_name="config",
            overrides=["algo=td3", "task=td3/diff_drive_point_goal/mujoco"],
        )


class Td3PointGoalPolicy:
    """Deterministic NumPy policy backed by a loaded FastTD3 actor."""

    def __init__(
        self,
        actor: TD3Actor,
        *,
        device: torch.device,
        normalizer: EmpiricalNormalization | None,
    ) -> None:
        self.actor = actor
        self.device = device
        self.normalizer = normalizer

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        actor_obs, _ = split_obs_dict(observations)
        obs_tensor = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        if self.normalizer is not None:
            obs_tensor = self.normalizer(obs_tensor, update=False)
        with torch.inference_mode():
            actions = self.actor(obs_tensor)
        return actions.cpu().numpy().astype(np.float32, copy=False)


def build_point_goal_td3_policy_factory(
    cfg: DictConfig,
    *,
    checkpoint: str | Path,
    device: str,
):
    """Build an evaluator factory for one FastTD3 checkpoint."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TD3 checkpoint does not exist: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "actor" not in state:
        raise ValueError(
            f"TD3 checkpoint must contain actor; found keys {sorted(state.keys())}"
        )

    def factory(env: Any) -> Td3PointGoalPolicy:
        obs_dim, _ = get_obs_dims(env.obs_groups_spec)
        action_shape = env.action_space.shape
        if action_shape is None or len(action_shape) != 1:
            raise ValueError("PointGoal TD3 requires a one-dimensional action shape")
        torch_device = torch.device(device)
        actor = TD3Actor(
            obs_dim=obs_dim,
            n_act=int(action_shape[0]),
            num_envs=env.num_envs,
            init_scale=float(cfg.algo.algo_params.init_scale),
            hidden_dim=int(cfg.algo.actor_hidden_dim),
            log_std_min=float(cfg.algo.algo_params.log_std_min),
            log_std_max=float(cfg.algo.algo_params.log_std_max),
            device=torch_device,
        )
        actor_state = {
            key: value for key, value in state["actor"].items() if key != "noise_scales"
        }
        actor.load_state_dict(actor_state, strict=False)
        actor.eval()
        normalizer = None
        if bool(cfg.algo.obs_normalization):
            normalizer = EmpiricalNormalization(shape=obs_dim, device=torch_device)
            normalizer_state = state.get("obs_normalizer")
            if normalizer_state is None:
                raise ValueError("normalized TD3 checkpoint is missing obs_normalizer")
            normalizer.load_state_dict(normalizer_state)
            normalizer.eval()
        return Td3PointGoalPolicy(
            actor,
            device=torch_device,
            normalizer=normalizer,
        )

    return factory
