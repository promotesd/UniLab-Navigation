"""FastSAC checkpoint adapter for fixed-episode PointGoal evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.fast_sac.learner import SACActor
from unilab.base.observations import get_obs_dims, split_obs_dict


def load_point_goal_sac_config(root_dir: str | Path) -> DictConfig:
    """Compose the formal PointGoal SAC task configuration."""
    config_dir = Path(root_dir) / "conf" / "offpolicy"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(
            config_name="config",
            overrides=["algo=sac", "task=sac/diff_drive_point_goal/mujoco"],
        )


class SacPointGoalPolicy:
    """Deterministic NumPy policy backed by a loaded FastSAC actor."""

    def __init__(
        self,
        actor: SACActor,
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
            actions = self.actor.explore(obs_tensor, deterministic=True)
        return actions.cpu().numpy().astype(np.float32, copy=False)


def build_point_goal_sac_policy_factory(
    cfg: DictConfig,
    *,
    checkpoint: str | Path,
    device: str,
):
    """Build an evaluator factory for one FastSAC checkpoint."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAC checkpoint does not exist: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "actor" not in state:
        raise ValueError(
            f"SAC checkpoint must contain actor; found keys {sorted(state.keys())}"
        )

    def factory(env: Any) -> SacPointGoalPolicy:
        obs_dim, _ = get_obs_dims(env.obs_groups_spec)
        action_shape = env.action_space.shape
        if action_shape is None or len(action_shape) != 1:
            raise ValueError("PointGoal SAC requires a one-dimensional action shape")
        torch_device = torch.device(device)
        actor = SACActor(
            obs_dim=obs_dim,
            action_dim=int(action_shape[0]),
            hidden_dim=int(cfg.algo.actor_hidden_dim),
            use_layer_norm=bool(cfg.algo.use_layer_norm),
            device=torch_device,
        )
        actor.load_state_dict(state["actor"])
        actor.eval()
        normalizer = None
        if bool(cfg.algo.obs_normalization):
            normalizer = EmpiricalNormalization(shape=obs_dim, device=torch_device)
            normalizer_state = state.get("obs_normalizer")
            if normalizer_state is None:
                raise ValueError("normalized SAC checkpoint is missing obs_normalizer")
            normalizer.load_state_dict(normalizer_state)
            normalizer.eval()
        return SacPointGoalPolicy(
            actor,
            device=torch_device,
            normalizer=normalizer,
        )

    return factory
