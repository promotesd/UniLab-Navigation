"""Deterministic fixed-episode evaluation for PointGoal policies."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unilab.envs.navigation.diff_drive.controllers import (
    STATIONARY_ACTION,
    PointGoalPolicy,
)


class PpoPointGoalPolicy:
    """NumPy evaluator adapter around a deterministic PPO inference callable."""

    def __init__(
        self,
        inference_policy: Callable[[Any], Any],
        observation_adapter: Callable[[dict[str, np.ndarray]], Any],
    ) -> None:
        self._inference_policy = inference_policy
        self._observation_adapter = observation_adapter

    def __call__(self, observations: dict[str, np.ndarray]) -> np.ndarray:
        policy_input = self._observation_adapter(observations)
        actions = self._inference_policy(policy_input)
        detach = getattr(actions, "detach", None)
        if callable(detach):
            actions = detach()
        cpu = getattr(actions, "cpu", None)
        if callable(cpu):
            actions = cpu()
        to_numpy = getattr(actions, "numpy", None)
        if callable(to_numpy):
            actions = to_numpy()
        return np.asarray(actions, dtype=np.float32)


@dataclass(frozen=True)
class PointGoalManifest:
    """Serializable set of fixed PointGoal episode initial conditions."""

    seed: int
    robot_states: np.ndarray
    goals: np.ndarray

    def __post_init__(self) -> None:
        states = np.asarray(self.robot_states, dtype=np.float32)
        goals = np.asarray(self.goals, dtype=np.float32)
        if self.seed < 0:
            raise ValueError("manifest seed must be non-negative")
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError("manifest robot_states must have shape (episodes, 3)")
        if goals.shape != (len(states), 2):
            raise ValueError("manifest goals must have shape (episodes, 2)")
        if len(states) == 0:
            raise ValueError("manifest must contain at least one episode")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(goals)):
            raise ValueError("manifest values must be finite")
        object.__setattr__(self, "robot_states", states.copy())
        object.__setattr__(self, "goals", goals.copy())

    @property
    def episode_count(self) -> int:
        return len(self.robot_states)

    def to_dict(self) -> dict[str, Any]:
        conditions = [
            {
                "episode_id": episode_id,
                "robot_state": self.robot_states[episode_id].tolist(),
                "goal_position": self.goals[episode_id].tolist(),
            }
            for episode_id in range(self.episode_count)
        ]
        payload: dict[str, Any] = {
            "seed": self.seed,
            "episode_count": self.episode_count,
            "initial_conditions": conditions,
        }
        payload["sha256"] = _manifest_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PointGoalManifest:
        conditions = payload.get("initial_conditions")
        if not isinstance(conditions, list):
            raise ValueError("manifest initial_conditions must be a list")
        declared_count = payload.get("episode_count")
        if declared_count is not None and int(declared_count) != len(conditions):
            raise ValueError("manifest episode_count does not match initial_conditions")
        episode_ids = [condition.get("episode_id") for condition in conditions]
        if episode_ids != list(range(len(conditions))):
            raise ValueError("manifest episode_id values must be contiguous and zero-based")
        states = [condition["robot_state"] for condition in conditions]
        goals = [condition["goal_position"] for condition in conditions]
        manifest = cls(seed=int(payload["seed"]), robot_states=states, goals=goals)
        expected_hash = payload.get("sha256")
        if expected_hash is not None and expected_hash != manifest.to_dict()["sha256"]:
            raise ValueError("manifest sha256 does not match its initial conditions")
        return manifest


def _manifest_sha256(payload: Mapping[str, Any]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "sha256"}
    encoded = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def generate_point_goal_manifest(
    *,
    episode_count: int,
    seed: int,
    min_goal_distance: float,
    max_goal_distance: float,
) -> PointGoalManifest:
    """Generate starts using the task's reset distribution and draw order."""
    if episode_count <= 0:
        raise ValueError("episode_count must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if min_goal_distance <= 0.0 or max_goal_distance <= min_goal_distance:
        raise ValueError("goal distance bounds are invalid")

    rng = np.random.default_rng(seed)
    headings = rng.uniform(-np.pi, np.pi, size=episode_count)
    distances = rng.uniform(min_goal_distance, max_goal_distance, size=episode_count)
    goal_angles = rng.uniform(-np.pi, np.pi, size=episode_count)
    robot_states = np.zeros((episode_count, 3), dtype=np.float32)
    robot_states[:, 2] = headings
    goals = np.stack(
        [distances * np.cos(goal_angles), distances * np.sin(goal_angles)], axis=1
    ).astype(np.float32)
    return PointGoalManifest(seed=seed, robot_states=robot_states, goals=goals)


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def _validate_actions(actions: np.ndarray, num_envs: int) -> np.ndarray:
    action_array = np.asarray(actions, dtype=np.float32)
    if action_array.shape != (num_envs, 2):
        raise ValueError(f"policy actions must have shape {(num_envs, 2)}, got {action_array.shape}")
    if not np.all(np.isfinite(action_array)):
        raise ValueError("policy actions must contain only finite values")
    return np.clip(action_array, -1.0, 1.0)


def evaluate_point_goal_policy(
    env: Any,
    policy: PointGoalPolicy,
    manifest: PointGoalManifest,
    *,
    policy_name: str,
) -> dict[str, Any]:
    """Evaluate one policy once per manifest row without autoreset."""
    if env.num_envs != manifest.episode_count:
        raise ValueError(
            f"environment has {env.num_envs} slots but manifest has "
            f"{manifest.episode_count} episodes"
        )
    max_episode_steps = env.cfg.max_episode_steps
    if max_episode_steps is None or max_episode_steps <= 0:
        raise ValueError("PointGoal evaluation requires a positive episode limit")

    env.set_autoreset(False)
    state = env.reset_to_initial_conditions(manifest.robot_states, manifest.goals)
    if not np.allclose(env.robot_states, manifest.robot_states, atol=1.0e-5, rtol=0.0):
        raise RuntimeError("environment did not apply the manifest robot states")
    if not np.allclose(env.goals, manifest.goals, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("environment did not apply the manifest goal positions")

    initial_distance = np.linalg.norm(manifest.goals - manifest.robot_states[:, :2], axis=1)
    final_distance = np.full(manifest.episode_count, np.nan, dtype=np.float64)
    episode_length = np.zeros(manifest.episode_count, dtype=np.int64)
    success = np.zeros(manifest.episode_count, dtype=bool)
    timeout = np.zeros(manifest.episode_count, dtype=bool)
    active = np.ones(manifest.episode_count, dtype=bool)

    for _ in range(int(max_episode_steps)):
        actions = _validate_actions(policy(state.obs), manifest.episode_count)
        actions[~active] = STATIONARY_ACTION
        state = env.step(actions)
        done_now = active & (state.terminated | state.truncated)
        if np.any(done_now):
            final_distance[done_now] = np.asarray(state.info["distance_to_goal"])[done_now]
            episode_length[done_now] = np.asarray(state.info["steps"])[done_now]
            success[done_now] = state.terminated[done_now]
            timeout[done_now] = state.truncated[done_now] & ~state.terminated[done_now]
            active[done_now] = False
        if not np.any(active):
            break

    if np.any(active):
        raise RuntimeError(
            f"{int(np.count_nonzero(active))} episodes did not finish within the configured limit"
        )
    progress_ratio = (initial_distance - final_distance) / np.maximum(
        initial_distance, np.finfo(np.float32).eps
    )
    episodes = [
        {
            "episode_id": episode_id,
            "success": bool(success[episode_id]),
            "timeout": bool(timeout[episode_id]),
            "initial_distance": float(initial_distance[episode_id]),
            "final_distance": float(final_distance[episode_id]),
            "progress_ratio": float(progress_ratio[episode_id]),
            "episode_length": int(episode_length[episode_id]),
        }
        for episode_id in range(manifest.episode_count)
    ]
    metrics = {
        "episode_count": manifest.episode_count,
        "success_rate": float(np.mean(success)),
        "timeout_rate": float(np.mean(timeout)),
        "initial_distance": _distribution(initial_distance),
        "final_distance": _distribution(final_distance),
        "progress_ratio": _distribution(progress_ratio),
        "episode_length": _distribution(episode_length),
        "successful_episode_length": _distribution(episode_length[success]),
    }
    return {
        "policy": policy_name,
        "manifest_sha256": manifest.to_dict()["sha256"],
        "metrics": metrics,
        "episodes": episodes,
    }


PolicyFactory = Callable[[Any], PointGoalPolicy]


def evaluate_point_goal_policies(
    env_factory: Callable[[], Any],
    policy_factories: Mapping[str, PolicyFactory],
    manifest: PointGoalManifest,
) -> dict[str, Any]:
    """Evaluate policy factories on fresh environments and one shared manifest."""
    if not policy_factories:
        raise ValueError("at least one policy must be configured")
    results: dict[str, Any] = {}
    for policy_name, policy_factory in policy_factories.items():
        env = env_factory()
        try:
            policy = policy_factory(env)
            results[policy_name] = evaluate_point_goal_policy(
                env, policy, manifest, policy_name=policy_name
            )
        finally:
            env.close()
    return {"manifest": manifest.to_dict(), "policies": results}


def format_point_goal_summary(report: Mapping[str, Any]) -> str:
    """Format the aggregate metrics as a human-readable console table."""
    header = (
        "policy     episodes success timeout initial_dist final_dist progress "
        "ep_len success_len"
    )
    rows = [header]
    policies = report.get("policies", {})
    for name, result in policies.items():
        metrics = result["metrics"]

        def mean_std(metric_name: str) -> str:
            metric = metrics[metric_name]
            if metric["mean"] is None:
                return "n/a"
            return f"{metric['mean']:.3f}+/-{metric['std']:.3f}"

        rows.append(
            f"{name:<10} {metrics['episode_count']:>8d} "
            f"{metrics['success_rate']:>7.3f} {metrics['timeout_rate']:>7.3f} "
            f"{mean_std('initial_distance'):>12} {mean_std('final_distance'):>12} "
            f"{mean_std('progress_ratio'):>12} {mean_std('episode_length'):>12} "
            f"{mean_std('successful_episode_length'):>12}"
        )
    return "\n".join(rows)


def write_point_goal_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write strict machine-readable JSON without NaN values."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def read_point_goal_manifest(input_path: str | Path) -> PointGoalManifest:
    """Read a standalone manifest or the embedded manifest from an evaluation report."""
    path = Path(input_path)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"PointGoal manifest does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"PointGoal manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("PointGoal manifest JSON must contain an object")
    manifest_payload = payload.get("manifest", payload)
    if not isinstance(manifest_payload, dict):
        raise ValueError("evaluation report manifest must contain an object")
    return PointGoalManifest.from_dict(manifest_payload)


def write_point_goal_manifest(
    manifest: PointGoalManifest,
    output_path: str | Path,
) -> Path:
    """Write a reusable strict JSON initial-condition manifest."""
    return write_point_goal_report(manifest.to_dict(), output_path)
