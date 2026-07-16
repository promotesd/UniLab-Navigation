"""Benchmark UniLab PointGoal against a controlled standalone MDP loop."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.benchmarks import (
    StandalonePointGoalMujoco,
    StandalonePointGoalSpec,
    aggregate_point_goal_framework_benchmark,
    format_point_goal_framework_benchmark,
    write_point_goal_framework_benchmark,
)
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
)
from unilab.evaluation import generate_point_goal_manifest


class PolicyValueNetwork(nn.Module):
    """Identical benchmark actor/value workload used by both paths."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, 2)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(observation)
        return torch.tanh(self.actor(features)), self.value(features).squeeze(-1)


class FrameworkPath:
    def __init__(self, cfg: DiffDrivePointGoalCfg, environment_count: int) -> None:
        self.env = DiffDrivePointGoalMujocoEnv(cfg=cfg, num_envs=environment_count)
        self.env.set_autoreset(False)

    def reset(self, states: np.ndarray, goals: np.ndarray) -> np.ndarray:
        return self.env.reset_to_initial_conditions(states, goals).obs["obs"]

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = self.env.step(actions)
        return state.obs["obs"], state.reward

    def close(self) -> None:
        self.env.close()


class StandalonePath:
    def __init__(self, spec: StandalonePointGoalSpec, environment_count: int) -> None:
        self.env = StandalonePointGoalMujoco(spec, environment_count)

    def reset(self, states: np.ndarray, goals: np.ndarray) -> np.ndarray:
        return self.env.reset_to_initial_conditions(states, goals)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        observation, reward, _, _ = self.env.step(actions)
        return observation, reward

    def close(self) -> None:
        self.env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--warmup-repetitions", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/unilab_point_goal_framework_benchmark.json"),
    )
    return parser.parse_args()


def _measure(
    path: FrameworkPath | StandalonePath,
    *,
    states: np.ndarray,
    goals: np.ndarray,
    action_sequence: np.ndarray,
    initial_model_state: dict[str, Any],
    hidden_dim: int,
    learning_rate: float,
) -> dict[str, float]:
    environment_count = len(states)
    rollout_steps = len(action_sequence)
    path.reset(states, goals)
    start = time.perf_counter()
    reward_checksum = 0.0
    for actions in action_sequence:
        _, reward = path.step(actions)
        reward_checksum += float(np.sum(reward))
    simulator_seconds = time.perf_counter() - start

    model = PolicyValueNetwork(hidden_dim)
    model.load_state_dict(initial_model_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    observation = path.reset(states, goals)
    losses: list[torch.Tensor] = []
    start = time.perf_counter()
    for _ in range(rollout_steps):
        observation_tensor = torch.from_numpy(observation).to(torch.float32)
        actions, values = model(observation_tensor)
        observation, reward = path.step(actions.detach().numpy())
        reward_tensor = torch.from_numpy(np.asarray(reward)).to(torch.float32)
        losses.append(torch.mean((values - reward_tensor) ** 2) + 1.0e-3 * torch.mean(actions**2))
    optimizer.zero_grad(set_to_none=True)
    torch.stack(losses).mean().backward()
    optimizer.step()
    training_seconds = time.perf_counter() - start
    transitions = environment_count * rollout_steps
    return {
        "simulator_seconds": simulator_seconds,
        "simulator_steps_per_second": transitions / simulator_seconds,
        "training_iteration_seconds": training_seconds,
        "training_steps_per_second": transitions / training_seconds,
        "reward_checksum": reward_checksum,
    }


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _mujoco_version() -> str:
    try:
        return importlib.metadata.version("mujoco")
    except importlib.metadata.PackageNotFoundError:
        import mujoco

        return str(getattr(mujoco, "__version__", "unknown"))


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    if min(
        args.environments,
        args.rollout_steps,
        args.repetitions,
        args.hidden_dim,
        args.chunk_size,
    ) <= 0:
        raise ValueError("positive benchmark dimensions are required")
    if args.warmup_repetitions < 0 or args.learning_rate <= 0.0:
        raise ValueError("benchmark warmup and learning rate are invalid")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("benchmark seeds must be unique and non-negative")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(991)
    master_model = PolicyValueNetwork(args.hidden_dim)
    initial_model_state = {
        key: value.detach().clone() for key, value in master_model.state_dict().items()
    }
    cfg = DiffDrivePointGoalCfg(
        seed=args.seeds[0],
        chunk_size=args.chunk_size,
        adaptive_chunk_size=False,
    )
    spec = StandalonePointGoalSpec(chunk_size=args.chunk_size)
    paths: dict[str, FrameworkPath | StandalonePath] = {
        "framework": FrameworkPath(cfg, args.environments),
        "standalone": StandalonePath(spec, args.environments),
    }
    runs: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    try:
        for seed in args.seeds:
            manifest = generate_point_goal_manifest(
                episode_count=args.environments,
                seed=seed,
                min_goal_distance=cfg.min_goal_distance,
                max_goal_distance=cfg.max_goal_distance,
            )
            rng = np.random.default_rng(seed + 17)
            actions = rng.uniform(
                -1.0,
                1.0,
                size=(args.rollout_steps, args.environments, 2),
            ).astype(np.float32)
            for _ in range(args.warmup_repetitions):
                for path_name in ("framework", "standalone"):
                    _measure(
                        paths[path_name],
                        states=manifest.robot_states,
                        goals=manifest.goals,
                        action_sequence=actions,
                        initial_model_state=initial_model_state,
                        hidden_dim=args.hidden_dim,
                        learning_rate=args.learning_rate,
                    )
            order_rng = random.Random(seed)
            for repetition in range(args.repetitions):
                order = ["framework", "standalone"]
                order_rng.shuffle(order)
                orders.append({"seed": seed, "repetition": repetition, "order": order})
                for order_position, path_name in enumerate(order):
                    measurement = _measure(
                        paths[path_name],
                        states=manifest.robot_states,
                        goals=manifest.goals,
                        action_sequence=actions,
                        initial_model_state=initial_model_state,
                        hidden_dim=args.hidden_dim,
                        learning_rate=args.learning_rate,
                    )
                    runs.append(
                        {
                            "seed": seed,
                            "repetition": repetition,
                            "order_position": order_position,
                            "path": path_name,
                            **measurement,
                        }
                    )
    finally:
        for path in paths.values():
            path.close()
    report = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "protocol": {
            "task": "DiffDrivePointGoal",
            "backend": "mujoco",
            "model_file": spec.model_file,
            "model_file_sha256": _file_sha256(spec.model_file),
            "environments": args.environments,
            "rollout_steps": args.rollout_steps,
            "warmup_repetitions": args.warmup_repetitions,
            "measured_repetitions_per_seed": args.repetitions,
            "seeds": args.seeds,
            "sim_dt": spec.sim_dt,
            "ctrl_dt": spec.ctrl_dt,
            "chunk_size": args.chunk_size,
            "autoreset": False,
            "precision": "float32",
            "device": "cpu",
            "torch_threads": 1,
            "network": {"input": 5, "hidden": [args.hidden_dim, args.hidden_dim], "action": 2},
            "network_initialization_seed": 991,
            "optimizer": {"name": "Adam", "learning_rate": args.learning_rate},
            "loss": "value_mse_plus_0.001_action_l2",
            "action_sequence": "seeded_uniform_-1_1_reused_by_both_paths",
        },
        "software_hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": _mujoco_version(),
        },
        "measurement_orders": orders,
        "runs": runs,
        "aggregate": aggregate_point_goal_framework_benchmark(runs),
    }
    print(format_point_goal_framework_benchmark(report))
    output = write_point_goal_framework_benchmark(report, args.output)
    print(f"JSON report: {output}")


if __name__ == "__main__":
    main()
