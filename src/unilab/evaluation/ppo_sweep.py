"""Cross-seed aggregation for fixed-manifest PPO checkpoint evaluation."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_CHECKPOINT_PATTERN = re.compile(r"^model_(\d+)\.pt$")
_DISTRIBUTION_METRICS = (
    "initial_distance",
    "final_distance",
    "progress_ratio",
    "episode_length",
    "successful_episode_length",
    "path_length",
    "spl",
)


@dataclass(frozen=True)
class PpoCheckpointSpec:
    """One training seed and checkpoint iteration selected for evaluation."""

    seed: int
    iteration: int
    path: Path


def parse_ppo_checkpoint_spec(value: str) -> PpoCheckpointSpec:
    """Parse ``SEED=/path/to/model_ITERATION.pt``."""
    seed_text, separator, path_text = value.partition("=")
    if not separator or not seed_text or not path_text:
        raise ValueError("checkpoint must use SEED=/path/to/model_ITERATION.pt")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ValueError(f"checkpoint seed must be an integer: {seed_text!r}") from exc
    if seed < 0:
        raise ValueError("checkpoint seed must be non-negative")
    path = Path(path_text).expanduser().resolve()
    match = _CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError("checkpoint filename must use model_ITERATION.pt")
    return PpoCheckpointSpec(seed=seed, iteration=int(match.group(1)), path=path)


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "seed_count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "seed_count": len(values),
    }


def aggregate_ppo_checkpoint_results(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Aggregate independent evaluator metrics by checkpoint iteration."""
    expected = sorted(set(int(seed) for seed in expected_seeds))
    if not expected:
        raise ValueError("expected_seeds must not be empty")
    if expected[0] < 0:
        raise ValueError("expected_seeds must be non-negative")
    if len(expected) != len(expected_seeds):
        raise ValueError("expected_seeds must not contain duplicates")

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for result in results:
        seed = int(result["seed"])
        iteration = int(result["iteration"])
        key = (seed, iteration)
        if key in seen:
            raise ValueError(f"duplicate PPO result for seed={seed}, iteration={iteration}")
        seen.add(key)
        grouped[iteration].append(result)
    if not grouped:
        raise ValueError("at least one PPO checkpoint result is required")

    iterations: dict[str, Any] = {}
    for iteration in sorted(grouped):
        iteration_results = grouped[iteration]
        actual_seeds = sorted(int(result["seed"]) for result in iteration_results)
        if actual_seeds != expected:
            raise ValueError(
                f"iteration {iteration} has seeds {actual_seeds}; expected {expected}"
            )
        aggregated_metrics: dict[str, Any] = {}
        for metric_name in ("success_rate", "collision_rate", "timeout_rate"):
            aggregated_metrics[metric_name] = _summary(
                [float(result["metrics"][metric_name]) for result in iteration_results]
            )
        for metric_name in _DISTRIBUTION_METRICS:
            values = [
                result["metrics"][metric_name]["mean"] for result in iteration_results
            ]
            aggregated_metrics[metric_name] = _summary(
                [float(value) for value in values if value is not None]
            )
        iterations[str(iteration)] = {
            "iteration": iteration,
            "seeds": actual_seeds,
            "metrics": aggregated_metrics,
        }
    return {"expected_seeds": expected, "iterations": iterations}


def format_ppo_sweep_summary(aggregate: Mapping[str, Any]) -> str:
    """Format cross-seed PPO checkpoint aggregates as a console table."""
    rows = ["iteration seeds success timeout final_dist progress ep_len"]
    for iteration, result in aggregate["iterations"].items():
        metrics = result["metrics"]

        def mean_std(metric_name: str) -> str:
            metric = metrics[metric_name]
            if metric["mean"] is None:
                return "n/a"
            return f"{metric['mean']:.3f}+/-{metric['std']:.3f}"

        rows.append(
            f"{int(iteration):>9d} {len(result['seeds']):>5d} "
            f"{mean_std('success_rate'):>13} {mean_std('timeout_rate'):>13} "
            f"{mean_std('final_distance'):>13} {mean_std('progress_ratio'):>13} "
            f"{mean_std('episode_length'):>13}"
        )
    return "\n".join(rows)
