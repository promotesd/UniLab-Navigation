"""Aggregation and reporting for controlled framework/standalone timings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_PATHS = ("framework", "standalone")
_METRICS = (
    "simulator_seconds",
    "simulator_steps_per_second",
    "training_iteration_seconds",
    "training_steps_per_second",
)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("benchmark summary values must be non-empty and finite")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "repetition_count": len(array),
    }


def aggregate_point_goal_framework_benchmark(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate a complete paired randomized measurement matrix."""
    if not runs:
        raise ValueError("benchmark requires measured runs")
    keys: set[tuple[int, int, str]] = set()
    pairs: dict[tuple[int, int], set[str]] = {}
    for run in runs:
        path = str(run["path"])
        if path not in _PATHS:
            raise ValueError(f"unknown benchmark path: {path}")
        key = (int(run["seed"]), int(run["repetition"]), path)
        if key in keys:
            raise ValueError(f"duplicate benchmark run: {key}")
        keys.add(key)
        pairs.setdefault(key[:2], set()).add(path)
        for metric in _METRICS:
            if not np.isfinite(float(run[metric])) or float(run[metric]) <= 0.0:
                raise ValueError(f"benchmark metric {metric} must be finite and positive")
    incomplete = sorted(pair for pair, paths in pairs.items() if paths != set(_PATHS))
    if incomplete:
        raise ValueError(f"benchmark pairs are incomplete: {incomplete}")
    maximum_reward_checksum_difference = 0.0
    for pair in pairs:
        pair_runs = {
            str(run["path"]): run
            for run in runs
            if (int(run["seed"]), int(run["repetition"])) == pair
        }
        try:
            difference = abs(
                float(pair_runs["framework"]["reward_checksum"])
                - float(pair_runs["standalone"]["reward_checksum"])
            )
        except KeyError as exc:
            raise ValueError("benchmark runs must record reward_checksum") from exc
        maximum_reward_checksum_difference = max(
            maximum_reward_checksum_difference,
            difference,
        )
    if maximum_reward_checksum_difference > 1.0e-5:
        raise ValueError("framework and standalone reward checksums differ")

    paths: dict[str, Any] = {}
    for path in _PATHS:
        selected = [run for run in runs if run["path"] == path]
        paths[path] = {
            metric: _summary([float(run[metric]) for run in selected])
            for metric in _METRICS
        }
    framework = paths["framework"]
    standalone = paths["standalone"]
    overhead = {
        "simulator_time_percent": 100.0
        * (
            framework["simulator_seconds"]["mean"]
            / standalone["simulator_seconds"]["mean"]
            - 1.0
        ),
        "training_iteration_time_percent": 100.0
        * (
            framework["training_iteration_seconds"]["mean"]
            / standalone["training_iteration_seconds"]["mean"]
            - 1.0
        ),
    }
    return {
        "paths": paths,
        "framework_overhead_percent": overhead,
        "contract_checks": {
            "maximum_paired_reward_checksum_difference": (
                maximum_reward_checksum_difference
            )
        },
    }


def format_point_goal_framework_benchmark(report: Mapping[str, Any]) -> str:
    """Format measured workload-specific results without a global speed claim."""
    rows = ["path       sim_steps/s train_steps/s sim_sec train_sec"]
    for path in _PATHS:
        result = report["aggregate"]["paths"][path]

        def value(metric: str) -> str:
            summary = result[metric]
            precision = 4 if metric.endswith("seconds") else 1
            return (
                f"{summary['mean']:.{precision}f}"
                f"+/-{summary['std']:.{precision}f}"
            )

        rows.append(
            f"{path:<10} {value('simulator_steps_per_second'):>16} "
            f"{value('training_steps_per_second'):>16} "
            f"{value('simulator_seconds'):>12} "
            f"{value('training_iteration_seconds'):>12}"
        )
    overhead = report["aggregate"]["framework_overhead_percent"]
    rows.append(
        "framework overhead under this protocol: "
        f"simulator={overhead['simulator_time_percent']:.2f}% "
        f"training_iteration={overhead['training_iteration_time_percent']:.2f}%"
    )
    return "\n".join(rows)


def write_point_goal_framework_benchmark(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
