"""Fair cross-algorithm aggregation for PointGoal training runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

POINT_GOAL_ALGORITHMS = ("ppo", "sac", "td3")
_RATE_METRICS = ("success_rate", "collision_rate", "timeout_rate")
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
class AlgorithmRunSpec:
    """Paths identifying one training seed and its held-out evaluation."""

    algorithm: str
    seed: int
    training_summary: Path
    evaluation_report: Path


def parse_algorithm_run_spec(value: str) -> AlgorithmRunSpec:
    """Parse ``ALGORITHM:SEED:TRAINING_SUMMARY:EVALUATION_REPORT``."""
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            "run must use ALGORITHM:SEED:TRAINING_SUMMARY:EVALUATION_REPORT"
        )
    algorithm, seed_text, summary_text, evaluation_text = parts
    if algorithm not in POINT_GOAL_ALGORITHMS:
        raise ValueError(f"unsupported PointGoal algorithm: {algorithm!r}")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise ValueError(f"run seed must be an integer: {seed_text!r}") from exc
    if seed < 0:
        raise ValueError("run seed must be non-negative")
    if not summary_text or not evaluation_text:
        raise ValueError("run summary and evaluation paths must be non-empty")
    return AlgorithmRunSpec(
        algorithm=algorithm,
        seed=seed,
        training_summary=Path(summary_text).expanduser().resolve(),
        evaluation_report=Path(evaluation_text).expanduser().resolve(),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"benchmark input does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark input is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark input must contain a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "std": None, "seed_count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "seed_count": len(values),
    }


def _training_precision(run_config: Mapping[str, Any]) -> str:
    config = run_config["config"]
    training = config["training"]
    if not bool(training.get("use_amp", False)):
        return "float32"
    amp_dtype = config["algo"].get("algo_params", {}).get("amp_dtype", "auto")
    return f"mixed:{amp_dtype}"


def _canonical_task_contract(run_config: Mapping[str, Any]) -> dict[str, Any]:
    config = run_config["config"]
    return {
        "task": run_config["run"]["task"],
        "sim_backend": run_config["run"]["sim_backend"],
        "reward": config.get("reward", {}),
        "environment": {
            key: value for key, value in config.get("env", {}).items() if key != "seed"
        },
    }


def _validate_expected_matrix(
    specs: Sequence[AlgorithmRunSpec],
    expected_seeds: Sequence[int],
) -> list[int]:
    seeds = sorted(set(int(seed) for seed in expected_seeds))
    if not seeds or len(seeds) != len(expected_seeds) or seeds[0] < 0:
        raise ValueError("expected_seeds must be unique non-negative integers")
    seen: set[tuple[str, int]] = set()
    for spec in specs:
        key = (spec.algorithm, spec.seed)
        if key in seen:
            raise ValueError(f"duplicate benchmark run for {spec.algorithm} seed {spec.seed}")
        seen.add(key)
    expected = {
        (algorithm, seed) for algorithm in POINT_GOAL_ALGORITHMS for seed in seeds
    }
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing or extra:
        raise ValueError(f"benchmark run matrix mismatch; missing={missing}, extra={extra}")
    return seeds


def aggregate_point_goal_algorithm_runs(
    specs: Sequence[AlgorithmRunSpec],
    *,
    expected_seeds: Sequence[int],
    environment_step_budget: int,
) -> dict[str, Any]:
    """Validate a controlled experiment and aggregate held-out seed metrics."""
    if environment_step_budget <= 0:
        raise ValueError("environment_step_budget must be positive")
    seeds = _validate_expected_matrix(specs, expected_seeds)
    raw_runs: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None

    for spec in sorted(specs, key=lambda item: (item.algorithm, item.seed)):
        summary = _read_json(spec.training_summary)
        evaluation = _read_json(spec.evaluation_report)
        run_config_path = spec.training_summary.parent / "run_config.json"
        run_config = _read_json(run_config_path)

        if summary.get("status") != "completed":
            raise ValueError(f"training did not complete for {spec.algorithm} seed {spec.seed}")
        for source, actual in (
            ("training summary algorithm", summary.get("algo")),
            ("run config algorithm", run_config["run"].get("algo")),
        ):
            if actual != spec.algorithm:
                raise ValueError(f"{source} is {actual!r}; expected {spec.algorithm!r}")
        for source, actual in (
            ("training summary seed", summary.get("effective_seed")),
            ("run config seed", run_config["run"].get("effective_seed")),
        ):
            if int(actual) != spec.seed:
                raise ValueError(f"{source} is {actual!r}; expected {spec.seed}")
        total_steps = int(summary["total_env_steps"])
        if total_steps != environment_step_budget:
            raise ValueError(
                f"{spec.algorithm} seed {spec.seed} used {total_steps} environment "
                f"steps; expected exactly {environment_step_budget}"
            )

        checkpoint = Path(summary["last_checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"training checkpoint does not exist: {checkpoint}")
        evaluated_checkpoint = evaluation.get("checkpoints", {}).get(spec.algorithm)
        if evaluated_checkpoint is None or Path(evaluated_checkpoint).resolve() != checkpoint:
            raise ValueError(
                f"evaluation checkpoint does not match training for {spec.algorithm} "
                f"seed {spec.seed}"
            )
        policy_result = evaluation.get("policies", {}).get(spec.algorithm)
        if not isinstance(policy_result, dict):
            raise ValueError(
                f"evaluation lacks {spec.algorithm} policy result for seed {spec.seed}"
            )
        checkpoint_hash = _sha256(checkpoint)
        if policy_result.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(
                f"evaluation checkpoint hash does not match training for "
                f"{spec.algorithm} seed {spec.seed}"
            )
        manifest = evaluation.get("manifest")
        if not isinstance(manifest, dict) or not manifest.get("sha256"):
            raise ValueError("evaluation report lacks a hashed fixed-episode manifest")
        if policy_result.get("manifest_sha256") != manifest["sha256"]:
            raise ValueError("policy result manifest hash does not match its report")

        task_contract = _canonical_task_contract(run_config)
        evaluation_config = evaluation["config"]
        evaluation_device = evaluation.get("evaluation", {}).get("device")
        if not evaluation_device:
            raise ValueError("evaluation report must record its inference device")
        training_device = str(run_config["run"]["device"])
        precision = _training_precision(run_config)
        protocol = {
            **task_contract,
            "training_num_envs": int(run_config["config"]["algo"]["num_envs"]),
            "training_device": training_device,
            "training_precision": precision,
            "evaluation_device": str(evaluation_device),
            "evaluation_episodes": int(evaluation_config["episodes"]),
            "evaluation_seed": int(evaluation_config["seed"]),
            "max_episode_steps": int(evaluation_config["max_episode_steps"]),
            "ctrl_dt": float(evaluation_config["ctrl_dt"]),
            "goal_tolerance": float(evaluation_config["goal_tolerance"]),
            "manifest_sha256": manifest["sha256"],
            "hardware": run_config["run"].get("hardware"),
            "git": run_config["run"].get("git"),
        }
        if evaluation_config["task"] != task_contract["task"]:
            raise ValueError("training and evaluation task names do not match")
        if evaluation_config["sim_backend"] != task_contract["sim_backend"]:
            raise ValueError("training and evaluation simulation backends do not match")
        if common is None:
            common = protocol
        elif protocol != common:
            differing = sorted(
                key for key in protocol if protocol.get(key) != common.get(key)
            )
            raise ValueError(
                f"benchmark protocol differs for {spec.algorithm} seed {spec.seed}: "
                f"{differing}"
            )

        raw_runs.append(
            {
                "algorithm": spec.algorithm,
                "seed": spec.seed,
                "environment_steps": total_steps,
                "training_wall_time_sec": float(summary["training_wall_time_sec"]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "training_summary": str(spec.training_summary),
                "training_summary_sha256": _sha256(spec.training_summary),
                "run_config": str(run_config_path),
                "run_config_sha256": _sha256(run_config_path),
                "evaluation_report": str(spec.evaluation_report),
                "evaluation_report_sha256": _sha256(spec.evaluation_report),
                "metrics": policy_result["metrics"],
            }
        )

    assert common is not None
    algorithms: dict[str, Any] = {}
    for algorithm in POINT_GOAL_ALGORITHMS:
        runs = [run for run in raw_runs if run["algorithm"] == algorithm]
        metrics: dict[str, Any] = {}
        for metric_name in _RATE_METRICS:
            metrics[metric_name] = _summary(
                [float(run["metrics"][metric_name]) for run in runs]
            )
        for metric_name in _DISTRIBUTION_METRICS:
            values = [run["metrics"][metric_name]["mean"] for run in runs]
            metrics[metric_name] = _summary(
                [float(value) for value in values if value is not None]
            )
        algorithms[algorithm] = {
            "seeds": seeds,
            "environment_steps_per_seed": environment_step_budget,
            "training_wall_time_sec": _summary(
                [float(run["training_wall_time_sec"]) for run in runs]
            ),
            "sample_efficiency": {
                "measurement": "held_out_metrics_at_fixed_environment_step_budget",
                "success_rate_at_budget": metrics["success_rate"],
                "spl_at_budget": metrics["spl"],
            },
            "metrics": metrics,
        }

    return {
        "schema_version": 1,
        "protocol": {
            "algorithms": list(POINT_GOAL_ALGORITHMS),
            "seeds": seeds,
            "environment_step_budget_per_seed": environment_step_budget,
            **common,
        },
        "algorithms": algorithms,
        "runs": raw_runs,
    }


def format_point_goal_algorithm_comparison(report: Mapping[str, Any]) -> str:
    """Format the controlled comparison without declaring a universal winner."""
    rows = ["algorithm seeds env_steps success spl timeout final_dist train_sec"]
    for algorithm in POINT_GOAL_ALGORITHMS:
        result = report["algorithms"][algorithm]
        metrics = result["metrics"]

        def mean_std(metric: Mapping[str, Any]) -> str:
            if metric["mean"] is None:
                return "n/a"
            return f"{metric['mean']:.3f}+/-{metric['std']:.3f}"

        rows.append(
            f"{algorithm:<9} {len(result['seeds']):>5d} "
            f"{result['environment_steps_per_seed']:>9d} "
            f"{mean_std(metrics['success_rate']):>13} "
            f"{mean_std(metrics['spl']):>13} "
            f"{mean_std(metrics['timeout_rate']):>13} "
            f"{mean_std(metrics['final_distance']):>13} "
            f"{mean_std(result['training_wall_time_sec']):>13}"
        )
    return "\n".join(rows)


def write_point_goal_algorithm_comparison(
    report: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Write strict machine-readable comparison JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
