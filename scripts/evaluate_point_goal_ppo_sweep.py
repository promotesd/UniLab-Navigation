"""Evaluate PPO checkpoints across training seeds on one PointGoal manifest."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.evaluation.point_goal import (
    evaluate_point_goal_policy,
    read_point_goal_manifest,
    write_point_goal_report,
)
from unilab.evaluation.point_goal_ppo import (
    build_point_goal_ppo_policy_factory,
    load_point_goal_ppo_config,
)
from unilab.evaluation.ppo_sweep import (
    aggregate_ppo_checkpoint_results,
    format_ppo_sweep_summary,
    parse_ppo_checkpoint_spec,
)
from unilab.training import BackendAdapter, create_env, ensure_registries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-input", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeat SEED=/path/to/model_ITERATION.pt for every seed/checkpoint.",
    )
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/unilab_point_goal_ppo_sweep.json"),
    )
    return parser.parse_args()


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    specs = [parse_ppo_checkpoint_spec(value) for value in args.checkpoint]
    missing = [str(spec.path) for spec in specs if not spec.path.is_file()]
    if missing:
        raise FileNotFoundError(f"PPO checkpoints do not exist: {missing}")

    ensure_registries()
    cfg = load_point_goal_ppo_config(ROOT_DIR)
    manifest = read_point_goal_manifest(args.manifest_input)
    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="ppo"
    ).build_task_env_cfg_override()
    env_cfg_override["seed"] = manifest.seed

    def env_factory():
        return create_env(
            cfg,
            num_envs=manifest.episode_count,
            env_cfg_override=env_cfg_override,
        )

    checkpoint_results: list[dict[str, Any]] = []
    for spec in specs:
        env = env_factory()
        try:
            policy = build_point_goal_ppo_policy_factory(
                cfg, checkpoint=spec.path, device=args.device
            )(env)
            evaluation = evaluate_point_goal_policy(
                env,
                policy,
                manifest,
                policy_name=f"ppo_seed_{spec.seed}_iteration_{spec.iteration}",
            )
        finally:
            env.close()
        checkpoint_results.append(
            {
                "seed": spec.seed,
                "iteration": spec.iteration,
                "checkpoint": str(spec.path),
                "checkpoint_sha256": _file_sha256(spec.path),
                **evaluation,
            }
        )

    aggregate = aggregate_ppo_checkpoint_results(
        checkpoint_results, expected_seeds=args.expected_seeds
    )
    report = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "config": {
            "task": str(cfg.training.task_name),
            "sim_backend": str(cfg.training.sim_backend),
            "evaluation_episodes": manifest.episode_count,
            "manifest_seed": manifest.seed,
            "training_seeds": aggregate["expected_seeds"],
            "training_num_envs": int(cfg.algo.num_envs),
            "training_rollout_steps": int(cfg.algo.num_steps_per_env),
            "training_max_iterations": int(cfg.algo.max_iterations),
            "checkpoint_interval": int(cfg.algo.save_interval),
            "evaluation_device": args.device,
        },
        "manifest": manifest.to_dict(),
        "checkpoints": checkpoint_results,
        "aggregate": aggregate,
    }
    print(format_ppo_sweep_summary(aggregate))
    output = write_point_goal_report(report, args.output)
    print(f"JSON report: {output}")


if __name__ == "__main__":
    main()
