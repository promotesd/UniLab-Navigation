"""Evaluate PointGoal policies on one deterministic fixed-episode manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.envs.navigation.diff_drive import (
    HeuristicPointGoalPolicy,
    RandomPointGoalPolicy,
    ZeroPointGoalPolicy,
)
from unilab.evaluation.point_goal import (
    evaluate_point_goal_policies,
    format_point_goal_summary,
    generate_point_goal_manifest,
    read_point_goal_manifest,
    write_point_goal_manifest,
    write_point_goal_report,
)
from unilab.evaluation.point_goal_ppo import (
    build_point_goal_ppo_policy_factory,
    load_point_goal_ppo_config,
    ppo_algo_config_dict,
)
from unilab.evaluation.point_goal_sac import (
    build_point_goal_sac_policy_factory,
    load_point_goal_sac_config,
)
from unilab.training import BackendAdapter, create_env, ensure_registries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-policy-seed", type=int, default=2)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("zero", "random", "heuristic", "ppo", "sac"),
        default=("zero", "random", "heuristic"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sac-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-episode-seconds", type=float)
    parser.add_argument(
        "--record-trajectories",
        action="store_true",
        help="Include terminal-safe per-step robot states in the JSON report.",
    )
    parser.add_argument(
        "--manifest-input",
        type=Path,
        help="Reuse a standalone manifest or a manifest embedded in a prior report.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Write the exact reusable initial-condition manifest as strict JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/unilab_point_goal_evaluation.json"),
    )
    return parser.parse_args()


def _load_config() -> DictConfig:
    return load_point_goal_ppo_config(ROOT_DIR)


def _git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _algo_config_dict(cfg: DictConfig) -> dict[str, Any]:
    return ppo_algo_config_dict(cfg)


def _ppo_policy_factory(
    cfg: DictConfig,
    *,
    checkpoint: Path,
    device: str,
):
    return build_point_goal_ppo_policy_factory(
        cfg, checkpoint=checkpoint, device=device
    )


def main() -> None:
    args = _parse_args()
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.seed < 0 or args.random_policy_seed < 0:
        raise ValueError("evaluation seeds must be non-negative")
    if "ppo" in args.policies and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --policies includes ppo")
    if "sac" in args.policies and args.sac_checkpoint is None:
        raise ValueError("--sac-checkpoint is required when --policies includes sac")

    manifest = (
        read_point_goal_manifest(args.manifest_input) if args.manifest_input is not None else None
    )
    if manifest is not None and args.episodes is not None and args.episodes != manifest.episode_count:
        raise ValueError(
            f"--episodes={args.episodes} does not match manifest episode_count="
            f"{manifest.episode_count}"
        )
    episode_count = manifest.episode_count if manifest is not None else (args.episodes or 128)

    ensure_registries()
    cfg = _load_config()
    sac_cfg = load_point_goal_sac_config(ROOT_DIR) if "sac" in args.policies else None
    env_cfg_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="ppo"
    ).build_task_env_cfg_override()
    env_cfg_override["seed"] = manifest.seed if manifest is not None else args.seed
    if args.max_episode_seconds is not None:
        if args.max_episode_seconds <= 0.0:
            raise ValueError("--max-episode-seconds must be positive")
        env_cfg_override["max_episode_seconds"] = args.max_episode_seconds

    def env_factory():
        return create_env(
            cfg,
            num_envs=episode_count,
            env_cfg_override=env_cfg_override,
        )

    probe_env = env_factory()
    try:
        if manifest is None:
            manifest = generate_point_goal_manifest(
                episode_count=episode_count,
                seed=args.seed,
                min_goal_distance=probe_env.cfg.min_goal_distance,
                max_goal_distance=probe_env.cfg.max_goal_distance,
            )
        task_config = {
            "task": str(cfg.training.task_name),
            "sim_backend": str(cfg.training.sim_backend),
            "episodes": episode_count,
            "seed": manifest.seed,
            "random_policy_seed": args.random_policy_seed,
            "max_episode_steps": int(probe_env.cfg.max_episode_steps),
            "ctrl_dt": float(probe_env.cfg.ctrl_dt),
            "goal_tolerance": float(probe_env.cfg.goal_tolerance),
        }
    finally:
        probe_env.close()

    if args.manifest_output is not None:
        write_point_goal_manifest(manifest, args.manifest_output)

    policy_factories: dict[str, Any] = {}
    for policy_name in args.policies:
        if policy_name == "zero":
            policy_factories[policy_name] = lambda env: ZeroPointGoalPolicy()
        elif policy_name == "random":
            policy_factories[policy_name] = lambda env: RandomPointGoalPolicy(
                args.random_policy_seed
            )
        elif policy_name == "heuristic":
            policy_factories[policy_name] = lambda env: HeuristicPointGoalPolicy()
        elif policy_name == "ppo":
            assert args.checkpoint is not None
            policy_factories[policy_name] = _ppo_policy_factory(
                cfg,
                checkpoint=args.checkpoint.resolve(),
                device=args.device,
            )
        else:
            assert args.sac_checkpoint is not None
            assert sac_cfg is not None
            policy_factories[policy_name] = build_point_goal_sac_policy_factory(
                sac_cfg,
                checkpoint=args.sac_checkpoint.resolve(),
                device=args.device,
            )

    evaluation = evaluate_point_goal_policies(
        env_factory,
        policy_factories,
        manifest,
        record_trajectories=args.record_trajectories,
    )
    report = {
        "schema_version": 1,
        "git_sha": _git_sha(),
        "config": task_config,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoints": {
            "ppo": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "sac": (
                str(args.sac_checkpoint.resolve()) if args.sac_checkpoint else None
            ),
        },
        "trajectories_recorded": bool(args.record_trajectories),
        **evaluation,
    }
    for policy_name, result in report["policies"].items():
        result["checkpoint"] = report["checkpoints"].get(policy_name)

    print(format_point_goal_summary(report))
    output_path = write_point_goal_report(report, args.output)
    if args.manifest_output is not None:
        print(f"Manifest: {args.manifest_output}")
    print(f"JSON report: {output_path}")


if __name__ == "__main__":
    main()
