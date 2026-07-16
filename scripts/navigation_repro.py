"""One-command reproducibility workflows for UniLab Navigation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.reproducibility import (
    audit_navigation_release,
    build_navigation_provenance,
    write_navigation_provenance,
)

SEED_MANIFEST_PATH = ROOT_DIR / "conf" / "navigation" / "reproducibility.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/unilab_navigation_repro"),
    )
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    subparsers.add_parser("audit", help="Audit lock, files, ignore rules, and Git artifacts.")

    quick = subparsers.add_parser("quick", help="Run the bounded CI-style release workflow.")
    quick.add_argument("--skip-tests", action="store_true")

    train = subparsers.add_parser("train", help="Run one reference training command.")
    train.add_argument("algorithm", choices=("ppo", "sac", "td3"))
    train.add_argument("--profile", choices=("quick", "formal"), default="quick")
    train.add_argument("--seed", type=int, default=1)

    evaluate = subparsers.add_parser("evaluate", help="Run fixed-episode baseline evaluation.")
    evaluate.add_argument("--profile", choices=("quick", "formal"), default="quick")
    evaluate.add_argument(
        "--policies",
        nargs="+",
        choices=("zero", "random", "heuristic"),
        default=["zero", "random", "heuristic"],
    )

    benchmark = subparsers.add_parser(
        "benchmark", help="Run the framework-versus-standalone matrix."
    )
    benchmark.add_argument("--profile", choices=("quick", "formal"), default="quick")
    return parser.parse_args()


def _seed_manifest() -> dict[str, Any]:
    payload = json.loads(SEED_MANIFEST_PATH.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("navigation reproducibility seed schema must be version 1")
    return payload


def _run(command: Sequence[str], results: list[dict[str, Any]]) -> None:
    started = time.perf_counter()
    subprocess.run(list(command), cwd=ROOT_DIR, check=True)
    results.append(
        {
            "command": list(command),
            "wall_time_seconds": time.perf_counter() - started,
            "status": "passed",
        }
    )


def _write_audit(output: Path) -> Path:
    report = audit_navigation_release(ROOT_DIR)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "passed":
        raise RuntimeError("navigation release audit failed: " + "; ".join(report["errors"]))
    return output


def _provenance(
    *,
    workflow: str,
    invocation: Sequence[str],
    output_dir: Path,
    config: dict[str, Any],
    seeds: Sequence[int],
    outputs: Sequence[Path],
    command_results: Sequence[dict[str, Any]],
) -> None:
    payload = build_navigation_provenance(
        ROOT_DIR,
        workflow=workflow,
        command=invocation,
        config=config,
        seeds=seeds,
        input_files=(
            SEED_MANIFEST_PATH,
            ROOT_DIR / "pyproject.toml",
            ROOT_DIR / "uv.lock",
            ROOT_DIR / "src" / "unilab" / "assets" / "robots" / "diff_drive" / "scene.xml",
        ),
        output_files=outputs,
        command_results=command_results,
    )
    write_navigation_provenance(
        payload,
        output_dir / "provenance.json",
        output_dir / "provenance.csv",
    )


def _benchmark_command(profile: str, output: Path, manifest: dict[str, Any]) -> list[str]:
    settings = manifest[profile]
    if profile == "quick":
        warmups = 1
        repetitions = settings["benchmark_repetitions"]
    else:
        warmups = settings["benchmark_warmup_repetitions"]
        repetitions = settings["benchmark_repetitions"]
    return [
        sys.executable,
        "scripts/benchmark_point_goal_framework.py",
        "--environments",
        str(settings["benchmark_environments"]),
        "--rollout-steps",
        str(settings["benchmark_rollout_steps"]),
        "--warmup-repetitions",
        str(warmups),
        "--repetitions",
        str(repetitions),
        "--seeds",
        *(str(seed) for seed in settings["benchmark_seeds"]),
        "--hidden-dim",
        "64",
        "--chunk-size",
        "8",
        "--output",
        str(output),
    ]


def run_quick(args: argparse.Namespace, invocation: Sequence[str]) -> None:
    manifest = _seed_manifest()
    settings = manifest["quick"]
    output_dir = args.output_root / "quick"
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if not args.skip_tests:
        _run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "src/unilab/benchmarks",
                "src/unilab/bridges",
                "src/unilab/envs/navigation",
                "src/unilab/evaluation",
                "scripts",
                "tests/envs/navigation",
            ],
            results,
        )
        _run(
            [sys.executable, "-m", "pytest", "tests/envs/navigation", "-q"],
            results,
        )
    evaluation = output_dir / "evaluation.json"
    initial_conditions = output_dir / "manifest.json"
    _run(
        [
            sys.executable,
            "scripts/evaluate_point_goal.py",
            "--episodes",
            str(settings["evaluation_episodes"]),
            "--seed",
            str(settings["evaluation_seed"]),
            "--policies",
            "zero",
            "random",
            "heuristic",
            "--manifest-output",
            str(initial_conditions),
            "--output",
            str(evaluation),
        ],
        results,
    )
    benchmark = output_dir / "benchmark.json"
    _run(_benchmark_command("quick", benchmark, manifest), results)
    audit = _write_audit(output_dir / "audit.json")
    _provenance(
        workflow="quick",
        invocation=invocation,
        output_dir=output_dir,
        config=settings,
        seeds=[settings["evaluation_seed"], *settings["benchmark_seeds"]],
        outputs=(evaluation, initial_conditions, benchmark, audit),
        command_results=results,
    )
    print(f"Quick workflow: {output_dir}")


def run_train(args: argparse.Namespace, invocation: Sequence[str]) -> None:
    if args.seed < 0:
        raise ValueError("training seed must be non-negative")
    output_dir = args.output_root / f"train_{args.algorithm}_{args.profile}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.algorithm == "ppo":
        environments = 1024 if args.profile == "formal" else 128
        steps_per_env = 24 if args.profile == "formal" else 8
        iterations = 4 if args.profile == "formal" else 1
        command = [
            sys.executable,
            "scripts/train_rsl_rl.py",
            "task=diff_drive_point_goal/mujoco",
            f"algo.seed={args.seed}",
            f"algo.num_envs={environments}",
            f"algo.num_steps_per_env={steps_per_env}",
            f"algo.max_iterations={iterations}",
            "algo.save_interval=1",
            "training.device=cuda",
            "training.logger=tensorboard",
            "training.no_play=true",
            f"training.log_root={output_dir}",
        ]
    else:
        environments = 1024 if args.profile == "formal" else 128
        iterations = 93 if args.profile == "formal" else 1
        command = [
            sys.executable,
            "scripts/train_offpolicy.py",
            f"algo={args.algorithm}",
            f"task={args.algorithm}/diff_drive_point_goal/mujoco",
            f"algo.seed={args.seed}",
            f"algo.num_envs={environments}",
            f"algo.max_iterations={iterations}",
            f"algo.save_interval={iterations}",
            "training.device=cuda",
            "training.use_amp=false",
            "training.logger=tensorboard",
            "training.no_play=true",
            f"training.log_dir={output_dir}",
        ]
    results: list[dict[str, Any]] = []
    _run(command, results)
    generated = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"provenance.json", "provenance.csv"}
    )
    _provenance(
        workflow="train",
        invocation=invocation,
        output_dir=output_dir,
        config={"algorithm": args.algorithm, "profile": args.profile},
        seeds=[args.seed],
        outputs=generated,
        command_results=results,
    )
    print(f"Training workflow: {output_dir}")


def run_evaluate(args: argparse.Namespace, invocation: Sequence[str]) -> None:
    manifest = _seed_manifest()
    settings = manifest[args.profile]
    episodes = settings["evaluation_episodes"]
    seed = settings.get("evaluation_manifest_seed", settings.get("evaluation_seed"))
    output_dir = args.output_root / f"evaluate_{args.profile}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "evaluation.json"
    initial_conditions = output_dir / "manifest.json"
    command = [
        sys.executable,
        "scripts/evaluate_point_goal.py",
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--policies",
        *args.policies,
        "--manifest-output",
        str(initial_conditions),
        "--output",
        str(report),
    ]
    results: list[dict[str, Any]] = []
    _run(command, results)
    _provenance(
        workflow="evaluate",
        invocation=invocation,
        output_dir=output_dir,
        config={"profile": args.profile, "episodes": episodes, "policies": args.policies},
        seeds=[seed],
        outputs=(report, initial_conditions),
        command_results=results,
    )
    print(f"Evaluation workflow: {output_dir}")


def run_benchmark(args: argparse.Namespace, invocation: Sequence[str]) -> None:
    manifest = _seed_manifest()
    settings = manifest[args.profile]
    output_dir = args.output_root / f"benchmark_{args.profile}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / "benchmark.json"
    command = _benchmark_command(args.profile, report, manifest)
    results: list[dict[str, Any]] = []
    _run(command, results)
    _provenance(
        workflow="benchmark",
        invocation=invocation,
        output_dir=output_dir,
        config=settings,
        seeds=settings["benchmark_seeds"],
        outputs=(report,),
        command_results=results,
    )
    print(f"Benchmark workflow: {output_dir}")


def main() -> None:
    args = _parse_args()
    invocation = [sys.executable, *sys.argv]
    if args.workflow == "audit":
        output = args.output_root / "audit.json"
        _write_audit(output)
        print(f"Audit: {output}")
    elif args.workflow == "quick":
        run_quick(args, invocation)
    elif args.workflow == "train":
        run_train(args, invocation)
    elif args.workflow == "evaluate":
        run_evaluate(args, invocation)
    else:
        run_benchmark(args, invocation)


if __name__ == "__main__":
    main()
