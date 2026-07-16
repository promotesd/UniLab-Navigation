"""Validate and aggregate a fair fixed-budget PointGoal algorithm comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.evaluation.algorithm_comparison import (
    aggregate_point_goal_algorithm_runs,
    format_point_goal_algorithm_comparison,
    parse_algorithm_run_spec,
    write_point_goal_algorithm_comparison,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Repeat ALGORITHM:SEED:TRAINING_SUMMARY:EVALUATION_REPORT.",
    )
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--environment-step-budget", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/unilab_point_goal_algorithm_comparison.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = aggregate_point_goal_algorithm_runs(
        [parse_algorithm_run_spec(value) for value in args.run],
        expected_seeds=args.expected_seeds,
        environment_step_budget=args.environment_step_budget,
    )
    print(format_point_goal_algorithm_comparison(report))
    output = write_point_goal_algorithm_comparison(report, args.output)
    print(f"JSON report: {output}")


if __name__ == "__main__":
    main()
