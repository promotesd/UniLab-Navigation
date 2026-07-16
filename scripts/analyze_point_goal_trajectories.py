"""Analyze a trajectory-enabled PointGoal report and render representative SVG paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.evaluation.point_goal import write_point_goal_report
from unilab.evaluation.trajectory_analysis import (
    analyze_point_goal_trajectories,
    write_point_goal_trajectory_svg,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/point_goal_trajectory_analysis.json")
    )
    parser.add_argument(
        "--svg", type=Path, default=Path("/tmp/point_goal_trajectory_analysis.svg")
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    analysis = analyze_point_goal_trajectories(report)
    write_point_goal_report(analysis, args.output)
    write_point_goal_trajectory_svg(report, analysis, args.svg)
    print(f"Analysis JSON: {args.output}")
    print(f"Trajectory SVG: {args.svg}")


if __name__ == "__main__":
    main()
