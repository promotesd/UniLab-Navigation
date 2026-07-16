"""Compute planar SLAM/localization metrics from a strict trace JSON file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unilab.evaluation import (
    evaluate_localization_trace,
    format_localization_metrics,
    read_localization_trace,
    write_localization_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/unilab_localization_metrics.json"),
    )
    args = parser.parse_args()
    report = evaluate_localization_trace(read_localization_trace(args.input))
    print(format_localization_metrics(report))
    output = write_localization_metrics(report, args.output)
    print(f"JSON report: {output}")


if __name__ == "__main__":
    main()
