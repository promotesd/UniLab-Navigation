"""Deterministic PointGoal trajectory analysis and SVG visualization."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _representative_episode(
    episodes: list[Mapping[str, Any]],
    *,
    outcome: str,
) -> dict[str, Any] | None:
    matching = [episode for episode in episodes if bool(episode["success"]) == (outcome == "success")]
    if not matching:
        return None
    metric = "spl" if outcome == "success" else "progress_ratio"
    ordered = sorted(matching, key=lambda episode: (float(episode[metric]), int(episode["episode_id"])))
    selected = ordered[(len(ordered) - 1) // 2]
    return {
        "outcome": outcome,
        "episode_id": int(selected["episode_id"]),
        "selection_metric": metric,
        "selection_rule": "lower median over all matching episodes",
        "selection_value": float(selected[metric]),
    }


def analyze_point_goal_trajectories(report: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize all trajectories and choose reproducible representatives."""
    policies = report.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise ValueError("trajectory report must contain policy results")
    summaries: dict[str, Any] = {}
    selections: list[dict[str, Any]] = []
    for policy_name, result in policies.items():
        episodes = result.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"policy {policy_name!r} has no episodes")
        if any("trajectory" not in episode for episode in episodes):
            raise ValueError(f"policy {policy_name!r} report does not contain trajectories")
        success_count = sum(bool(episode["success"]) for episode in episodes)
        collision_count = sum(bool(episode.get("collision", False)) for episode in episodes)
        timeout_count = sum(bool(episode["timeout"]) for episode in episodes)
        summaries[policy_name] = {
            "episode_count": len(episodes),
            "success_count": success_count,
            "collision_count": collision_count,
            "timeout_count": timeout_count,
            "failure_count": len(episodes) - success_count,
            "path_length": result["metrics"]["path_length"],
            "spl": result["metrics"]["spl"],
        }
        for outcome in ("success", "failure"):
            selected = _representative_episode(episodes, outcome=outcome)
            if selected is not None:
                selections.append({"policy": policy_name, **selected})
    return {
        "selection_policy": (
            "Representatives are deterministic lower medians, using SPL for successes "
            "and progress ratio for failures; all episodes remain in the source report."
        ),
        "policies": summaries,
        "representatives": selections,
    }


def _episode_lookup(report: Mapping[str, Any], policy: str, episode_id: int) -> Mapping[str, Any]:
    episodes = report["policies"][policy]["episodes"]
    return next(episode for episode in episodes if int(episode["episode_id"]) == episode_id)


def write_point_goal_trajectory_svg(
    report: Mapping[str, Any],
    analysis: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Render selected trajectories as a dependency-free SVG report."""
    representatives = analysis.get("representatives")
    if not isinstance(representatives, list) or not representatives:
        raise ValueError("trajectory analysis has no representatives to visualize")
    panel_width = 520
    panel_height = 360
    margin = 50
    width = panel_width
    height = panel_height * len(representatives)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]
    manifest_conditions = report["manifest"]["initial_conditions"]
    for panel_index, representative in enumerate(representatives):
        policy = str(representative["policy"])
        episode_id = int(representative["episode_id"])
        episode = _episode_lookup(report, policy, episode_id)
        trajectory = episode["trajectory"]
        positions = np.asarray([step["robot_state"][:2] for step in trajectory], dtype=float)
        goal = np.asarray(manifest_conditions[episode_id]["goal_position"], dtype=float)
        points = np.vstack([positions, goal[None, :]])
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        span = np.maximum(upper - lower, 0.5)
        scale = min(
            (panel_width - 2 * margin) / span[0],
            (panel_height - 2 * margin - 30) / span[1],
        )
        origin_y = panel_index * panel_height

        def project(point: np.ndarray) -> tuple[float, float]:
            x = margin + (point[0] - lower[0]) * scale
            y = origin_y + panel_height - margin - (point[1] - lower[1]) * scale
            return float(x), float(y)

        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, positions))
        start_x, start_y = project(positions[0])
        goal_x, goal_y = project(goal)
        title = html.escape(
            f"{policy} | {representative['outcome']} | episode {episode_id} | "
            f"path={episode['path_length']:.3f} m | SPL={episode['spl']:.3f}"
        )
        elements.extend(
            [
                f'<text x="20" y="{origin_y + 24}" font-family="sans-serif" '
                f'font-size="15" fill="#0f172a">{title}</text>',
                f'<polyline points="{polyline}" fill="none" stroke="#2563eb" '
                'stroke-width="3" stroke-linejoin="round"/>',
                f'<circle cx="{start_x:.2f}" cy="{start_y:.2f}" r="6" fill="#16a34a"/>',
                f'<circle cx="{goal_x:.2f}" cy="{goal_y:.2f}" r="8" fill="#dc2626"/>',
                f'<line x1="20" y1="{origin_y + panel_height - 1}" x2="500" '
                f'y2="{origin_y + panel_height - 1}" stroke="#cbd5e1"/>',
            ]
        )
    elements.append("</svg>")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n")
    return path
