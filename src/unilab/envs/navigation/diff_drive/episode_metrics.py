"""Episode-level metrics for PointGoal navigation."""

from __future__ import annotations

import numpy as np

from unilab.dtype_config import get_global_dtype


def build_point_goal_episode_log(
    *,
    initial_distance: np.ndarray,
    final_distance: np.ndarray,
    reached_goal: np.ndarray,
    episode_steps: np.ndarray,
    max_episode_steps: int,
) -> dict[str, np.ndarray] | None:
    """Build metrics only for episodes completed on the current step.

    An episode is completed when either:

    1. The robot reaches the goal.
    2. The episode reaches its time limit.

    Returns:
        A dictionary containing one value per completed episode.
        Returns ``None`` when no episode completed on this step.
    """
    initial_array = np.asarray(initial_distance)
    final_array = np.asarray(final_distance)
    success_array = np.asarray(
        reached_goal,
        dtype=bool,
    )
    steps_array = np.asarray(episode_steps)

    if initial_array.ndim != 1:
        raise ValueError(
            "initial_distance must be one-dimensional"
        )

    expected_shape = initial_array.shape

    named_arrays = {
        "final_distance": final_array,
        "reached_goal": success_array,
        "episode_steps": steps_array,
    }

    for name, array in named_arrays.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, "
                f"got {array.shape}"
            )

    if max_episode_steps <= 0:
        raise ValueError(
            "max_episode_steps must be positive"
        )

    # A success on the final allowed step is still counted as success,
    # not as a timeout failure.
    timed_out = (
        steps_array >= max_episode_steps
    ) & ~success_array

    completed = success_array | timed_out

    if not np.any(completed):
        return None

    dtype = get_global_dtype()

    completed_initial = initial_array[
        completed
    ].astype(dtype, copy=False)

    completed_final = final_array[
        completed
    ].astype(dtype, copy=False)

    completed_success = success_array[
        completed
    ].astype(dtype)

    completed_timeout = timed_out[
        completed
    ].astype(dtype)

    completed_steps = steps_array[
        completed
    ].astype(dtype)

    safe_initial = np.maximum(
        completed_initial,
        np.finfo(dtype).eps,
    )

    progress_ratio = (
        completed_initial - completed_final
    ) / safe_initial

    return {
        "Navigation/success_rate": completed_success,
        "Navigation/timeout_rate": completed_timeout,
        "Navigation/initial_distance": completed_initial,
        "Navigation/final_distance": completed_final,
        "Navigation/progress_ratio": progress_ratio,
        "Navigation/episode_length": completed_steps,
    }
