"""Unit tests for fixed-episode PointGoal policies."""

import numpy as np

from unilab.envs.navigation.diff_drive.controllers import (
    HeuristicPointGoalPolicy,
    RandomPointGoalPolicy,
    ZeroPointGoalPolicy,
)
from unilab.evaluation.point_goal import PpoPointGoalPolicy


def observations(bearings: np.ndarray) -> dict[str, np.ndarray]:
    actor = np.zeros((len(bearings), 5), dtype=np.float32)
    actor[:, 0] = 0.5
    actor[:, 1] = np.sin(bearings)
    actor[:, 2] = np.cos(bearings)
    return {"obs": actor, "critic": actor.copy()}


def test_zero_policy_emits_physical_stationary_action() -> None:
    actions = ZeroPointGoalPolicy()(observations(np.array([0.0, 1.0])))
    np.testing.assert_array_equal(actions, np.array([[-1.0, 0.0], [-1.0, 0.0]]))


def test_random_policy_is_bounded_and_reproducible() -> None:
    obs = observations(np.zeros(4))
    first = RandomPointGoalPolicy(seed=9)(obs)
    second = RandomPointGoalPolicy(seed=9)(obs)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)


def test_heuristic_policy_drives_straight_and_rotates_in_place() -> None:
    actions = HeuristicPointGoalPolicy()(observations(np.array([0.0, np.pi])))
    np.testing.assert_allclose(actions[0], np.array([1.0, 0.0]), atol=1.0e-6)
    assert actions[1, 0] == -1.0
    assert abs(actions[1, 1]) == 1.0


def test_ppo_policy_adapts_tensor_like_output_to_numpy() -> None:
    actor = lambda adapted: adapted[:, :2]  # noqa: E731
    policy = PpoPointGoalPolicy(actor, lambda obs: obs["obs"])
    actions = policy(observations(np.array([0.0, 0.5])))
    assert actions.shape == (2, 2)
    assert actions.dtype == np.float32
