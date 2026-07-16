"""Tests for the navigation localization provider contract."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from unilab.envs.navigation import (
    GroundTruthPoseProvider,
    PoseEstimate,
    PoseStatus,
)
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalEnv,
    DiffDrivePointGoalMujocoEnv,
    build_point_goal_observation,
)


class OffsetPoseProvider:
    """Test provider that exposes a biased estimate without changing truth."""

    def reset(
        self,
        source_pose: np.ndarray,
        timestamp_s: np.ndarray,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        del env_indices
        return self.update(source_pose, timestamp_s)

    def update(self, source_pose: np.ndarray, timestamp_s: np.ndarray) -> PoseEstimate:
        pose = np.asarray(source_pose).copy()
        pose[:, 1] += 1.0
        count = len(pose)
        return PoseEstimate(
            pose=pose,
            covariance=np.broadcast_to(np.diag([0.1, 0.1, 0.05]), (count, 3, 3)),
            valid=np.ones(count, dtype=bool),
            status=np.full(count, PoseStatus.DEGRADED, dtype=np.uint8),
            timestamp_s=timestamp_s,
            parent_frame="map",
            child_frame="base_link",
        )


def test_pose_estimate_rejects_inconsistent_validity_and_covariance() -> None:
    with pytest.raises(ValueError, match="valid must agree"):
        PoseEstimate(
            pose=np.zeros((1, 3)),
            covariance=np.zeros((1, 3, 3)),
            valid=np.array([False]),
            status=np.array([PoseStatus.TRACKING]),
            timestamp_s=np.array([0.0]),
        )
    covariance = np.zeros((1, 3, 3))
    covariance[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        PoseEstimate(
            pose=np.zeros((1, 3)),
            covariance=covariance,
            valid=np.array([True]),
            status=np.array([PoseStatus.TRACKING]),
            timestamp_s=np.array([0.0]),
        )


def test_ground_truth_provider_copies_pose_and_validates_monotonic_time() -> None:
    provider = GroundTruthPoseProvider(parent_frame="odom", child_frame="robot")
    source = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, -0.5]], dtype=np.float32)
    estimate = provider.reset(source, np.zeros(2), np.array([0, 1], dtype=np.int32))
    source.fill(0.0)
    np.testing.assert_array_equal(estimate.pose, [[1.0, 2.0, 0.5], [3.0, 4.0, -0.5]])
    np.testing.assert_array_equal(estimate.covariance, np.zeros((2, 3, 3)))
    assert estimate.parent_frame == "odom"
    assert estimate.child_frame == "robot"
    with pytest.raises(ValueError, match="monotonic"):
        provider.update(estimate.pose, np.array([0.1, -0.1]))


def test_point_goal_observation_uses_provider_while_metrics_keep_truth() -> None:
    backend = MagicMock()
    backend.step.return_value = None
    env = DiffDrivePointGoalEnv(
        cfg=DiffDrivePointGoalCfg(),
        backend=backend,
        num_envs=1,
        pose_provider=OffsetPoseProvider(),
    )
    state = env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    expected = build_point_goal_observation(
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        max_goal_distance=5.0,
    )
    np.testing.assert_allclose(state.obs["obs"][:, :3], expected, atol=1.0e-6)
    np.testing.assert_allclose(state.info["distance_to_goal"], [1.0])
    np.testing.assert_array_equal(state.info["localization_status"], [PoseStatus.DEGRADED])
    np.testing.assert_allclose(state.info["localization_covariance"][0], np.diag([0.1, 0.1, 0.05]))


def test_real_mujoco_ground_truth_provider_preserves_observation_and_frames() -> None:
    cfg = DiffDrivePointGoalCfg()
    env = DiffDrivePointGoalMujocoEnv(cfg=cfg, num_envs=2)
    states = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]], dtype=np.float32)
    goals = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    state = env.reset_to_initial_conditions(states, goals)
    expected = build_point_goal_observation(states, goals, cfg.max_goal_distance)
    np.testing.assert_allclose(state.obs["obs"][:, :3], expected, atol=1.0e-6)
    np.testing.assert_allclose(state.info["localization_pose"], states, atol=1.0e-6)
    np.testing.assert_array_equal(state.info["localization_valid"], [True, True])
    np.testing.assert_array_equal(state.info["localization_timestamp_s"], [0.0, 0.0])
    np.testing.assert_array_equal(state.info["localization_parent_frame"], ["map", "map"])
    next_state = env.step(np.array([[-1.0, 0.0], [-1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(next_state.info["localization_timestamp_s"], [0.1, 0.1])
    env.close()
