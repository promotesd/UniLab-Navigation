"""Tests for the navigation localization provider contract."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from unilab.base import registry
from unilab.envs.navigation import (
    AnalyticalWheelOdometryPlugin,
    DeadReckoningCfg,
    DeadReckoningPoseProvider,
    GroundTruthPosePacket,
    GroundTruthPoseProvider,
    LocalizationCfg,
    LocalizationPacket,
    NoisyPoseCfg,
    NoisyPoseProvider,
    OnlineEstimatorCfg,
    OnlineEstimatorPoseProvider,
    PoseEstimate,
    PoseStatus,
    RecordedPoseCfg,
    RecordedPoseProvider,
    RecordedPoseStream,
    WheelOdometryPacket,
    create_online_estimator_plugin,
    read_recorded_pose_stream,
    write_recorded_pose_stream,
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
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        del env_indices
        return self.update(packet)

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        pose = packet.ground_truth.pose.copy()
        pose[:, 1] += 1.0
        count = len(pose)
        return PoseEstimate(
            pose=pose,
            covariance=np.broadcast_to(np.diag([0.1, 0.1, 0.05]), (count, 3, 3)),
            valid=np.ones(count, dtype=bool),
            status=np.full(count, PoseStatus.DEGRADED, dtype=np.uint8),
            timestamp_s=packet.ground_truth.timestamp_s,
            parent_frame="map",
            child_frame="base_link",
        )


def packet(
    timestamp: np.ndarray,
    dt: np.ndarray,
    linear: np.ndarray,
    angular: np.ndarray,
    truth: np.ndarray | None,
    *,
    parent_frame: str = "map",
    child_frame: str = "base_link",
) -> LocalizationPacket:
    truth_pose = np.zeros((len(timestamp), 3), dtype=np.float32) if truth is None else truth
    return LocalizationPacket(
        ground_truth=GroundTruthPosePacket(
            timestamp_s=timestamp,
            pose=truth_pose,
            parent_frame=parent_frame,
            child_frame=child_frame,
        ),
        wheel_odometry=WheelOdometryPacket(
            timestamp_s=timestamp,
            dt_s=dt,
            linear_velocity=linear,
            angular_velocity=angular,
            frame_id=child_frame,
        ),
    )


def recorded_stream(
    environment_count: int = 2,
    *,
    status: np.ndarray | None = None,
) -> RecordedPoseStream:
    timestamps = np.array([0.0, 0.1, 0.2])
    pose = np.zeros((3, environment_count, 3), dtype=np.float32)
    pose[:, :, 0] = np.array([0.0, 1.0, 2.0])[:, None]
    pose[0, :, 2] = np.deg2rad(170.0)
    pose[1, :, 2] = np.deg2rad(-170.0)
    covariance = np.zeros((3, environment_count, 3, 3), dtype=np.float32)
    covariance[:, :, 0, 0] = np.array([0.1, 0.2, 0.3])[:, None]
    statuses = (
        np.full((3, environment_count), PoseStatus.TRACKING, dtype=np.uint8)
        if status is None
        else status
    )
    return RecordedPoseStream(
        timestamps_s=timestamps,
        pose=pose,
        covariance=covariance,
        status=statuses,
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


def test_typed_packet_batch_rejects_timestamp_and_frame_mismatch() -> None:
    truth = GroundTruthPosePacket(
        timestamp_s=np.array([0.0]),
        pose=np.zeros((1, 3)),
    )
    with pytest.raises(ValueError, match="timestamps must match"):
        LocalizationPacket(
            ground_truth=truth,
            wheel_odometry=WheelOdometryPacket(
                timestamp_s=np.array([0.1]),
                dt_s=np.array([0.1]),
                linear_velocity=np.zeros(1),
                angular_velocity=np.zeros(1),
            ),
        )
    with pytest.raises(ValueError, match="frames must match"):
        LocalizationPacket(
            ground_truth=truth,
            wheel_odometry=WheelOdometryPacket(
                timestamp_s=np.array([0.0]),
                dt_s=np.array([0.0]),
                linear_velocity=np.zeros(1),
                angular_velocity=np.zeros(1),
                frame_id="robot",
            ),
        )


def test_localization_config_rejects_ambiguous_dead_reckoning_frame() -> None:
    with pytest.raises(ValueError, match="odometry frame"):
        LocalizationCfg(provider="dead_reckoning", parent_frame="map").validate()
    with pytest.raises(ValueError, match="must be map"):
        LocalizationCfg(provider="noisy_pose", parent_frame="odom").validate()


def test_ground_truth_provider_copies_pose_and_validates_monotonic_time() -> None:
    provider = GroundTruthPoseProvider(parent_frame="odom", child_frame="robot")
    source = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, -0.5]], dtype=np.float32)
    reset_packet = packet(
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        source,
        parent_frame="odom",
        child_frame="robot",
    )
    estimate = provider.reset(reset_packet, np.array([0, 1], dtype=np.int32))
    source.fill(0.0)
    np.testing.assert_array_equal(estimate.pose, [[1.0, 2.0, 0.5], [3.0, 4.0, -0.5]])
    np.testing.assert_array_equal(estimate.covariance, np.zeros((2, 3, 3)))
    assert estimate.parent_frame == "odom"
    assert estimate.child_frame == "robot"
    provider.update(
        packet(
            np.array([0.2, 0.2]),
            np.full(2, 0.2),
            np.zeros(2),
            np.zeros(2),
            estimate.pose,
            parent_frame="odom",
            child_frame="robot",
        )
    )
    with pytest.raises(ValueError, match="monotonic"):
        provider.update(
            packet(
                np.array([0.1, 0.2]),
                np.zeros(2),
                np.zeros(2),
                np.zeros(2),
                estimate.pose,
                parent_frame="odom",
                child_frame="robot",
            )
        )


def test_dead_reckoning_integrates_packets_without_reading_update_truth() -> None:
    provider = DeadReckoningPoseProvider(DeadReckoningCfg())
    initial = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    provider.reset(
        packet(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), initial),
        np.array([0], dtype=np.int32),
    )
    estimate = provider.update(
        packet(
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([[100.0, 100.0, 2.0]], dtype=np.float32),
        )
    )
    np.testing.assert_allclose(estimate.pose, [[1.0, 0.0, 0.0]], atol=1.0e-6)
    estimate = provider.update(
        packet(
            np.array([2.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([np.pi / 2]),
            None,
        )
    )
    np.testing.assert_allclose(estimate.pose, [[1.0, 0.0, np.pi / 2]], atol=1.0e-6)
    assert estimate.parent_frame == "odom"


def test_noisy_pose_is_seeded_reports_covariance_and_tracks_new_truth() -> None:
    cfg = NoisyPoseCfg(
        seed=23,
        position_noise_std=0.1,
        heading_noise_std=0.05,
        x_bias=0.2,
        heading_bias=0.1,
    )
    providers = [NoisyPoseProvider(cfg), NoisyPoseProvider(cfg)]
    initial = np.zeros((3, 3), dtype=np.float32)
    reset_packet = packet(
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        initial,
    )
    estimates = [
        provider.reset(reset_packet, np.arange(3, dtype=np.int32))
        for provider in providers
    ]
    np.testing.assert_array_equal(estimates[0].pose, estimates[1].pose)
    np.testing.assert_allclose(estimates[0].covariance[:, 0, 0], 0.01)
    np.testing.assert_allclose(estimates[0].covariance[:, 2, 2], 0.0025)
    np.testing.assert_array_equal(
        estimates[0].status,
        np.full(3, PoseStatus.DEGRADED),
    )

    moved_truth = np.tile(np.array([[1.0, 2.0, 0.3]], dtype=np.float32), (3, 1))
    update_packet = packet(
        np.full(3, 0.1),
        np.full(3, 0.1),
        np.full(3, 99.0),
        np.full(3, 99.0),
        moved_truth,
    )
    updated = [provider.update(update_packet) for provider in providers]
    np.testing.assert_array_equal(updated[0].pose, updated[1].pose)
    assert np.all(updated[0].pose[:, 0] > 1.0)


def test_dead_reckoning_noise_is_seeded_and_covariance_grows() -> None:
    cfg = DeadReckoningCfg(
        seed=17,
        linear_velocity_noise_std=0.1,
        angular_velocity_noise_std=0.05,
        initial_position_variance=0.01,
        initial_heading_variance=0.02,
    )
    providers = [DeadReckoningPoseProvider(cfg), DeadReckoningPoseProvider(cfg)]
    initial = np.zeros((2, 3), dtype=np.float32)
    reset_packet = packet(
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        initial,
    )
    update_packet = packet(
        np.full(2, 0.5),
        np.full(2, 0.5),
        np.ones(2),
        np.zeros(2),
        None,
    )
    estimates = []
    for provider in providers:
        provider.reset(reset_packet, np.array([0, 1], dtype=np.int32))
        estimates.append(provider.update(update_packet))
    np.testing.assert_array_equal(estimates[0].pose, estimates[1].pose)
    np.testing.assert_array_equal(estimates[0].covariance, estimates[1].covariance)
    assert np.all(estimates[0].covariance[:, 0, 0] > 0.01)
    assert np.all(estimates[0].covariance[:, 2, 2] > 0.02)


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


def test_real_mujoco_dead_reckoning_drift_changes_only_localized_observation() -> None:
    cfg = DiffDrivePointGoalCfg()
    provider = DeadReckoningPoseProvider(
        DeadReckoningCfg(linear_velocity_bias=0.2)
    )
    env = DiffDrivePointGoalMujocoEnv(cfg=cfg, num_envs=1, pose_provider=provider)
    state = env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(state.info["localization_pose"], [[0.0, 0.0, 0.0]])
    next_state = env.step(np.array([[-1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        next_state.info["localization_pose"][0, 0],
        0.02,
        atol=1.0e-5,
    )
    physical_x = float(next_state.info["robot_state"][0, 0])
    assert abs(float(next_state.info["localization_pose"][0, 0]) - physical_x) > 0.019
    true_distance = np.linalg.norm(
        next_state.info["goal_position"] - next_state.info["robot_state"][:, :2],
        axis=1,
    )
    np.testing.assert_allclose(next_state.info["distance_to_goal"], true_distance)
    assert next_state.obs["obs"][0, 0] < state.obs["obs"][0, 0]
    np.testing.assert_array_equal(next_state.info["localization_parent_frame"], ["odom"])
    env.close()


def test_registry_selects_configured_dead_reckoning_provider() -> None:
    registry.ensure_registries()
    env = registry.make(
        "DiffDrivePointGoal",
        sim_backend="mujoco",
        env_cfg_override={
            "localization": {
                "provider": "dead_reckoning",
                "parent_frame": "odom",
                "dead_reckoning": {"linear_velocity_bias": 0.1, "seed": 31},
            }
        },
        num_envs=2,
    )
    assert isinstance(env.pose_provider, DeadReckoningPoseProvider)
    state = env.init_state()
    np.testing.assert_array_equal(
        state.info["localization_parent_frame"],
        ["odom", "odom"],
    )
    env.close()


def test_recorded_pose_stream_round_trips_and_rejects_tampering(tmp_path) -> None:
    source = recorded_stream()
    path = write_recorded_pose_stream(source, tmp_path / "recording.json")
    restored = read_recorded_pose_stream(path)
    np.testing.assert_array_equal(restored.timestamps_s, source.timestamps_s)
    np.testing.assert_array_equal(restored.pose, source.pose)
    np.testing.assert_array_equal(restored.covariance, source.covariance)
    np.testing.assert_array_equal(restored.status, source.status)

    payload = json.loads(path.read_text())
    payload["pose"][0][0][0] = 99.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha256"):
        read_recorded_pose_stream(path)


def test_recorded_pose_stream_rejects_duplicate_timestamps() -> None:
    source = recorded_stream()
    with pytest.raises(ValueError, match="strictly increasing"):
        RecordedPoseStream(
            timestamps_s=np.array([0.0, 0.1, 0.1]),
            pose=source.pose,
            covariance=source.covariance,
            status=source.status,
        )


def test_recorded_pose_provider_interpolates_by_timestamp_and_ignores_truth() -> None:
    source = recorded_stream(environment_count=1)
    source.status[1, 0] = PoseStatus.DEGRADED
    provider = RecordedPoseProvider(source, RecordedPoseCfg(interpolation="linear"))
    provider.reset(
        packet(
            np.array([0.0]),
            np.array([0.0]),
            np.zeros(1),
            np.zeros(1),
            np.array([[100.0, 100.0, 0.0]], dtype=np.float32),
        ),
        np.array([0], dtype=np.int32),
    )
    estimate = provider.update(
        packet(
            np.array([0.05]),
            np.array([0.05]),
            np.zeros(1),
            np.zeros(1),
            np.array([[-100.0, -100.0, 0.0]], dtype=np.float32),
        )
    )
    np.testing.assert_allclose(estimate.pose[0, 0], 0.5, atol=1.0e-6)
    assert abs(abs(float(estimate.pose[0, 2])) - np.pi) < 1.0e-5
    np.testing.assert_allclose(estimate.covariance[0, 0, 0], 0.15)
    assert estimate.status[0] == PoseStatus.DEGRADED
    np.testing.assert_allclose(estimate.timestamp_s, [0.05])


def test_recorded_pose_provider_defines_missing_stale_and_range_behavior() -> None:
    statuses = np.full((3, 1), PoseStatus.TRACKING, dtype=np.uint8)
    statuses[1, 0] = PoseStatus.LOST
    source = recorded_stream(environment_count=1, status=statuses)
    provider = RecordedPoseProvider(source, RecordedPoseCfg(interpolation="linear"))
    provider.reset(
        packet(np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1), None),
        np.array([0], dtype=np.int32),
    )
    missing = provider.update(
        packet(np.array([0.05]), np.full(1, 0.05), np.zeros(1), np.zeros(1), None)
    )
    assert missing.status[0] == PoseStatus.LOST
    assert not missing.valid[0]

    stale_provider = RecordedPoseProvider(
        recorded_stream(environment_count=1),
        RecordedPoseCfg(interpolation="previous", max_sample_age_s=0.04),
    )
    stale_provider.reset(
        packet(np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1), None),
        np.array([0], dtype=np.int32),
    )
    stale = stale_provider.update(
        packet(np.array([0.05]), np.full(1, 0.05), np.zeros(1), np.zeros(1), None)
    )
    assert stale.status[0] == PoseStatus.LOST
    np.testing.assert_allclose(stale.timestamp_s, [0.0])

    with pytest.raises(ValueError, match="out of range"):
        RecordedPoseProvider(source, RecordedPoseCfg(out_of_range="error")).reset(
            packet(np.array([0.3]), np.zeros(1), np.zeros(1), np.zeros(1), None),
            np.array([0], dtype=np.int32),
        )
    endpoint = RecordedPoseProvider(
        source,
        RecordedPoseCfg(out_of_range="error"),
    ).reset(
        packet(
            np.array([0.2 + 1.0e-13]),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            None,
        ),
        np.array([0], dtype=np.int32),
    )
    np.testing.assert_allclose(endpoint.pose[0], source.pose[-1, 0])
    clamped = RecordedPoseProvider(
        source,
        RecordedPoseCfg(out_of_range="clamp"),
    ).reset(
        packet(np.array([0.3]), np.zeros(1), np.zeros(1), np.zeros(1), None),
        np.array([0], dtype=np.int32),
    )
    np.testing.assert_allclose(clamped.pose[0], source.pose[-1, 0])
    lost = RecordedPoseProvider(source, RecordedPoseCfg(out_of_range="lost")).reset(
        packet(np.array([0.3]), np.zeros(1), np.zeros(1), np.zeros(1), None),
        np.array([0], dtype=np.int32),
    )
    assert lost.status[0] == PoseStatus.LOST


def test_recorded_pose_provider_replays_partial_reset_timestamps_independently() -> None:
    provider = RecordedPoseProvider(recorded_stream(), RecordedPoseCfg())
    provider.reset(
        packet(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), None),
        np.array([0, 1], dtype=np.int32),
    )
    provider.update(
        packet(
            np.full(2, 0.1),
            np.full(2, 0.1),
            np.zeros(2),
            np.zeros(2),
            None,
        )
    )
    reset = provider.reset(
        packet(
            np.array([0.0, 0.1]),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            None,
        ),
        np.array([0], dtype=np.int32),
    )
    np.testing.assert_allclose(reset.pose[:, 0], [0.0, 1.0])
    updated = provider.update(
        packet(
            np.array([0.05, 0.2]),
            np.array([0.05, 0.1]),
            np.zeros(2),
            np.zeros(2),
            None,
        )
    )
    np.testing.assert_allclose(updated.pose[:, 0], [0.5, 2.0])


def test_recorded_pose_config_requires_path_and_identity_transform() -> None:
    with pytest.raises(ValueError, match="requires recorded_pose.path"):
        LocalizationCfg(provider="recorded_pose").validate()
    with pytest.raises(ValueError, match="only identity"):
        RecordedPoseCfg(frame_transform="map_to_odom").validate()


def test_real_mujoco_recorded_pose_changes_observation_but_not_truth() -> None:
    source = RecordedPoseStream(
        timestamps_s=np.array([0.0, 0.1]),
        pose=np.array([[[0.0, 0.0, 0.0]], [[0.5, 0.0, 0.0]]]),
        covariance=np.zeros((2, 1, 3, 3)),
        status=np.full((2, 1), PoseStatus.TRACKING),
    )
    env = DiffDrivePointGoalMujocoEnv(
        cfg=DiffDrivePointGoalCfg(),
        num_envs=1,
        pose_provider=RecordedPoseProvider(source, RecordedPoseCfg()),
    )
    state = env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(state.obs["obs"][0, 0], 0.2)
    next_state = env.step(np.array([[-1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(next_state.info["localization_pose"], [[0.5, 0.0, 0.0]])
    np.testing.assert_allclose(next_state.obs["obs"][0, 0], 0.1, atol=1.0e-5)
    true_distance = np.linalg.norm(
        next_state.info["goal_position"] - next_state.info["robot_state"][:, :2],
        axis=1,
    )
    np.testing.assert_allclose(next_state.info["distance_to_goal"], true_distance)
    assert abs(float(next_state.info["robot_state"][0, 0]) - 0.5) > 0.49
    env.close()


def test_registry_loads_recorded_pose_stream_from_nested_config(tmp_path) -> None:
    path = write_recorded_pose_stream(recorded_stream(), tmp_path / "recording.json")
    registry.ensure_registries()
    env = registry.make(
        "DiffDrivePointGoal",
        sim_backend="mujoco",
        env_cfg_override={
            "localization": {
                "provider": "recorded_pose",
                "recorded_pose": {"path": str(path), "interpolation": "previous"},
            }
        },
        num_envs=2,
    )
    assert isinstance(env.pose_provider, RecordedPoseProvider)
    state = env.init_state()
    np.testing.assert_allclose(state.info["localization_pose"][:, 0], 0.0)
    env.close()


def test_online_estimator_applies_publish_cadence_after_processing_packets() -> None:
    cfg = OnlineEstimatorCfg(update_interval_steps=2)
    plugin = AnalyticalWheelOdometryPlugin(
        cfg,
        parent_frame="odom",
        child_frame="base_link",
    )
    provider = OnlineEstimatorPoseProvider(plugin, cfg)
    provider.reset(
        packet(np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1), None),
        np.array([0], dtype=np.int32),
    )
    first = provider.update(
        packet(np.array([1.0]), np.ones(1), np.ones(1), np.zeros(1), None)
    )
    np.testing.assert_allclose(first.pose, [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(first.timestamp_s, [0.0])
    second = provider.update(
        packet(np.array([2.0]), np.ones(1), np.ones(1), np.zeros(1), None)
    )
    np.testing.assert_allclose(second.pose, [[2.0, 0.0, 0.0]], atol=1.0e-6)
    np.testing.assert_allclose(second.timestamp_s, [2.0])


def test_online_estimator_applies_latency_dropout_and_staleness() -> None:
    latency_cfg = OnlineEstimatorCfg(latency_steps=1)
    latency_provider = OnlineEstimatorPoseProvider(
        AnalyticalWheelOdometryPlugin(
            latency_cfg,
            parent_frame="odom",
            child_frame="base_link",
        ),
        latency_cfg,
    )
    reset_packet = packet(
        np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1), None
    )
    latency_provider.reset(reset_packet, np.array([0], dtype=np.int32))
    delayed = latency_provider.update(
        packet(np.array([1.0]), np.ones(1), np.ones(1), np.zeros(1), None)
    )
    np.testing.assert_allclose(delayed.pose, [[0.0, 0.0, 0.0]])
    delayed = latency_provider.update(
        packet(np.array([2.0]), np.ones(1), np.ones(1), np.zeros(1), None)
    )
    np.testing.assert_allclose(delayed.pose, [[1.0, 0.0, 0.0]], atol=1.0e-6)

    unhealthy_cfg = OnlineEstimatorCfg(
        latency_steps=1,
        dropout_probability=1.0,
        max_staleness_s=0.5,
    )
    unhealthy = OnlineEstimatorPoseProvider(
        AnalyticalWheelOdometryPlugin(
            unhealthy_cfg,
            parent_frame="odom",
            child_frame="base_link",
        ),
        unhealthy_cfg,
    )
    initial = unhealthy.reset(reset_packet, np.array([0], dtype=np.int32))
    assert initial.status[0] == PoseStatus.LOST
    stale = unhealthy.update(
        packet(np.array([1.0]), np.ones(1), np.ones(1), np.zeros(1), None)
    )
    assert stale.status[0] == PoseStatus.LOST
    assert not stale.valid[0]


def test_online_estimator_dynamic_loading_and_close_lifecycle() -> None:
    cfg = OnlineEstimatorCfg()
    plugin = create_online_estimator_plugin(
        cfg,
        parent_frame="odom",
        child_frame="base_link",
    )
    assert isinstance(plugin, AnalyticalWheelOdometryPlugin)
    provider = OnlineEstimatorPoseProvider(plugin, cfg)
    provider.reset(
        packet(np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1), None),
        np.array([0], dtype=np.int32),
    )
    provider.close()
    assert plugin.closed
    provider.close()
    with pytest.raises(RuntimeError, match="closed"):
        provider.update(
            packet(np.array([0.1]), np.full(1, 0.1), np.zeros(1), np.zeros(1), None)
        )
    with pytest.raises(ValueError, match="cannot resolve"):
        create_online_estimator_plugin(
            OnlineEstimatorCfg(plugin="missing.module:factory"),
            parent_frame="odom",
            child_frame="base_link",
        )


def test_real_mujoco_online_estimator_honors_latency_and_closes_plugin() -> None:
    cfg = DiffDrivePointGoalCfg(
        localization=LocalizationCfg(
            provider="online_estimator",
            parent_frame="odom",
            online_estimator=OnlineEstimatorCfg(
                latency_steps=1,
                dead_reckoning=DeadReckoningCfg(linear_velocity_bias=0.2),
            ),
        )
    )
    env = DiffDrivePointGoalMujocoEnv(cfg=cfg, num_envs=1)
    assert isinstance(env.pose_provider, OnlineEstimatorPoseProvider)
    plugin = env.pose_provider.plugin
    assert isinstance(plugin, AnalyticalWheelOdometryPlugin)
    env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    first = env.step(np.array([[-1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(first.info["localization_pose"][0, 0], 0.0, atol=1.0e-6)
    second = env.step(np.array([[-1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(second.info["localization_pose"][0, 0], 0.02, atol=1.0e-5)
    env.close()
    assert plugin.closed


def test_registry_builds_online_estimator_from_nested_config() -> None:
    registry.ensure_registries()
    env = registry.make(
        "DiffDrivePointGoal",
        sim_backend="mujoco",
        env_cfg_override={
            "localization": {
                "provider": "online_estimator",
                "parent_frame": "odom",
                "online_estimator": {
                    "update_interval_steps": 2,
                    "latency_steps": 1,
                    "dropout_probability": 0.25,
                    "seed": 47,
                    "dead_reckoning": {"linear_velocity_bias": 0.1},
                },
            }
        },
        num_envs=2,
    )
    assert isinstance(env.pose_provider, OnlineEstimatorPoseProvider)
    assert env.pose_provider.cfg.update_interval_steps == 2
    assert env.pose_provider.cfg.dead_reckoning.linear_velocity_bias == 0.1
    plugin = env.pose_provider.plugin
    env.close()
    assert isinstance(plugin, AnalyticalWheelOdometryPlugin)
    assert plugin.closed
