"""Transport-free tests for ROS2-shaped navigation localization input."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from unilab.bridges.ros2 import (
    PlanarTransform,
    Ros2BufferedLocalizationPlugin,
    Ros2LocalizationBridge,
    Ros2LocalizationBridgeCfg,
    Ros2QoSContract,
)
from unilab.bridges.ros2.navigation import online_provider_cfg_for_ros2
from unilab.envs.navigation import OnlineEstimatorPoseProvider, PoseStatus
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
)


@dataclass
class Stamp:
    sec: int = 0
    nanosec: int = 0


@dataclass
class Header:
    stamp: Stamp = field(default_factory=Stamp)
    frame_id: str = "map"


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Pose:
    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)


@dataclass
class PoseWithCovariance:
    pose: Pose = field(default_factory=Pose)
    covariance: list[float] = field(default_factory=lambda: [0.0] * 36)


@dataclass
class Odometry:
    header: Header = field(default_factory=Header)
    child_frame_id: str = "base_link"
    pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)


def odometry(
    *,
    timestamp_s: float,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    parent_frame: str = "map",
) -> Odometry:
    sec = int(timestamp_s)
    nanosec = int(round((timestamp_s - sec) * 1.0e9))
    covariance = np.zeros((6, 6), dtype=np.float64)
    covariance[0, 0] = 0.1
    covariance[1, 1] = 0.2
    covariance[5, 5] = 0.3
    return Odometry(
        header=Header(stamp=Stamp(sec, nanosec), frame_id=parent_frame),
        pose=PoseWithCovariance(
            pose=Pose(
                position=Vector3(x, y, 0.0),
                orientation=Quaternion(z=np.sin(yaw / 2.0), w=np.cos(yaw / 2.0)),
            ),
            covariance=covariance.reshape(-1).tolist(),
        ),
    )


def test_ros2_odometry_converts_stamp_pose_covariance_and_frames() -> None:
    bridge = Ros2LocalizationBridge(Ros2LocalizationBridgeCfg(), 1)
    bridge.ingest_odometry(
        0,
        odometry(timestamp_s=1.25, x=2.0, y=-1.0, yaw=0.5),
        received_time_s=1.3,
    )
    estimate = bridge.estimate(np.array([1.4]))
    np.testing.assert_allclose(estimate.pose, [[2.0, -1.0, 0.5]], atol=1.0e-6)
    np.testing.assert_allclose(estimate.covariance[0], np.diag([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(estimate.timestamp_s, [1.25])
    assert estimate.parent_frame == "map"
    assert estimate.child_frame == "base_link"
    assert estimate.status[0] == PoseStatus.TRACKING


def test_ros2_bridge_requires_explicit_matching_tf_and_applies_it() -> None:
    bridge = Ros2LocalizationBridge(Ros2LocalizationBridgeCfg(), 1)
    message = odometry(timestamp_s=0.0, x=1.0, yaw=0.25, parent_frame="odom")
    with pytest.raises(ValueError, match="explicit TF"):
        bridge.ingest_odometry(0, message, received_time_s=0.0)
    bridge.ingest_odometry(
        0,
        message,
        received_time_s=0.0,
        transform=PlanarTransform(
            target_frame="map",
            source_frame="odom",
            translation_xy=np.array([2.0, 3.0]),
            yaw=np.pi / 2.0,
        ),
    )
    estimate = bridge.estimate(np.array([0.0]))
    np.testing.assert_allclose(estimate.pose[0, :2], [2.0, 4.0], atol=1.0e-6)
    np.testing.assert_allclose(estimate.pose[0, 2], 0.25 + np.pi / 2.0)
    np.testing.assert_allclose(np.diag(estimate.covariance[0]), [0.2, 0.1, 0.3])


def test_ros2_bridge_reports_dropout_and_stale_messages_as_lost() -> None:
    bridge = Ros2LocalizationBridge(
        Ros2LocalizationBridgeCfg(max_message_age_s=0.2),
        2,
    )
    missing = bridge.estimate(np.zeros(2))
    np.testing.assert_array_equal(missing.status, [PoseStatus.LOST, PoseStatus.LOST])
    bridge.ingest_odometry(0, odometry(timestamp_s=0.1), received_time_s=0.1)
    estimate = bridge.estimate(np.array([0.2, 0.2]))
    np.testing.assert_array_equal(estimate.valid, [True, False])
    stale = bridge.estimate(np.array([0.31, 0.31]))
    assert stale.status[0] == PoseStatus.LOST


def test_ros2_bridge_rejects_future_nonplanar_and_nonmonotonic_messages() -> None:
    bridge = Ros2LocalizationBridge(Ros2LocalizationBridgeCfg(), 1)
    with pytest.raises(ValueError, match="future"):
        bridge.ingest_odometry(
            0,
            odometry(timestamp_s=1.0),
            received_time_s=0.5,
        )
    nonplanar = odometry(timestamp_s=0.1)
    nonplanar.pose.pose.orientation = Quaternion(x=np.sin(0.1), w=np.cos(0.1))
    with pytest.raises(ValueError, match="not planar"):
        bridge.ingest_odometry(0, nonplanar, received_time_s=0.1)
    bridge.ingest_odometry(0, odometry(timestamp_s=0.2), received_time_s=0.2)
    with pytest.raises(ValueError, match="increase"):
        bridge.ingest_odometry(0, odometry(timestamp_s=0.2), received_time_s=0.2)


def test_ros2_qos_contract_is_explicit_and_validated() -> None:
    Ros2QoSContract(
        reliability="reliable",
        durability="transient_local",
        depth=1,
    ).validate()
    with pytest.raises(ValueError, match="reliability"):
        Ros2QoSContract(reliability="system_default").validate()
    with pytest.raises(ValueError, match="depth"):
        Ros2QoSContract(depth=0).validate()


def test_real_mujoco_consumes_buffered_ros2_pose_without_changing_truth() -> None:
    bridge = Ros2LocalizationBridge(Ros2LocalizationBridgeCfg(), 1)
    plugin = Ros2BufferedLocalizationPlugin(bridge)
    provider = OnlineEstimatorPoseProvider(plugin, online_provider_cfg_for_ros2())
    env = DiffDrivePointGoalMujocoEnv(
        cfg=DiffDrivePointGoalCfg(),
        num_envs=1,
        pose_provider=provider,
    )
    env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    bridge.ingest_odometry(
        0,
        odometry(timestamp_s=0.1, x=0.5),
        received_time_s=0.1,
    )
    state = env.step(np.array([[-1.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(state.info["localization_pose"], [[0.5, 0.0, 0.0]])
    np.testing.assert_allclose(state.obs["obs"][0, 0], 0.1, atol=1.0e-5)
    true_distance = np.linalg.norm(
        state.info["goal_position"] - state.info["robot_state"][:, :2],
        axis=1,
    )
    np.testing.assert_allclose(state.info["distance_to_goal"], true_distance)
    env.close()
    assert plugin.closed
