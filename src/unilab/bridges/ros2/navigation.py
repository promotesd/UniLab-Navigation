"""Dependency-free ROS2 odometry conversion and buffering for navigation."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.navigation.localization import (
    LocalizationPacket,
    OnlineEstimatorCfg,
    PoseEstimate,
    PoseStatus,
)


@dataclass(frozen=True)
class Ros2QoSContract:
    """Transport policy mirrored by a future rclpy adapter."""

    reliability: str = "best_effort"
    durability: str = "volatile"
    history: str = "keep_last"
    depth: int = 5

    def validate(self) -> None:
        if self.reliability not in {"best_effort", "reliable"}:
            raise ValueError("ROS2 reliability must be best_effort or reliable")
        if self.durability not in {"volatile", "transient_local"}:
            raise ValueError("ROS2 durability must be volatile or transient_local")
        if self.history != "keep_last":
            raise ValueError("ROS2 navigation bridge currently requires keep_last history")
        if self.depth <= 0:
            raise ValueError("ROS2 QoS depth must be positive")


@dataclass(frozen=True)
class PlanarTransform:
    """Explicit target-from-source planar transform supplied by a TF adapter."""

    target_frame: str
    source_frame: str
    translation_xy: np.ndarray
    yaw: float

    def __post_init__(self) -> None:
        translation = np.asarray(self.translation_xy, dtype=get_global_dtype())
        if not self.target_frame or not self.source_frame:
            raise ValueError("TF frame names must be non-empty")
        if self.target_frame == self.source_frame:
            raise ValueError("TF target and source frames must differ")
        if translation.shape != (2,) or not np.all(np.isfinite(translation)):
            raise ValueError("TF translation_xy must be a finite two-vector")
        if not np.isfinite(self.yaw):
            raise ValueError("TF yaw must be finite")
        object.__setattr__(self, "translation_xy", translation.copy())


@dataclass
class Ros2LocalizationBridgeCfg:
    """Frames, timing, planarity, and QoS for ROS2 localization input."""

    parent_frame: str = "map"
    child_frame: str = "base_link"
    max_message_age_s: float = 0.5
    future_tolerance_s: float = 1.0e-6
    planar_tolerance_rad: float = 1.0e-4
    qos: Ros2QoSContract = field(default_factory=Ros2QoSContract)

    def validate(self) -> None:
        if not self.parent_frame or not self.child_frame:
            raise ValueError("ROS2 localization frames must be non-empty")
        if self.parent_frame == self.child_frame:
            raise ValueError("ROS2 parent and child frames must differ")
        for name, value in (
            ("max_message_age_s", self.max_message_age_s),
            ("future_tolerance_s", self.future_tolerance_s),
            ("planar_tolerance_rad", self.planar_tolerance_rad),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"ROS2 {name} must be finite and non-negative")
        self.qos.validate()


def _stamp_seconds(stamp: Any) -> float:
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError("ROS2 header stamp is invalid")
    return sec + nanosec * 1.0e-9


def _planar_yaw(orientation: Any, tolerance: float) -> float:
    quaternion = np.array(
        [orientation.w, orientation.x, orientation.y, orientation.z],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("ROS2 orientation quaternion must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("ROS2 orientation quaternion must be non-zero")
    quaternion /= norm
    w, x, y, z = quaternion
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    if abs(float(roll)) > tolerance or abs(float(pitch)) > tolerance:
        raise ValueError("ROS2 localization orientation is not planar")
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _planar_covariance(covariance: Any) -> np.ndarray:
    matrix = np.asarray(covariance, dtype=get_global_dtype())
    if matrix.size != 36:
        raise ValueError("ROS2 pose covariance must contain 36 values")
    matrix = matrix.reshape(6, 6)
    planar = matrix[np.ix_([0, 1, 5], [0, 1, 5])]
    if not np.all(np.isfinite(planar)):
        raise ValueError("ROS2 planar covariance must be finite")
    return planar


def _apply_transform(
    pose: np.ndarray,
    covariance: np.ndarray,
    transform: PlanarTransform,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.cos(transform.yaw)
    sine = np.sin(transform.yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=get_global_dtype())
    transformed_pose = pose.copy()
    transformed_pose[:2] = rotation @ pose[:2] + transform.translation_xy
    transformed_pose[2] = (pose[2] + transform.yaw + np.pi) % (2.0 * np.pi) - np.pi
    jacobian = np.eye(3, dtype=get_global_dtype())
    jacobian[:2, :2] = rotation
    return transformed_pose, jacobian @ covariance @ jacobian.T


class Ros2LocalizationBridge:
    """Thread-safe latest-message buffer with strict ROS2-shaped validation."""

    def __init__(self, cfg: Ros2LocalizationBridgeCfg, environment_count: int) -> None:
        cfg.validate()
        if environment_count <= 0:
            raise ValueError("ROS2 bridge environment_count must be positive")
        self.cfg = cfg
        self.environment_count = environment_count
        dtype = get_global_dtype()
        self._pose = np.zeros((environment_count, 3), dtype=dtype)
        self._covariance = np.zeros((environment_count, 3, 3), dtype=dtype)
        self._timestamp_s = np.zeros(environment_count, dtype=np.float64)
        self._received = np.zeros(environment_count, dtype=bool)
        self._lock = threading.Lock()

    def ingest_odometry(
        self,
        env_index: int,
        message: Any,
        *,
        received_time_s: float,
        transform: PlanarTransform | None = None,
    ) -> None:
        """Validate and buffer one nav_msgs/Odometry-shaped object."""
        if not 0 <= env_index < self.environment_count:
            raise IndexError("ROS2 bridge environment index is out of range")
        if not np.isfinite(received_time_s) or received_time_s < 0.0:
            raise ValueError("ROS2 message receipt time must be finite and non-negative")
        timestamp_s = _stamp_seconds(message.header.stamp)
        if timestamp_s > received_time_s + self.cfg.future_tolerance_s:
            raise ValueError("ROS2 message stamp is in the future")
        source_parent = str(message.header.frame_id)
        child_frame = str(message.child_frame_id)
        if child_frame != self.cfg.child_frame:
            raise ValueError("ROS2 odometry child frame does not match bridge config")
        pose_message = message.pose.pose
        position = pose_message.position
        pose = np.array(
            [position.x, position.y, _planar_yaw(pose_message.orientation, self.cfg.planar_tolerance_rad)],
            dtype=get_global_dtype(),
        )
        if not np.all(np.isfinite(pose)):
            raise ValueError("ROS2 planar pose must be finite")
        covariance = _planar_covariance(message.pose.covariance)
        if source_parent != self.cfg.parent_frame:
            if transform is None:
                raise ValueError("ROS2 parent frame requires an explicit TF transform")
            if (
                transform.source_frame != source_parent
                or transform.target_frame != self.cfg.parent_frame
            ):
                raise ValueError("ROS2 TF transform frames do not match the message")
            pose, covariance = _apply_transform(pose, covariance, transform)
        elif transform is not None:
            raise ValueError("ROS2 TF transform is ambiguous for an already matching frame")
        PoseEstimate(
            pose=pose[None, :],
            covariance=covariance[None, :, :],
            valid=np.ones(1, dtype=bool),
            status=np.full(1, PoseStatus.TRACKING, dtype=np.uint8),
            timestamp_s=np.array([timestamp_s]),
            parent_frame=self.cfg.parent_frame,
            child_frame=self.cfg.child_frame,
        )
        with self._lock:
            if self._received[env_index] and timestamp_s <= self._timestamp_s[env_index]:
                raise ValueError("ROS2 message stamps must increase per environment")
            self._pose[env_index] = pose
            self._covariance[env_index] = covariance
            self._timestamp_s[env_index] = timestamp_s
            self._received[env_index] = True

    def reset(self, env_indices: np.ndarray) -> None:
        indices = np.asarray(env_indices, dtype=np.int32)
        with self._lock:
            self._pose[indices] = 0.0
            self._covariance[indices] = 0.0
            self._timestamp_s[indices] = 0.0
            self._received[indices] = False

    def estimate(self, query_timestamp_s: np.ndarray) -> PoseEstimate:
        query = np.asarray(query_timestamp_s, dtype=np.float64)
        if query.shape != (self.environment_count,):
            raise ValueError("ROS2 query timestamps must match environment count")
        if not np.all(np.isfinite(query)) or np.any(query < 0.0):
            raise ValueError("ROS2 query timestamps must be finite and non-negative")
        with self._lock:
            pose = self._pose.copy()
            covariance = self._covariance.copy()
            timestamp = self._timestamp_s.copy()
            received = self._received.copy()
        future = received & (timestamp > query + self.cfg.future_tolerance_s)
        if np.any(future):
            raise ValueError("ROS2 buffered message is newer than the query time")
        age = query - timestamp
        tracking = received & (age <= self.cfg.max_message_age_s)
        status = np.full(self.environment_count, PoseStatus.LOST, dtype=np.uint8)
        status[tracking] = PoseStatus.TRACKING
        return PoseEstimate(
            pose=pose,
            covariance=covariance,
            valid=tracking,
            status=status,
            timestamp_s=timestamp,
            parent_frame=self.cfg.parent_frame,
            child_frame=self.cfg.child_frame,
        )


class Ros2BufferedLocalizationPlugin:
    """Transport-free online plugin backed by a ROS2 localization buffer."""

    def __init__(self, bridge: Ros2LocalizationBridge) -> None:
        self.bridge = bridge
        self.parent_frame = bridge.cfg.parent_frame
        self.child_frame = bridge.cfg.child_frame
        self.closed = False

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        if self.closed:
            raise RuntimeError("ROS2 localization plugin is closed")
        self.bridge.reset(env_indices)
        return self.bridge.estimate(packet.timestamp_s)

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        if self.closed:
            raise RuntimeError("ROS2 localization plugin is closed")
        return self.bridge.estimate(packet.timestamp_s)

    def close(self) -> None:
        self.closed = True


def online_provider_cfg_for_ros2() -> OnlineEstimatorCfg:
    """Return neutral scheduling defaults for an injected ROS2 plugin."""
    return OnlineEstimatorCfg(
        update_interval_steps=1,
        latency_steps=0,
        dropout_probability=0.0,
        max_staleness_s=None,
    )
