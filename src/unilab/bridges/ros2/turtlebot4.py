"""Transport-optional TurtleBot 4 navigation command, scan, and episode bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import numpy as np

from unilab.dtype_config import get_global_dtype

from .navigation import Ros2QoSContract, _stamp_seconds


@dataclass(frozen=True)
class TurtleBot4Profile:
    """Explicit simulation or real-robot topic, frame, and safety profile."""

    mode: str
    namespace: str
    command_topic: str
    odometry_topic: str
    scan_topic: str
    parent_frame: str
    child_frame: str
    scan_frame: str
    max_linear_velocity: float
    max_angular_velocity: float
    command_timeout_s: float
    scan_timeout_s: float
    reset_service: str | None
    command_qos: Ros2QoSContract
    localization_qos: Ros2QoSContract
    scan_qos: Ros2QoSContract

    def validate(self) -> None:
        if self.mode not in {"simulation", "real_robot"}:
            raise ValueError("TurtleBot 4 mode must be simulation or real_robot")
        for name in (
            "command_topic",
            "odometry_topic",
            "scan_topic",
            "parent_frame",
            "child_frame",
            "scan_frame",
        ):
            if not getattr(self, name):
                raise ValueError(f"TurtleBot 4 {name} must be non-empty")
        if self.parent_frame == self.child_frame:
            raise ValueError("TurtleBot 4 parent and child frames must differ")
        for name in (
            "max_linear_velocity",
            "max_angular_velocity",
            "command_timeout_s",
            "scan_timeout_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"TurtleBot 4 {name} must be finite and positive")
        if self.mode == "simulation" and not self.reset_service:
            raise ValueError("TurtleBot 4 simulation profile requires a reset service")
        if self.mode == "real_robot" and self.reset_service is not None:
            raise ValueError("TurtleBot 4 real-robot reset must require operator control")
        self.command_qos.validate()
        self.localization_qos.validate()
        self.scan_qos.validate()


def turtlebot4_simulation_profile(namespace: str = "") -> TurtleBot4Profile:
    """Return conservative project defaults for simulator integration."""
    prefix = f"/{namespace.strip('/')}" if namespace.strip("/") else ""
    profile = TurtleBot4Profile(
        mode="simulation",
        namespace=namespace,
        command_topic=f"{prefix}/cmd_vel",
        odometry_topic=f"{prefix}/odom",
        scan_topic=f"{prefix}/scan",
        parent_frame="odom",
        child_frame="base_link",
        scan_frame="rplidar_link",
        max_linear_velocity=0.5,
        max_angular_velocity=1.5,
        command_timeout_s=0.25,
        scan_timeout_s=0.5,
        reset_service=f"{prefix}/reset_episode",
        command_qos=Ros2QoSContract(reliability="reliable", depth=1),
        localization_qos=Ros2QoSContract(reliability="reliable", depth=10),
        scan_qos=Ros2QoSContract(reliability="best_effort", depth=5),
    )
    profile.validate()
    return profile


def turtlebot4_real_robot_profile(namespace: str = "") -> TurtleBot4Profile:
    """Return a separately validated conservative real-robot profile."""
    prefix = f"/{namespace.strip('/')}" if namespace.strip("/") else ""
    profile = TurtleBot4Profile(
        mode="real_robot",
        namespace=namespace,
        command_topic=f"{prefix}/cmd_vel",
        odometry_topic=f"{prefix}/odom",
        scan_topic=f"{prefix}/scan",
        parent_frame="odom",
        child_frame="base_link",
        scan_frame="rplidar_link",
        max_linear_velocity=0.3,
        max_angular_velocity=1.0,
        command_timeout_s=0.2,
        scan_timeout_s=0.3,
        reset_service=None,
        command_qos=Ros2QoSContract(reliability="reliable", depth=1),
        localization_qos=Ros2QoSContract(reliability="reliable", depth=10),
        scan_qos=Ros2QoSContract(reliability="best_effort", depth=5),
    )
    profile.validate()
    return profile


@dataclass(frozen=True)
class TwistCommand:
    """Dependency-free planar geometry_msgs/Twist payload plus watchdog state."""

    linear_x: float
    angular_z: float
    timestamp_s: float
    timed_out: bool = False

    def __post_init__(self) -> None:
        values = (self.linear_x, self.angular_z, self.timestamp_s)
        if not np.all(np.isfinite(values)) or self.timestamp_s < 0.0:
            raise ValueError("Twist command values must be finite with non-negative time")


class TurtleBot4CommandBridge:
    """Map PointGoal normalized actions to bounded Twist with watchdog stop."""

    def __init__(self, profile: TurtleBot4Profile) -> None:
        profile.validate()
        self.profile = profile
        self._latest = TwistCommand(0.0, 0.0, 0.0, timed_out=True)
        self._received = False

    def submit_normalized_action(
        self,
        action: np.ndarray,
        *,
        timestamp_s: float,
    ) -> TwistCommand:
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("TurtleBot 4 normalized action must be a finite two-vector")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("TurtleBot 4 normalized action must remain in [-1, 1]")
        linear_x = 0.5 * (float(values[0]) + 1.0) * self.profile.max_linear_velocity
        angular_z = float(values[1]) * self.profile.max_angular_velocity
        self._latest = TwistCommand(linear_x, angular_z, timestamp_s)
        self._received = True
        return self._latest

    def command(self, *, now_s: float) -> TwistCommand:
        if not np.isfinite(now_s) or now_s < 0.0:
            raise ValueError("TurtleBot 4 command time must be finite and non-negative")
        if self._received and now_s < self._latest.timestamp_s:
            raise ValueError("TurtleBot 4 command query precedes the latest action")
        if not self._received or now_s - self._latest.timestamp_s > self.profile.command_timeout_s:
            return TwistCommand(0.0, 0.0, now_s, timed_out=True)
        return self._latest

    def stop(self, *, timestamp_s: float) -> TwistCommand:
        self._latest = TwistCommand(0.0, 0.0, timestamp_s)
        self._received = True
        return self._latest


@dataclass(frozen=True)
class LaserScanPacket:
    """Fixed-shape clipped LaserScan ranges and validity at one timestamp."""

    timestamp_s: float
    frame_id: str
    beam_angles: np.ndarray
    ranges: np.ndarray
    normalized_ranges: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        angles = np.asarray(self.beam_angles, dtype=get_global_dtype())
        ranges = np.asarray(self.ranges, dtype=get_global_dtype())
        normalized = np.asarray(self.normalized_ranges, dtype=get_global_dtype())
        valid = np.asarray(self.valid, dtype=bool)
        if angles.ndim != 1 or len(angles) == 0:
            raise ValueError("LaserScan beam angles must be a non-empty vector")
        if ranges.shape != angles.shape or normalized.shape != angles.shape:
            raise ValueError("LaserScan range arrays must match beam angles")
        if valid.shape != angles.shape:
            raise ValueError("LaserScan validity must match beam angles")
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(ranges)):
            raise ValueError("LaserScan angles and clipped ranges must be finite")
        if not np.all(np.isfinite(normalized)):
            raise ValueError("LaserScan normalized ranges must be finite")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("LaserScan timestamp must be finite and non-negative")
        if not self.frame_id:
            raise ValueError("LaserScan frame_id must be non-empty")
        object.__setattr__(self, "beam_angles", angles.copy())
        object.__setattr__(self, "ranges", ranges.copy())
        object.__setattr__(self, "normalized_ranges", normalized.copy())
        object.__setattr__(self, "valid", valid.copy())


@dataclass(frozen=True)
class LaserScanBridgeCfg:
    beam_count: int = 16
    angle_min: float = -np.pi
    angle_max: float = np.pi
    min_range: float = 0.05
    max_range: float = 5.0

    def validate(self) -> None:
        if self.beam_count <= 0:
            raise ValueError("LaserScan beam_count must be positive")
        if not np.isfinite(self.angle_min) or not np.isfinite(self.angle_max):
            raise ValueError("LaserScan target angles must be finite")
        if self.angle_max <= self.angle_min:
            raise ValueError("LaserScan angle_max must exceed angle_min")
        if not np.isfinite(self.min_range) or not np.isfinite(self.max_range):
            raise ValueError("LaserScan target ranges must be finite")
        if self.min_range < 0.0 or self.max_range <= self.min_range:
            raise ValueError("LaserScan target range bounds are invalid")


class TurtleBot4LaserScanBridge:
    """Validate and resample sensor_msgs/LaserScan-shaped messages."""

    def __init__(self, profile: TurtleBot4Profile, cfg: LaserScanBridgeCfg) -> None:
        profile.validate()
        cfg.validate()
        self.profile = profile
        self.cfg = cfg
        self.beam_angles = np.linspace(
            cfg.angle_min,
            cfg.angle_max,
            cfg.beam_count,
            endpoint=False,
            dtype=np.float64,
        )

    def convert(self, message: Any, *, received_time_s: float) -> LaserScanPacket:
        timestamp_s = _stamp_seconds(message.header.stamp)
        if not np.isfinite(received_time_s) or received_time_s < 0.0:
            raise ValueError("LaserScan receipt time must be finite and non-negative")
        if timestamp_s > received_time_s + 1.0e-6:
            raise ValueError("LaserScan stamp is in the future")
        if received_time_s - timestamp_s > self.profile.scan_timeout_s:
            raise ValueError("LaserScan message is stale")
        if str(message.header.frame_id) != self.profile.scan_frame:
            raise ValueError("LaserScan frame does not match TurtleBot 4 profile")
        source_ranges = np.asarray(message.ranges, dtype=np.float64)
        if source_ranges.ndim != 1 or len(source_ranges) < 2:
            raise ValueError("LaserScan ranges must contain at least two beams")
        angle_min = float(message.angle_min)
        angle_increment = float(message.angle_increment)
        range_min = float(message.range_min)
        range_max = float(message.range_max)
        if not np.all(np.isfinite([angle_min, angle_increment, range_min, range_max])):
            raise ValueError("LaserScan geometry and range limits must be finite")
        if angle_increment <= 0.0 or range_min < 0.0 or range_max <= range_min:
            raise ValueError("LaserScan geometry or range limits are invalid")
        source_angles = angle_min + np.arange(len(source_ranges)) * angle_increment
        if (
            self.beam_angles[0] < source_angles[0] - 1.0e-6
            or self.beam_angles[-1] > source_angles[-1] + 1.0e-6
        ):
            raise ValueError("LaserScan angular coverage does not include target beams")
        source_valid = (
            np.isfinite(source_ranges)
            & (source_ranges >= range_min)
            & (source_ranges <= range_max)
        )
        safe_ranges = np.where(source_valid, source_ranges, range_max)
        target_ranges = np.interp(self.beam_angles, source_angles, safe_ranges)
        target_valid = np.interp(
            self.beam_angles,
            source_angles,
            source_valid.astype(np.float64),
        ) >= 1.0 - 1.0e-9
        target_ranges = np.clip(target_ranges, self.cfg.min_range, self.cfg.max_range)
        normalized = (target_ranges - self.cfg.min_range) / (
            self.cfg.max_range - self.cfg.min_range
        )
        return LaserScanPacket(
            timestamp_s=timestamp_s,
            frame_id=self.profile.scan_frame,
            beam_angles=self.beam_angles,
            ranges=target_ranges,
            normalized_ranges=normalized,
            valid=target_valid,
        )


class EpisodeState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass(frozen=True)
class EpisodeResetRequest:
    episode_id: str
    timestamp_s: float
    seed: int | None = None


class TurtleBot4Transport(Protocol):
    def publish_twist(self, command: TwistCommand) -> None: ...

    def request_reset(self, service: str, request: EpisodeResetRequest) -> None: ...


class TurtleBot4EpisodeBridge:
    """Own episode state, command publication, watchdog, and reset safety."""

    def __init__(self, profile: TurtleBot4Profile, transport: TurtleBot4Transport) -> None:
        profile.validate()
        self.profile = profile
        self.transport = transport
        self.command_bridge = TurtleBot4CommandBridge(profile)
        self.state = EpisodeState.IDLE
        self.episode_id: str | None = None

    def start(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.state == EpisodeState.RUNNING:
            raise RuntimeError("an episode is already running")
        self.episode_id = episode_id
        self.state = EpisodeState.RUNNING

    def publish_action(self, action: np.ndarray, *, timestamp_s: float) -> TwistCommand:
        if self.state != EpisodeState.RUNNING:
            raise RuntimeError("cannot publish action outside a running episode")
        command = self.command_bridge.submit_normalized_action(
            action,
            timestamp_s=timestamp_s,
        )
        self.transport.publish_twist(command)
        return command

    def enforce_watchdog(self, *, now_s: float) -> TwistCommand:
        command = self.command_bridge.command(now_s=now_s)
        if command.timed_out:
            self.transport.publish_twist(command)
        return command

    def stop(self, *, timestamp_s: float) -> TwistCommand:
        command = self.command_bridge.stop(timestamp_s=timestamp_s)
        self.transport.publish_twist(command)
        self.state = EpisodeState.STOPPED
        return command

    def reset(
        self,
        request: EpisodeResetRequest,
        *,
        operator_confirmed: bool = False,
    ) -> None:
        self.transport.publish_twist(
            self.command_bridge.stop(timestamp_s=request.timestamp_s)
        )
        if self.profile.mode == "simulation":
            assert self.profile.reset_service is not None
            self.transport.request_reset(self.profile.reset_service, request)
        elif not operator_confirmed:
            raise PermissionError("real-robot reset requires operator confirmation")
        self.episode_id = request.episode_id
        self.state = EpisodeState.IDLE


class Ros2NodeTransport:
    """Optional live adapter around an injected rclpy-compatible node."""

    def __init__(
        self,
        node: Any,
        profile: TurtleBot4Profile,
        *,
        twist_message_type: type,
        reset_service_type: type | None = None,
    ) -> None:
        profile.validate()
        self.node = node
        self.profile = profile
        self.twist_message_type = twist_message_type
        self.reset_service_type = reset_service_type
        self.publisher = node.create_publisher(
            twist_message_type,
            profile.command_topic,
            profile.command_qos.depth,
        )
        self.reset_client = (
            node.create_client(reset_service_type, profile.reset_service)
            if reset_service_type is not None and profile.reset_service is not None
            else None
        )

    def publish_twist(self, command: TwistCommand) -> None:
        message = self.twist_message_type()
        message.linear.x = command.linear_x
        message.angular.z = command.angular_z
        self.publisher.publish(message)

    def request_reset(self, service: str, request: EpisodeResetRequest) -> None:
        if self.reset_client is None or service != self.profile.reset_service:
            raise RuntimeError("ROS2 reset client is unavailable")
        ros_request = self.reset_service_type.Request()
        if hasattr(ros_request, "episode_id"):
            ros_request.episode_id = request.episode_id
        if hasattr(ros_request, "seed") and request.seed is not None:
            ros_request.seed = request.seed
        self.reset_client.call_async(ros_request)
