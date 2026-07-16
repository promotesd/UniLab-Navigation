"""Backend-independent localization provider contracts for navigation."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from unilab.dtype_config import get_global_dtype


class PoseStatus(IntEnum):
    """Per-environment localization health state."""

    UNINITIALIZED = 0
    TRACKING = 1
    DEGRADED = 2
    LOST = 3


@dataclass(frozen=True)
class GroundTruthPosePacket:
    """Timestamped planar ground-truth pose batch used only by truth/reset paths."""

    timestamp_s: np.ndarray
    pose: np.ndarray
    parent_frame: str = "map"
    child_frame: str = "base_link"

    def __post_init__(self) -> None:
        timestamp = np.asarray(self.timestamp_s, dtype=np.float64)
        pose = np.asarray(self.pose, dtype=get_global_dtype())
        count = len(timestamp) if timestamp.ndim == 1 else 0
        if pose.shape != (count, 3):
            raise ValueError("ground-truth pose must have shape (environments, 3)")
        if not np.all(np.isfinite(timestamp)) or np.any(timestamp < 0.0):
            raise ValueError("ground-truth timestamps must be finite and non-negative")
        if not np.all(np.isfinite(pose)):
            raise ValueError("ground-truth pose must be finite")
        _validate_frame_pair(self.parent_frame, self.child_frame)
        object.__setattr__(self, "timestamp_s", timestamp.copy())
        object.__setattr__(self, "pose", pose.copy())


@dataclass(frozen=True)
class WheelOdometryPacket:
    """Timestamped planar velocity batch expressed in the robot base frame."""

    timestamp_s: np.ndarray
    dt_s: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    frame_id: str = "base_link"

    def __post_init__(self) -> None:
        timestamp = np.asarray(self.timestamp_s, dtype=np.float64)
        dt = np.asarray(self.dt_s, dtype=np.float64)
        linear = np.asarray(self.linear_velocity, dtype=get_global_dtype())
        angular = np.asarray(self.angular_velocity, dtype=get_global_dtype())
        count = len(timestamp) if timestamp.ndim == 1 else 0
        for name, value in (
            ("dt_s", dt),
            ("linear_velocity", linear),
            ("angular_velocity", angular),
        ):
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (environments,)")
        if not np.all(np.isfinite(timestamp)) or np.any(timestamp < 0.0):
            raise ValueError("odometry timestamps must be finite and non-negative")
        if not np.all(np.isfinite(dt)) or np.any(dt < 0.0):
            raise ValueError("odometry intervals must be finite and non-negative")
        if not np.all(np.isfinite(linear)) or not np.all(np.isfinite(angular)):
            raise ValueError("odometry velocities must be finite")
        if not self.frame_id:
            raise ValueError("odometry frame_id must be non-empty")
        object.__setattr__(self, "timestamp_s", timestamp.copy())
        object.__setattr__(self, "dt_s", dt.copy())
        object.__setattr__(self, "linear_velocity", linear.copy())
        object.__setattr__(self, "angular_velocity", angular.copy())


@dataclass(frozen=True)
class LocalizationPacket:
    """Typed sensor packet batch presented to localization providers."""

    ground_truth: GroundTruthPosePacket
    wheel_odometry: WheelOdometryPacket

    def __post_init__(self) -> None:
        if not np.array_equal(
            self.ground_truth.timestamp_s,
            self.wheel_odometry.timestamp_s,
        ):
            raise ValueError("localization packet timestamps must match")
        if self.ground_truth.child_frame != self.wheel_odometry.frame_id:
            raise ValueError("ground-truth child and odometry frames must match")

    @property
    def timestamp_s(self) -> np.ndarray:
        return self.ground_truth.timestamp_s


def _validate_frame_pair(parent_frame: str, child_frame: str) -> None:
    if not parent_frame or not child_frame:
        raise ValueError("frame names must be non-empty")
    if parent_frame == child_frame:
        raise ValueError("parent and child frames must differ")


@dataclass(frozen=True)
class PoseEstimate:
    """Validated batched planar localization output."""

    pose: np.ndarray
    covariance: np.ndarray
    valid: np.ndarray
    status: np.ndarray
    timestamp_s: np.ndarray
    parent_frame: str = "map"
    child_frame: str = "base_link"

    def __post_init__(self) -> None:
        pose = np.asarray(self.pose, dtype=get_global_dtype())
        covariance = np.asarray(self.covariance, dtype=get_global_dtype())
        valid = np.asarray(self.valid, dtype=bool)
        status = np.asarray(self.status, dtype=np.uint8)
        timestamp = np.asarray(self.timestamp_s, dtype=np.float64)
        count = len(pose) if pose.ndim >= 1 else 0
        if pose.shape != (count, 3):
            raise ValueError("pose must have shape (environments, 3)")
        if covariance.shape != (count, 3, 3):
            raise ValueError("covariance must have shape (environments, 3, 3)")
        for name, value in (
            ("valid", valid),
            ("status", status),
            ("timestamp_s", timestamp),
        ):
            if value.shape != (count,):
                raise ValueError(f"{name} must have shape (environments,)")
        if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(covariance)):
            raise ValueError("pose and covariance must be finite")
        if not np.all(np.isfinite(timestamp)) or np.any(timestamp < 0.0):
            raise ValueError("timestamps must be finite and non-negative")
        if not np.allclose(covariance, covariance.swapaxes(1, 2), atol=1.0e-6):
            raise ValueError("covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(covariance) < -1.0e-6):
            raise ValueError("covariance must be positive semidefinite")
        valid_statuses = np.array([int(value) for value in PoseStatus], dtype=np.uint8)
        if not np.all(np.isin(status, valid_statuses)):
            raise ValueError("status contains an unknown PoseStatus value")
        expected_valid = np.isin(
            status,
            np.array([PoseStatus.TRACKING, PoseStatus.DEGRADED], dtype=np.uint8),
        )
        if not np.array_equal(valid, expected_valid):
            raise ValueError("valid must agree with the localization status")
        if not self.parent_frame or not self.child_frame:
            raise ValueError("localization frame names must be non-empty")
        if self.parent_frame == self.child_frame:
            raise ValueError("parent and child localization frames must differ")
        object.__setattr__(self, "pose", pose.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "valid", valid.copy())
        object.__setattr__(self, "status", status.copy())
        object.__setattr__(self, "timestamp_s", timestamp.copy())


class PoseProvider(Protocol):
    """Stable localization interface consumed by navigation observations."""

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        """Reset provider state for explicit source poses."""

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        """Produce one batched pose estimate."""


class GroundTruthPoseProvider:
    """Pass simulator pose through the provider contract unchanged."""

    def __init__(self, *, parent_frame: str = "map", child_frame: str = "base_link") -> None:
        if not parent_frame or not child_frame or parent_frame == child_frame:
            raise ValueError("ground-truth provider requires distinct non-empty frames")
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self._last_timestamp_s: np.ndarray | None = None

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        timestamp = packet.timestamp_s
        indices = np.asarray(env_indices, dtype=np.int32)
        if self._last_timestamp_s is not None:
            if timestamp.shape != self._last_timestamp_s.shape:
                raise ValueError("timestamp batch shape changed")
            unchanged = np.ones(len(timestamp), dtype=bool)
            unchanged[indices] = False
            if np.any(timestamp[unchanged] < self._last_timestamp_s[unchanged]):
                raise ValueError("timestamps moved backward outside reset environments")
        self._last_timestamp_s = timestamp.copy()
        return self.update(packet)

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        truth = packet.ground_truth
        if truth.parent_frame != self.parent_frame or truth.child_frame != self.child_frame:
            raise ValueError(
                "ground-truth packet frames do not match the provider contract"
            )
        pose = truth.pose
        timestamp = packet.timestamp_s
        count = len(pose) if pose.ndim >= 1 else 0
        if self._last_timestamp_s is not None:
            if timestamp.shape != self._last_timestamp_s.shape:
                raise ValueError("timestamp batch shape changed")
            if np.any(timestamp < self._last_timestamp_s):
                raise ValueError("localization timestamps must be monotonic")
        estimate = PoseEstimate(
            pose=pose,
            covariance=np.zeros((count, 3, 3), dtype=get_global_dtype()),
            valid=np.ones(count, dtype=bool),
            status=np.full(count, PoseStatus.TRACKING, dtype=np.uint8),
            timestamp_s=timestamp,
            parent_frame=self.parent_frame,
            child_frame=self.child_frame,
        )
        self._last_timestamp_s = timestamp.copy()
        return estimate


@dataclass
class NoisyPoseCfg:
    """Noise, bias, and seed configuration for truth-derived pose estimates."""

    seed: int = 1
    position_noise_std: float = 0.0
    heading_noise_std: float = 0.0
    x_bias: float = 0.0
    y_bias: float = 0.0
    heading_bias: float = 0.0

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("noisy-pose seed must be non-negative")
        values = (
            self.position_noise_std,
            self.heading_noise_std,
            self.x_bias,
            self.y_bias,
            self.heading_bias,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("noisy-pose configuration must be finite")
        if self.position_noise_std < 0.0 or self.heading_noise_std < 0.0:
            raise ValueError("noisy-pose standard deviations must be non-negative")


class NoisyPoseProvider:
    """Perturb each ground-truth pose sample with deterministic seeded noise."""

    def __init__(
        self,
        cfg: NoisyPoseCfg,
        *,
        parent_frame: str = "map",
        child_frame: str = "base_link",
    ) -> None:
        cfg.validate()
        _validate_frame_pair(parent_frame, child_frame)
        self.cfg = cfg
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self._rng = np.random.default_rng(cfg.seed)
        self._pose: np.ndarray | None = None
        self._timestamp_s: np.ndarray | None = None

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        truth = self._validated_truth(packet)
        count = len(truth.pose)
        if self._pose is None or len(self._pose) != count:
            self._pose = truth.pose.copy()
            self._timestamp_s = truth.timestamp_s.copy()
            indices = np.arange(count, dtype=np.int32)
        else:
            indices = np.asarray(env_indices, dtype=np.int32)
            assert self._timestamp_s is not None
            unchanged = np.ones(count, dtype=bool)
            unchanged[indices] = False
            if np.any(truth.timestamp_s[unchanged] < self._timestamp_s[unchanged]):
                raise ValueError("timestamps moved backward outside reset environments")
            self._timestamp_s[indices] = truth.timestamp_s[indices]
        self._apply_noise(truth.pose, indices)
        return self._estimate()

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        truth = self._validated_truth(packet)
        if self._pose is None or self._timestamp_s is None:
            raise RuntimeError("noisy-pose provider must be reset before update")
        if len(truth.pose) != len(self._pose):
            raise ValueError("noisy-pose packet batch shape changed")
        if np.any(truth.timestamp_s < self._timestamp_s):
            raise ValueError("noisy-pose timestamps must be monotonic")
        indices = np.arange(len(self._pose), dtype=np.int32)
        self._apply_noise(truth.pose, indices)
        self._timestamp_s[:] = truth.timestamp_s
        return self._estimate()

    def _validated_truth(self, packet: LocalizationPacket) -> GroundTruthPosePacket:
        truth = packet.ground_truth
        if truth.parent_frame != self.parent_frame or truth.child_frame != self.child_frame:
            raise ValueError("ground-truth packet frames do not match noisy-pose provider")
        return truth

    def _apply_noise(self, truth_pose: np.ndarray, indices: np.ndarray) -> None:
        assert self._pose is not None
        count = len(indices)
        position_noise = self._rng.normal(
            0.0,
            self.cfg.position_noise_std,
            size=(count, 2),
        )
        heading_noise = self._rng.normal(
            0.0,
            self.cfg.heading_noise_std,
            size=count,
        )
        self._pose[indices, :2] = truth_pose[indices, :2] + position_noise
        self._pose[indices, 0] += self.cfg.x_bias
        self._pose[indices, 1] += self.cfg.y_bias
        self._pose[indices, 2] = (
            truth_pose[indices, 2] + self.cfg.heading_bias + heading_noise + np.pi
        ) % (2.0 * np.pi) - np.pi

    def _estimate(self) -> PoseEstimate:
        assert self._pose is not None
        assert self._timestamp_s is not None
        count = len(self._pose)
        covariance = np.zeros((count, 3, 3), dtype=get_global_dtype())
        covariance[:, 0, 0] = self.cfg.position_noise_std**2
        covariance[:, 1, 1] = self.cfg.position_noise_std**2
        covariance[:, 2, 2] = self.cfg.heading_noise_std**2
        degraded = any(
            value != 0.0
            for value in (
                self.cfg.position_noise_std,
                self.cfg.heading_noise_std,
                self.cfg.x_bias,
                self.cfg.y_bias,
                self.cfg.heading_bias,
            )
        )
        status = PoseStatus.DEGRADED if degraded else PoseStatus.TRACKING
        return PoseEstimate(
            pose=self._pose,
            covariance=covariance,
            valid=np.ones(count, dtype=bool),
            status=np.full(count, status, dtype=np.uint8),
            timestamp_s=self._timestamp_s,
            parent_frame=self.parent_frame,
            child_frame=self.child_frame,
        )


@dataclass
class DeadReckoningCfg:
    """Deterministic noise and covariance growth for planar dead reckoning."""

    seed: int = 1
    linear_velocity_noise_std: float = 0.0
    angular_velocity_noise_std: float = 0.0
    linear_velocity_bias: float = 0.0
    angular_velocity_bias: float = 0.0
    initial_position_variance: float = 0.0
    initial_heading_variance: float = 0.0

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("dead-reckoning seed must be non-negative")
        values = (
            self.linear_velocity_noise_std,
            self.angular_velocity_noise_std,
            self.linear_velocity_bias,
            self.angular_velocity_bias,
            self.initial_position_variance,
            self.initial_heading_variance,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("dead-reckoning configuration must be finite")
        if self.linear_velocity_noise_std < 0.0 or self.angular_velocity_noise_std < 0.0:
            raise ValueError("dead-reckoning noise standard deviations must be non-negative")
        if self.initial_position_variance < 0.0 or self.initial_heading_variance < 0.0:
            raise ValueError("dead-reckoning initial variances must be non-negative")


class DeadReckoningPoseProvider:
    """Integrate planar velocity packets without reading simulator truth on update."""

    def __init__(
        self,
        cfg: DeadReckoningCfg,
        *,
        parent_frame: str = "odom",
        child_frame: str = "base_link",
    ) -> None:
        cfg.validate()
        if not parent_frame or not child_frame or parent_frame == child_frame:
            raise ValueError("dead reckoning requires distinct non-empty frames")
        self.cfg = cfg
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self._rng = np.random.default_rng(cfg.seed)
        self._pose: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._timestamp_s: np.ndarray | None = None

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        truth = packet.ground_truth
        if truth.child_frame != self.child_frame:
            raise ValueError("ground-truth child frame does not match dead reckoning")
        count = len(packet.timestamp_s)
        if self._pose is None or len(self._pose) != count:
            self._pose = truth.pose.copy()
            self._covariance = np.zeros((count, 3, 3), dtype=get_global_dtype())
            self._timestamp_s = packet.timestamp_s.copy()
            indices = np.arange(count, dtype=np.int32)
        else:
            indices = np.asarray(env_indices, dtype=np.int32)
            unchanged = np.ones(count, dtype=bool)
            unchanged[indices] = False
            if np.any(packet.timestamp_s[unchanged] < self._timestamp_s[unchanged]):
                raise ValueError("timestamps moved backward outside reset environments")
            self._pose[indices] = truth.pose[indices]
            self._timestamp_s[indices] = packet.timestamp_s[indices]
        assert self._covariance is not None
        self._covariance[indices] = 0.0
        self._covariance[indices, 0, 0] = self.cfg.initial_position_variance
        self._covariance[indices, 1, 1] = self.cfg.initial_position_variance
        self._covariance[indices, 2, 2] = self.cfg.initial_heading_variance
        return self._estimate()

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        if self._pose is None or self._covariance is None or self._timestamp_s is None:
            raise RuntimeError("dead-reckoning provider must be reset before update")
        if len(packet.timestamp_s) != len(self._pose):
            raise ValueError("dead-reckoning packet batch shape changed")
        if np.any(packet.timestamp_s < self._timestamp_s):
            raise ValueError("dead-reckoning timestamps must be monotonic")
        count = len(self._pose)
        odometry = packet.wheel_odometry
        if odometry.frame_id != self.child_frame:
            raise ValueError("wheel-odometry frame does not match dead reckoning")
        elapsed = packet.timestamp_s - self._timestamp_s
        if not np.allclose(elapsed, odometry.dt_s, atol=1.0e-9, rtol=0.0):
            raise ValueError("wheel-odometry intervals do not match packet timestamps")
        linear_noise = self._rng.normal(
            0.0,
            self.cfg.linear_velocity_noise_std,
            size=count,
        )
        angular_noise = self._rng.normal(
            0.0,
            self.cfg.angular_velocity_noise_std,
            size=count,
        )
        linear = odometry.linear_velocity + self.cfg.linear_velocity_bias + linear_noise
        angular = odometry.angular_velocity + self.cfg.angular_velocity_bias + angular_noise
        heading = self._pose[:, 2]
        self._pose[:, 0] += linear * np.cos(heading) * odometry.dt_s
        self._pose[:, 1] += linear * np.sin(heading) * odometry.dt_s
        self._pose[:, 2] = (heading + angular * odometry.dt_s + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        position_variance = (self.cfg.linear_velocity_noise_std * odometry.dt_s) ** 2
        heading_variance = (self.cfg.angular_velocity_noise_std * odometry.dt_s) ** 2
        self._covariance[:, 0, 0] += position_variance
        self._covariance[:, 1, 1] += position_variance
        self._covariance[:, 2, 2] += heading_variance
        self._timestamp_s[:] = packet.timestamp_s
        return self._estimate()

    def _estimate(self) -> PoseEstimate:
        assert self._pose is not None
        assert self._covariance is not None
        assert self._timestamp_s is not None
        count = len(self._pose)
        return PoseEstimate(
            pose=self._pose,
            covariance=self._covariance,
            valid=np.ones(count, dtype=bool),
            status=np.full(count, PoseStatus.TRACKING, dtype=np.uint8),
            timestamp_s=self._timestamp_s,
            parent_frame=self.parent_frame,
            child_frame=self.child_frame,
        )


def _recorded_stream_sha256(payload: Mapping[str, Any]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "sha256"}
    encoded = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RecordedPoseStream:
    """Serializable batched planar pose estimates on one strict timestamp axis."""

    timestamps_s: np.ndarray
    pose: np.ndarray
    covariance: np.ndarray
    status: np.ndarray
    parent_frame: str = "map"
    child_frame: str = "base_link"

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        pose = np.asarray(self.pose, dtype=get_global_dtype())
        covariance = np.asarray(self.covariance, dtype=get_global_dtype())
        status = np.asarray(self.status, dtype=np.uint8)
        if timestamps.ndim != 1 or len(timestamps) == 0:
            raise ValueError("recorded timestamps must be a non-empty vector")
        if not np.all(np.isfinite(timestamps)) or np.any(timestamps < 0.0):
            raise ValueError("recorded timestamps must be finite and non-negative")
        if np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("recorded timestamps must be strictly increasing")
        if pose.ndim != 3 or pose.shape[0] != len(timestamps) or pose.shape[2] != 3:
            raise ValueError("recorded pose must have shape (samples, environments, 3)")
        sample_count, environment_count, _ = pose.shape
        if covariance.shape != (sample_count, environment_count, 3, 3):
            raise ValueError(
                "recorded covariance must have shape (samples, environments, 3, 3)"
            )
        if status.shape != (sample_count, environment_count):
            raise ValueError("recorded status must have shape (samples, environments)")
        _validate_frame_pair(self.parent_frame, self.child_frame)
        PoseEstimate(
            pose=pose.reshape(-1, 3),
            covariance=covariance.reshape(-1, 3, 3),
            valid=np.isin(
                status.reshape(-1),
                np.array([PoseStatus.TRACKING, PoseStatus.DEGRADED], dtype=np.uint8),
            ),
            status=status.reshape(-1),
            timestamp_s=np.repeat(timestamps, environment_count),
            parent_frame=self.parent_frame,
            child_frame=self.child_frame,
        )
        object.__setattr__(self, "timestamps_s", timestamps.copy())
        object.__setattr__(self, "pose", pose.copy())
        object.__setattr__(self, "covariance", covariance.copy())
        object.__setattr__(self, "status", status.copy())

    @property
    def environment_count(self) -> int:
        return self.pose.shape[1]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "pose_convention": "x_y_yaw_radians",
            "linear_units": "meters",
            "angular_units": "radians",
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "timestamps_s": self.timestamps_s.tolist(),
            "pose": self.pose.tolist(),
            "covariance": self.covariance.tolist(),
            "status": self.status.tolist(),
        }
        payload["sha256"] = _recorded_stream_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RecordedPoseStream:
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("recorded-pose schema_version must be 1")
        if payload.get("pose_convention") != "x_y_yaw_radians":
            raise ValueError("recorded-pose convention must be x_y_yaw_radians")
        if payload.get("linear_units") != "meters":
            raise ValueError("recorded-pose linear units must be meters")
        if payload.get("angular_units") != "radians":
            raise ValueError("recorded-pose angular units must be radians")
        expected_hash = payload.get("sha256")
        if not isinstance(expected_hash, str):
            raise ValueError("recorded-pose stream requires a sha256 content hash")
        if expected_hash != _recorded_stream_sha256(payload):
            raise ValueError("recorded-pose sha256 does not match its content")
        return cls(
            timestamps_s=payload["timestamps_s"],
            pose=payload["pose"],
            covariance=payload["covariance"],
            status=payload["status"],
            parent_frame=str(payload["parent_frame"]),
            child_frame=str(payload["child_frame"]),
        )


def write_recorded_pose_stream(
    stream: RecordedPoseStream,
    output_path: str | Path,
) -> Path:
    """Write one strict recorded-pose JSON stream."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stream.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def read_recorded_pose_stream(input_path: str | Path) -> RecordedPoseStream:
    """Read and validate one recorded-pose JSON stream."""
    path = Path(input_path)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"recorded-pose stream does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"recorded-pose stream is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("recorded-pose stream must contain a JSON object")
    return RecordedPoseStream.from_dict(payload)


@dataclass
class RecordedPoseCfg:
    """Timestamp lookup and failure behavior for offline pose replay."""

    path: str | None = None
    interpolation: str = "linear"
    out_of_range: str = "error"
    max_sample_age_s: float | None = None
    frame_transform: str = "identity"

    def validate(self) -> None:
        if self.interpolation not in {"linear", "previous"}:
            raise ValueError("recorded-pose interpolation must be linear or previous")
        if self.out_of_range not in {"error", "clamp", "lost"}:
            raise ValueError("recorded-pose out_of_range must be error, clamp, or lost")
        if self.frame_transform != "identity":
            raise ValueError("recorded-pose frame_transform currently supports only identity")
        if self.max_sample_age_s is not None:
            if not np.isfinite(self.max_sample_age_s) or self.max_sample_age_s < 0.0:
                raise ValueError("recorded-pose max_sample_age_s must be non-negative")


class RecordedPoseProvider:
    """Replay timestamped offline estimates without consuming update truth."""

    def __init__(self, stream: RecordedPoseStream, cfg: RecordedPoseCfg) -> None:
        cfg.validate()
        self.stream = stream
        self.cfg = cfg
        self.parent_frame = stream.parent_frame
        self.child_frame = stream.child_frame
        self._pose: np.ndarray | None = None
        self._covariance: np.ndarray | None = None
        self._status: np.ndarray | None = None
        self._estimate_timestamp_s: np.ndarray | None = None
        self._last_query_timestamp_s: np.ndarray | None = None

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        self._validate_packet(packet)
        count = len(packet.timestamp_s)
        indices = np.asarray(env_indices, dtype=np.int32)
        if self._pose is None:
            indices = np.arange(count, dtype=np.int32)
            self._pose = np.zeros((count, 3), dtype=get_global_dtype())
            self._covariance = np.zeros((count, 3, 3), dtype=get_global_dtype())
            self._status = np.full(count, PoseStatus.UNINITIALIZED, dtype=np.uint8)
            self._estimate_timestamp_s = np.zeros(count, dtype=np.float64)
            self._last_query_timestamp_s = packet.timestamp_s.copy()
        else:
            assert self._last_query_timestamp_s is not None
            unchanged = np.ones(count, dtype=bool)
            unchanged[indices] = False
            if np.any(packet.timestamp_s[unchanged] < self._last_query_timestamp_s[unchanged]):
                raise ValueError("timestamps moved backward outside reset environments")
            self._last_query_timestamp_s[indices] = packet.timestamp_s[indices]
        self._assign_lookup(packet.timestamp_s[indices], indices)
        return self._estimate()

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        self._validate_packet(packet)
        if self._pose is None or self._last_query_timestamp_s is None:
            raise RuntimeError("recorded-pose provider must be reset before update")
        if np.any(packet.timestamp_s < self._last_query_timestamp_s):
            raise ValueError("recorded-pose query timestamps must be monotonic")
        indices = np.arange(len(packet.timestamp_s), dtype=np.int32)
        self._assign_lookup(packet.timestamp_s, indices)
        self._last_query_timestamp_s[:] = packet.timestamp_s
        return self._estimate()

    def _validate_packet(self, packet: LocalizationPacket) -> None:
        if len(packet.timestamp_s) != self.stream.environment_count:
            raise ValueError("recorded-pose environment count does not match packet batch")
        if packet.ground_truth.child_frame != self.child_frame:
            raise ValueError("recorded-pose child frame does not match sensor packet")

    def _assign_lookup(self, query: np.ndarray, env_indices: np.ndarray) -> None:
        pose, covariance, status, timestamp = self._lookup(query, env_indices)
        assert self._pose is not None
        assert self._covariance is not None
        assert self._status is not None
        assert self._estimate_timestamp_s is not None
        self._pose[env_indices] = pose
        self._covariance[env_indices] = covariance
        self._status[env_indices] = status
        self._estimate_timestamp_s[env_indices] = timestamp

    def _lookup(
        self,
        query: np.ndarray,
        env_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        timestamps = self.stream.timestamps_s
        timestamp_tolerance_s = 1.0e-9
        below = query < timestamps[0] - timestamp_tolerance_s
        above = query > timestamps[-1] + timestamp_tolerance_s
        outside = below | above
        if self.cfg.out_of_range == "error" and np.any(outside):
            raise ValueError(
                f"{int(np.count_nonzero(outside))} recorded-pose queries are out of range"
            )
        lookup_query = np.clip(query, timestamps[0], timestamps[-1])

        if self.cfg.interpolation == "previous":
            lower = np.searchsorted(timestamps, lookup_query, side="right") - 1
            lower = np.clip(lower, 0, len(timestamps) - 1)
            upper = lower.copy()
            alpha = np.zeros(len(query), dtype=np.float64)
            estimate_timestamp = timestamps[lower].copy()
        else:
            upper = np.searchsorted(timestamps, lookup_query, side="left")
            upper = np.clip(upper, 0, len(timestamps) - 1)
            lower = np.maximum(upper - 1, 0)
            exact = timestamps[upper] == lookup_query
            lower[exact] = upper[exact]
            lower[below] = 0
            upper[below] = 0
            lower[above] = len(timestamps) - 1
            upper[above] = len(timestamps) - 1
            denominator = timestamps[upper] - timestamps[lower]
            alpha = np.divide(
                lookup_query - timestamps[lower],
                denominator,
                out=np.zeros(len(query), dtype=np.float64),
                where=denominator > 0.0,
            )
            estimate_timestamp = np.where(
                lower == upper,
                timestamps[lower],
                lookup_query,
            )

        pose0 = self.stream.pose[lower, env_indices]
        pose1 = self.stream.pose[upper, env_indices]
        covariance0 = self.stream.covariance[lower, env_indices]
        covariance1 = self.stream.covariance[upper, env_indices]
        weight = alpha[:, None]
        pose = pose0.copy()
        pose[:, :2] = pose0[:, :2] + weight * (pose1[:, :2] - pose0[:, :2])
        yaw_delta = (pose1[:, 2] - pose0[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
        pose[:, 2] = (pose0[:, 2] + alpha * yaw_delta + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        covariance = covariance0 + alpha[:, None, None] * (
            covariance1 - covariance0
        )
        status0 = self.stream.status[lower, env_indices]
        status1 = self.stream.status[upper, env_indices]
        valid0 = np.isin(status0, [PoseStatus.TRACKING, PoseStatus.DEGRADED])
        valid1 = np.isin(status1, [PoseStatus.TRACKING, PoseStatus.DEGRADED])
        status = np.maximum(status0, status1)
        status[~(valid0 & valid1)] = PoseStatus.LOST

        if self.cfg.out_of_range == "lost":
            status[outside] = PoseStatus.LOST
        if self.cfg.max_sample_age_s is not None:
            stale = np.abs(query - estimate_timestamp) > self.cfg.max_sample_age_s
            status[stale] = PoseStatus.LOST
        return pose, covariance, status, estimate_timestamp

    def _estimate(self) -> PoseEstimate:
        assert self._pose is not None
        assert self._covariance is not None
        assert self._status is not None
        assert self._estimate_timestamp_s is not None
        valid = np.isin(
            self._status,
            np.array([PoseStatus.TRACKING, PoseStatus.DEGRADED], dtype=np.uint8),
        )
        return PoseEstimate(
            pose=self._pose,
            covariance=self._covariance,
            valid=valid,
            status=self._status,
            timestamp_s=self._estimate_timestamp_s,
            parent_frame=self.parent_frame,
            child_frame=self.child_frame,
        )


class OnlineEstimatorPlugin(Protocol):
    """Dependency-free lifecycle contract implemented by online estimators."""

    parent_frame: str
    child_frame: str

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        """Reset estimator-owned state for selected environments."""

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        """Consume one typed sensor packet batch and return the newest estimate."""

    def close(self) -> None:
        """Release plugin-owned resources."""


@dataclass
class OnlineEstimatorCfg:
    """UniLab-owned scheduling and failure semantics for an online plugin."""

    plugin: str = (
        "unilab.envs.navigation.localization:AnalyticalWheelOdometryPlugin"
    )
    update_interval_steps: int = 1
    latency_steps: int = 0
    dropout_probability: float = 0.0
    seed: int = 1
    max_staleness_s: float | None = None
    dead_reckoning: DeadReckoningCfg = field(default_factory=DeadReckoningCfg)

    def validate(self) -> None:
        module_name, separator, symbol_name = self.plugin.partition(":")
        if not separator or not module_name or not symbol_name:
            raise ValueError("online-estimator plugin must use module:factory")
        if self.update_interval_steps <= 0:
            raise ValueError("online-estimator update_interval_steps must be positive")
        if self.latency_steps < 0:
            raise ValueError("online-estimator latency_steps must be non-negative")
        if (
            not np.isfinite(self.dropout_probability)
            or not 0.0 <= self.dropout_probability <= 1.0
        ):
            raise ValueError("online-estimator dropout_probability must be in [0, 1]")
        if self.seed < 0:
            raise ValueError("online-estimator seed must be non-negative")
        if self.max_staleness_s is not None:
            if not np.isfinite(self.max_staleness_s) or self.max_staleness_s < 0.0:
                raise ValueError("online-estimator max_staleness_s must be non-negative")
        self.dead_reckoning.validate()


class AnalyticalWheelOdometryPlugin:
    """Reference online plugin backed by analytical wheel dead reckoning."""

    def __init__(
        self,
        cfg: OnlineEstimatorCfg,
        *,
        parent_frame: str,
        child_frame: str,
    ) -> None:
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self._provider = DeadReckoningPoseProvider(
            cfg.dead_reckoning,
            parent_frame=parent_frame,
            child_frame=child_frame,
        )
        self.closed = False

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        if self.closed:
            raise RuntimeError("online estimator plugin is closed")
        return self._provider.reset(packet, env_indices)

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        if self.closed:
            raise RuntimeError("online estimator plugin is closed")
        return self._provider.update(packet)

    def close(self) -> None:
        self.closed = True


def _replace_estimate_indices(
    base: PoseEstimate,
    replacement: PoseEstimate,
    indices: np.ndarray,
) -> PoseEstimate:
    if (
        base.parent_frame != replacement.parent_frame
        or base.child_frame != replacement.child_frame
        or len(base.pose) != len(replacement.pose)
    ):
        raise ValueError("online estimator changed frame or batch contract")
    pose = base.pose.copy()
    covariance = base.covariance.copy()
    status = base.status.copy()
    timestamp = base.timestamp_s.copy()
    pose[indices] = replacement.pose[indices]
    covariance[indices] = replacement.covariance[indices]
    status[indices] = replacement.status[indices]
    timestamp[indices] = replacement.timestamp_s[indices]
    return PoseEstimate(
        pose=pose,
        covariance=covariance,
        valid=np.isin(status, [PoseStatus.TRACKING, PoseStatus.DEGRADED]),
        status=status,
        timestamp_s=timestamp,
        parent_frame=base.parent_frame,
        child_frame=base.child_frame,
    )


class OnlineEstimatorPoseProvider:
    """Own online scheduling, latency, dropout, staleness, and shutdown."""

    def __init__(
        self,
        plugin: OnlineEstimatorPlugin,
        cfg: OnlineEstimatorCfg,
    ) -> None:
        cfg.validate()
        _validate_frame_pair(plugin.parent_frame, plugin.child_frame)
        self.plugin = plugin
        self.cfg = cfg
        self.parent_frame = plugin.parent_frame
        self.child_frame = plugin.child_frame
        self._rng = np.random.default_rng(cfg.seed)
        self._update_count = 0
        self._latest_plugin_raw: PoseEstimate | None = None
        self._published_raw: PoseEstimate | None = None
        self._latency_queue: deque[PoseEstimate] | None = None
        self._closed = False

    def reset(
        self,
        packet: LocalizationPacket,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        self._ensure_open()
        indices = np.asarray(env_indices, dtype=np.int32)
        raw = self.plugin.reset(packet, indices)
        self._validate_estimate(raw, packet.timestamp_s)
        if self._published_raw is None:
            self._latest_plugin_raw = raw
            self._published_raw = raw
            self._latency_queue = deque(
                [raw] * (self.cfg.latency_steps + 1),
                maxlen=self.cfg.latency_steps + 1,
            )
        else:
            assert self._latest_plugin_raw is not None
            unchanged = np.ones(len(packet.timestamp_s), dtype=bool)
            unchanged[indices] = False
            if np.any(
                raw.timestamp_s[unchanged]
                < self._latest_plugin_raw.timestamp_s[unchanged]
            ):
                raise ValueError(
                    "online estimator timestamps moved backward outside reset environments"
                )
            self._latest_plugin_raw = _replace_estimate_indices(
                self._latest_plugin_raw,
                raw,
                indices,
            )
            self._published_raw = _replace_estimate_indices(
                self._published_raw,
                raw,
                indices,
            )
            assert self._latency_queue is not None
            self._latency_queue = deque(
                [
                    _replace_estimate_indices(estimate, raw, indices)
                    for estimate in self._latency_queue
                ],
                maxlen=self.cfg.latency_steps + 1,
            )
        assert self._latency_queue is not None
        return self._apply_health(self._latency_queue[0], packet.timestamp_s)

    def update(self, packet: LocalizationPacket) -> PoseEstimate:
        self._ensure_open()
        if self._published_raw is None or self._latency_queue is None:
            raise RuntimeError("online estimator provider must be reset before update")
        raw = self.plugin.update(packet)
        self._validate_estimate(raw, packet.timestamp_s)
        assert self._latest_plugin_raw is not None
        if np.any(raw.timestamp_s < self._latest_plugin_raw.timestamp_s):
            raise ValueError("online estimator timestamps must be monotonic")
        self._latest_plugin_raw = raw
        self._update_count += 1
        if self._update_count % self.cfg.update_interval_steps == 0:
            self._published_raw = raw
        self._latency_queue.append(self._published_raw)
        return self._apply_health(self._latency_queue[0], packet.timestamp_s)

    def close(self) -> None:
        if not self._closed:
            self.plugin.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("online estimator provider is closed")

    def _validate_estimate(
        self,
        estimate: PoseEstimate,
        query_timestamp_s: np.ndarray,
    ) -> None:
        if len(estimate.pose) != len(query_timestamp_s):
            raise ValueError("online estimator changed environment batch size")
        if (
            estimate.parent_frame != self.parent_frame
            or estimate.child_frame != self.child_frame
        ):
            raise ValueError("online estimator changed its frame contract")
        if np.any(estimate.timestamp_s > query_timestamp_s + 1.0e-9):
            raise ValueError("online estimator produced a future timestamp")

    def _apply_health(
        self,
        estimate: PoseEstimate,
        query_timestamp_s: np.ndarray,
    ) -> PoseEstimate:
        status = estimate.status.copy()
        dropout = self._rng.random(len(status)) < self.cfg.dropout_probability
        status[dropout] = PoseStatus.LOST
        if self.cfg.max_staleness_s is not None:
            age = query_timestamp_s - estimate.timestamp_s
            status[age > self.cfg.max_staleness_s] = PoseStatus.LOST
        return PoseEstimate(
            pose=estimate.pose,
            covariance=estimate.covariance,
            valid=np.isin(status, [PoseStatus.TRACKING, PoseStatus.DEGRADED]),
            status=status,
            timestamp_s=estimate.timestamp_s,
            parent_frame=self.parent_frame,
            child_frame=self.child_frame,
        )


def create_online_estimator_plugin(
    cfg: OnlineEstimatorCfg,
    *,
    parent_frame: str,
    child_frame: str,
) -> OnlineEstimatorPlugin:
    """Resolve one trusted ``module:factory`` plugin without core dependencies."""
    cfg.validate()
    module_name, _, symbol_name = cfg.plugin.partition(":")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, symbol_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot resolve online-estimator plugin {cfg.plugin!r}") from exc
    plugin = factory(cfg, parent_frame=parent_frame, child_frame=child_frame)
    for method_name in ("reset", "update", "close"):
        if not callable(getattr(plugin, method_name, None)):
            raise TypeError(f"online-estimator plugin is missing callable {method_name}")
    for frame_name in ("parent_frame", "child_frame"):
        if not isinstance(getattr(plugin, frame_name, None), str):
            raise TypeError(f"online-estimator plugin is missing string {frame_name}")
    return plugin


@dataclass
class LocalizationCfg:
    """Select and configure the navigation pose provider."""

    provider: str = "ground_truth"
    parent_frame: str = "map"
    child_frame: str = "base_link"
    noisy_pose: NoisyPoseCfg = field(default_factory=NoisyPoseCfg)
    dead_reckoning: DeadReckoningCfg = field(default_factory=DeadReckoningCfg)
    recorded_pose: RecordedPoseCfg = field(default_factory=RecordedPoseCfg)
    online_estimator: OnlineEstimatorCfg = field(default_factory=OnlineEstimatorCfg)

    def validate(self) -> None:
        if self.provider not in {
            "ground_truth",
            "noisy_pose",
            "dead_reckoning",
            "recorded_pose",
            "online_estimator",
        }:
            raise ValueError(
                "localization provider must be ground_truth, noisy_pose, "
                "dead_reckoning, recorded_pose, or online_estimator"
            )
        _validate_frame_pair(self.parent_frame, self.child_frame)
        self.noisy_pose.validate()
        self.dead_reckoning.validate()
        self.recorded_pose.validate()
        self.online_estimator.validate()
        if self.provider == "dead_reckoning" and self.parent_frame == "map":
            raise ValueError("dead-reckoning parent_frame must identify an odometry frame")
        if self.provider in {"ground_truth", "noisy_pose"} and self.parent_frame != "map":
            raise ValueError("truth-derived localization parent_frame must be map")
        if self.provider == "recorded_pose" and not self.recorded_pose.path:
            raise ValueError("recorded-pose provider requires recorded_pose.path")


def create_pose_provider(cfg: LocalizationCfg) -> PoseProvider:
    """Create one provider from validated task configuration."""
    cfg.validate()
    if cfg.provider == "ground_truth":
        return GroundTruthPoseProvider(
            parent_frame=cfg.parent_frame,
            child_frame=cfg.child_frame,
        )
    if cfg.provider == "noisy_pose":
        return NoisyPoseProvider(
            cfg.noisy_pose,
            parent_frame=cfg.parent_frame,
            child_frame=cfg.child_frame,
        )
    if cfg.provider == "dead_reckoning":
        return DeadReckoningPoseProvider(
            cfg.dead_reckoning,
            parent_frame=cfg.parent_frame,
            child_frame=cfg.child_frame,
        )
    if cfg.provider == "online_estimator":
        plugin = create_online_estimator_plugin(
            cfg.online_estimator,
            parent_frame=cfg.parent_frame,
            child_frame=cfg.child_frame,
        )
        return OnlineEstimatorPoseProvider(plugin, cfg.online_estimator)
    assert cfg.recorded_pose.path is not None
    stream = read_recorded_pose_stream(cfg.recorded_pose.path)
    if stream.parent_frame != cfg.parent_frame or stream.child_frame != cfg.child_frame:
        raise ValueError("recorded-pose stream frames do not match localization config")
    return RecordedPoseProvider(stream, cfg.recorded_pose)
