"""Backend-independent localization provider contracts for navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

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


@dataclass
class LocalizationCfg:
    """Select and configure the navigation pose provider."""

    provider: str = "ground_truth"
    parent_frame: str = "map"
    child_frame: str = "base_link"
    noisy_pose: NoisyPoseCfg = field(default_factory=NoisyPoseCfg)
    dead_reckoning: DeadReckoningCfg = field(default_factory=DeadReckoningCfg)

    def validate(self) -> None:
        if self.provider not in {"ground_truth", "noisy_pose", "dead_reckoning"}:
            raise ValueError(
                "localization provider must be ground_truth, noisy_pose, or dead_reckoning"
            )
        _validate_frame_pair(self.parent_frame, self.child_frame)
        self.noisy_pose.validate()
        self.dead_reckoning.validate()
        if self.provider == "dead_reckoning" and self.parent_frame == "map":
            raise ValueError("dead-reckoning parent_frame must identify an odometry frame")
        if self.provider != "dead_reckoning" and self.parent_frame != "map":
            raise ValueError("truth-derived localization parent_frame must be map")


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
    return DeadReckoningPoseProvider(
        cfg.dead_reckoning,
        parent_frame=cfg.parent_frame,
        child_frame=cfg.child_frame,
    )
