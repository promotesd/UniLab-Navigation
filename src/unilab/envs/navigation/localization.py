"""Backend-independent localization provider contracts for navigation."""

from __future__ import annotations

from dataclasses import dataclass
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
        source_pose: np.ndarray,
        timestamp_s: np.ndarray,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        """Reset provider state for explicit source poses."""

    def update(self, source_pose: np.ndarray, timestamp_s: np.ndarray) -> PoseEstimate:
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
        source_pose: np.ndarray,
        timestamp_s: np.ndarray,
        env_indices: np.ndarray,
    ) -> PoseEstimate:
        timestamp = np.asarray(timestamp_s, dtype=np.float64)
        indices = np.asarray(env_indices, dtype=np.int32)
        if self._last_timestamp_s is not None:
            if timestamp.shape != self._last_timestamp_s.shape:
                raise ValueError("timestamp batch shape changed")
            unchanged = np.ones(len(timestamp), dtype=bool)
            unchanged[indices] = False
            if np.any(timestamp[unchanged] < self._last_timestamp_s[unchanged]):
                raise ValueError("timestamps moved backward outside reset environments")
        self._last_timestamp_s = timestamp.copy()
        return self.update(source_pose, timestamp_s)

    def update(self, source_pose: np.ndarray, timestamp_s: np.ndarray) -> PoseEstimate:
        pose = np.asarray(source_pose, dtype=get_global_dtype())
        timestamp = np.asarray(timestamp_s, dtype=np.float64)
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
