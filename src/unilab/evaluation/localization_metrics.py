"""Deterministic planar SLAM/localization trace metrics for navigation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from unilab.dtype_config import get_global_dtype
from unilab.envs.navigation.localization import PoseStatus

_TRACKING_STATUSES = np.array(
    [PoseStatus.TRACKING, PoseStatus.DEGRADED],
    dtype=np.uint8,
)


def _trace_sha256(payload: Mapping[str, Any]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "sha256"}
    encoded = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class LocalizationTrace:
    """Batched truth and estimate sequence on a common query-time axis."""

    query_timestamp_s: np.ndarray
    truth_pose: np.ndarray
    estimate_pose: np.ndarray
    estimate_covariance: np.ndarray
    estimate_status: np.ndarray
    estimate_timestamp_s: np.ndarray
    parent_frame: str = "map"
    child_frame: str = "base_link"

    def __post_init__(self) -> None:
        query = np.asarray(self.query_timestamp_s, dtype=np.float64)
        truth = np.asarray(self.truth_pose, dtype=get_global_dtype())
        estimate = np.asarray(self.estimate_pose, dtype=get_global_dtype())
        covariance = np.asarray(self.estimate_covariance, dtype=get_global_dtype())
        status = np.asarray(self.estimate_status, dtype=np.uint8)
        estimate_time = np.asarray(self.estimate_timestamp_s, dtype=np.float64)
        if query.ndim != 2 or query.shape[0] < 2 or query.shape[1] == 0:
            raise ValueError("localization query time must have shape (steps>=2, environments)")
        steps, environments = query.shape
        if truth.shape != (steps, environments, 3):
            raise ValueError("localization truth pose has the wrong shape")
        if estimate.shape != truth.shape:
            raise ValueError("localization estimate pose must match truth pose")
        if covariance.shape != (steps, environments, 3, 3):
            raise ValueError("localization covariance has the wrong shape")
        if status.shape != query.shape or estimate_time.shape != query.shape:
            raise ValueError("localization status/timestamp must match query time")
        arrays = (query, truth, estimate, covariance, estimate_time)
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("localization trace values must be finite")
        if np.any(query < 0.0) or np.any(estimate_time < 0.0):
            raise ValueError("localization trace timestamps must be non-negative")
        if np.any(np.diff(query, axis=0) <= 0.0):
            raise ValueError("localization query timestamps must increase")
        if np.any(estimate_time > query + 1.0e-9):
            raise ValueError("localization estimate timestamps cannot be in the future")
        valid_statuses = np.array([int(value) for value in PoseStatus], dtype=np.uint8)
        if not np.all(np.isin(status, valid_statuses)):
            raise ValueError("localization trace contains an unknown status")
        if not np.allclose(covariance, covariance.swapaxes(2, 3), atol=1.0e-6):
            raise ValueError("localization covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(covariance) < -1.0e-6):
            raise ValueError("localization covariance must be positive semidefinite")
        if not self.parent_frame or not self.child_frame or self.parent_frame == self.child_frame:
            raise ValueError("localization trace frames must be distinct and non-empty")
        object.__setattr__(self, "query_timestamp_s", query.copy())
        object.__setattr__(self, "truth_pose", truth.copy())
        object.__setattr__(self, "estimate_pose", estimate.copy())
        object.__setattr__(self, "estimate_covariance", covariance.copy())
        object.__setattr__(self, "estimate_status", status.copy())
        object.__setattr__(self, "estimate_timestamp_s", estimate_time.copy())

    @property
    def steps(self) -> int:
        return self.query_timestamp_s.shape[0]

    @property
    def environment_count(self) -> int:
        return self.query_timestamp_s.shape[1]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "pose_convention": "x_y_yaw_radians",
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "query_timestamp_s": self.query_timestamp_s.tolist(),
            "truth_pose": self.truth_pose.tolist(),
            "estimate_pose": self.estimate_pose.tolist(),
            "estimate_covariance": self.estimate_covariance.tolist(),
            "estimate_status": self.estimate_status.tolist(),
            "estimate_timestamp_s": self.estimate_timestamp_s.tolist(),
        }
        payload["sha256"] = _trace_sha256(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalizationTrace:
        if payload.get("schema_version") != 1:
            raise ValueError("localization trace schema_version must be 1")
        if payload.get("pose_convention") != "x_y_yaw_radians":
            raise ValueError("localization trace pose convention is unsupported")
        expected_hash = payload.get("sha256")
        if not isinstance(expected_hash, str) or expected_hash != _trace_sha256(payload):
            raise ValueError("localization trace sha256 does not match its content")
        return cls(
            query_timestamp_s=payload["query_timestamp_s"],
            truth_pose=payload["truth_pose"],
            estimate_pose=payload["estimate_pose"],
            estimate_covariance=payload["estimate_covariance"],
            estimate_status=payload["estimate_status"],
            estimate_timestamp_s=payload["estimate_timestamp_s"],
            parent_frame=str(payload["parent_frame"]),
            child_frame=str(payload["child_frame"]),
        )


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"mean": None, "std": None, "sample_count": 0}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "sample_count": len(finite),
    }


def _error_summary(values: np.ndarray) -> dict[str, float | int | None]:
    summary = _summary(values)
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    summary["rmse"] = (
        float(np.sqrt(np.mean(np.square(finite)))) if len(finite) else None
    )
    return summary


def _relative_motion(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = pose[1:, :, :2] - pose[:-1, :, :2]
    yaw = pose[:-1, :, 2]
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    local_x = cosine * delta[:, :, 0] + sine * delta[:, :, 1]
    local_y = -sine * delta[:, :, 0] + cosine * delta[:, :, 1]
    translation = np.stack((local_x, local_y), axis=2)
    yaw_delta = _wrap_angle(pose[1:, :, 2] - pose[:-1, :, 2])
    return translation, yaw_delta


def evaluate_localization_trace(trace: LocalizationTrace) -> dict[str, Any]:
    """Compute navigation-independent localization quality and health metrics."""
    valid = np.isin(trace.estimate_status, _TRACKING_STATUSES)
    position_error = np.linalg.norm(
        trace.estimate_pose[:, :, :2] - trace.truth_pose[:, :, :2],
        axis=2,
    )
    yaw_error = np.abs(
        _wrap_angle(trace.estimate_pose[:, :, 2] - trace.truth_pose[:, :, 2])
    )
    ate_translation = _error_summary(position_error[valid])
    ate_yaw = _error_summary(yaw_error[valid])

    truth_translation, truth_yaw = _relative_motion(trace.truth_pose)
    estimate_translation, estimate_yaw = _relative_motion(trace.estimate_pose)
    valid_pairs = valid[1:] & valid[:-1]
    rpe_translation_error = np.linalg.norm(
        estimate_translation - truth_translation,
        axis=2,
    )
    rpe_yaw_error = np.abs(_wrap_angle(estimate_yaw - truth_yaw))

    truth_step_distance = np.linalg.norm(
        trace.truth_pose[1:, :, :2] - trace.truth_pose[:-1, :, :2],
        axis=2,
    )
    path_length = np.sum(truth_step_distance, axis=0)
    drift = np.full(trace.environment_count, np.nan, dtype=np.float64)
    for env_index in range(trace.environment_count):
        valid_indices = np.flatnonzero(valid[:, env_index])
        if len(valid_indices) and path_length[env_index] > np.finfo(np.float32).eps:
            drift[env_index] = (
                position_error[valid_indices[-1], env_index] / path_length[env_index]
            )

    duration = trace.query_timestamp_s[-1] - trace.query_timestamp_s[0]
    update_rates = np.full(trace.environment_count, np.nan, dtype=np.float64)
    for env_index in range(trace.environment_count):
        timestamps = trace.estimate_timestamp_s[valid[:, env_index], env_index]
        if len(timestamps) and duration[env_index] > 0.0:
            update_count = np.count_nonzero(np.diff(timestamps) > 1.0e-9)
            update_rates[env_index] = update_count / duration[env_index]

    latency = trace.query_timestamp_s - trace.estimate_timestamp_s
    error_vector = trace.estimate_pose - trace.truth_pose
    error_vector[:, :, 2] = _wrap_angle(error_vector[:, :, 2])
    flat_error = error_vector.reshape(-1, 3)
    flat_covariance = trace.estimate_covariance.reshape(-1, 3, 3)
    flat_valid = valid.reshape(-1)
    eigenvalues = np.linalg.eigvalsh(flat_covariance)
    calibrated = flat_valid & np.all(eigenvalues > 1.0e-9, axis=1)
    nees = np.empty(0, dtype=np.float64)
    if np.any(calibrated):
        inverse = np.linalg.inv(flat_covariance[calibrated])
        errors = flat_error[calibrated]
        nees = np.einsum("ni,nij,nj->n", errors, inverse, errors)

    return {
        "schema_version": 1,
        "config": {
            "steps": trace.steps,
            "environments": trace.environment_count,
            "parent_frame": trace.parent_frame,
            "child_frame": trace.child_frame,
            "rpe_delta_steps": 1,
            "covariance_95_percent_chi_square_threshold": 7.814727903251179,
        },
        "metrics": {
            "ate_translation_m": ate_translation,
            "ate_yaw_rad": ate_yaw,
            "rpe_translation_m": _error_summary(rpe_translation_error[valid_pairs]),
            "rpe_yaw_rad": _error_summary(rpe_yaw_error[valid_pairs]),
            "drift_per_meter": _summary(drift),
            "truth_path_length_m": _summary(path_length),
            "update_rate_hz": _summary(update_rates),
            "latency_s": _summary(latency[valid]),
            "dropout_rate": float(1.0 - np.mean(valid)),
            "covariance_nees": _summary(nees),
            "covariance_95_percent_coverage": (
                float(np.mean(nees <= 7.814727903251179)) if len(nees) else None
            ),
            "covariance_calibrated_sample_count": len(nees),
        },
        "trace_sha256": trace.to_dict()["sha256"],
    }


def format_localization_metrics(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]

    def value(name: str, field: str = "mean") -> str:
        value = metrics[name][field]
        return "n/a" if value is None else f"{value:.6f}"

    return (
        "ATE(m) RPE(m) drift/m update(Hz) latency(s) dropout NEES coverage95\n"
        f"{value('ate_translation_m', 'rmse')} "
        f"{value('rpe_translation_m', 'rmse')} "
        f"{value('drift_per_meter')} {value('update_rate_hz')} "
        f"{value('latency_s')} {metrics['dropout_rate']:.6f} "
        f"{value('covariance_nees')} "
        f"{metrics['covariance_95_percent_coverage']}"
    )


def read_localization_trace(input_path: str | Path) -> LocalizationTrace:
    path = Path(input_path)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"localization trace does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"localization trace is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("localization trace JSON must contain an object")
    return LocalizationTrace.from_dict(payload)


def write_localization_trace(trace: LocalizationTrace, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def write_localization_metrics(report: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
