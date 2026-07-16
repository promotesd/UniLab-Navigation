"""Analytical and real MuJoCo tests for planar localization quality metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from unilab.envs.navigation import (
    DeadReckoningCfg,
    DeadReckoningPoseProvider,
    PoseStatus,
)
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
    HeuristicPointGoalPolicy,
)
from unilab.evaluation import (
    LocalizationTrace,
    evaluate_localization_trace,
    read_localization_trace,
    write_localization_trace,
)


def trace(
    *,
    estimate_x: np.ndarray | None = None,
    status: np.ndarray | None = None,
    estimate_time: np.ndarray | None = None,
) -> LocalizationTrace:
    query = np.array([[0.0], [1.0], [2.0]])
    truth = np.zeros((3, 1, 3), dtype=np.float32)
    truth[:, 0, 0] = [0.0, 1.0, 2.0]
    estimate = truth.copy()
    if estimate_x is not None:
        estimate[:, 0, 0] = estimate_x
    covariance = np.broadcast_to(np.eye(3), (3, 1, 3, 3)).copy()
    statuses = (
        np.full((3, 1), PoseStatus.TRACKING, dtype=np.uint8)
        if status is None
        else status
    )
    return LocalizationTrace(
        query_timestamp_s=query,
        truth_pose=truth,
        estimate_pose=estimate,
        estimate_covariance=covariance,
        estimate_status=statuses,
        estimate_timestamp_s=query if estimate_time is None else estimate_time,
    )


def test_identity_trace_has_zero_ate_rpe_drift_and_full_calibration() -> None:
    report = evaluate_localization_trace(trace())
    metrics = report["metrics"]
    assert metrics["ate_translation_m"]["rmse"] == 0.0
    assert metrics["ate_yaw_rad"]["rmse"] == 0.0
    assert metrics["rpe_translation_m"]["rmse"] == 0.0
    assert metrics["rpe_yaw_rad"]["rmse"] == 0.0
    assert metrics["drift_per_meter"]["mean"] == 0.0
    assert metrics["update_rate_hz"]["mean"] == 1.0
    assert metrics["latency_s"]["mean"] == 0.0
    assert metrics["dropout_rate"] == 0.0
    assert metrics["covariance_nees"]["mean"] == 0.0
    assert metrics["covariance_95_percent_coverage"] == 1.0


def test_constant_offset_affects_ate_and_drift_but_not_rpe() -> None:
    report = evaluate_localization_trace(trace(estimate_x=np.array([1.0, 2.0, 3.0])))
    metrics = report["metrics"]
    assert metrics["ate_translation_m"]["rmse"] == 1.0
    assert metrics["rpe_translation_m"]["rmse"] == 0.0
    assert metrics["drift_per_meter"]["mean"] == 0.5
    assert metrics["covariance_nees"]["mean"] == 1.0


def test_scale_error_dropout_latency_and_update_rate_are_reported() -> None:
    statuses = np.full((3, 1), PoseStatus.TRACKING, dtype=np.uint8)
    statuses[1, 0] = PoseStatus.LOST
    query = np.array([[1.0], [2.0], [3.0]])
    delayed = np.array([[0.8], [1.8], [2.8]])
    source = trace(
        estimate_x=np.array([0.0, 2.0, 4.0]),
        status=statuses,
    )
    delayed_trace = LocalizationTrace(
        query_timestamp_s=query,
        truth_pose=source.truth_pose,
        estimate_pose=source.estimate_pose,
        estimate_covariance=source.estimate_covariance,
        estimate_status=source.estimate_status,
        estimate_timestamp_s=delayed,
    )
    metrics = evaluate_localization_trace(delayed_trace)["metrics"]
    assert metrics["ate_translation_m"]["rmse"] == pytest.approx(np.sqrt(2.0))
    assert metrics["rpe_translation_m"]["sample_count"] == 0
    assert metrics["dropout_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["latency_s"]["mean"] == pytest.approx(0.2)
    assert metrics["update_rate_hz"]["mean"] == pytest.approx(0.5)


def test_localization_trace_round_trip_hash_and_tamper_detection(tmp_path) -> None:
    source = trace()
    path = write_localization_trace(source, tmp_path / "trace.json")
    restored = read_localization_trace(path)
    np.testing.assert_array_equal(restored.truth_pose, source.truth_pose)
    payload = json.loads(path.read_text())
    payload["estimate_pose"][0][0][0] = 99.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sha256"):
        read_localization_trace(path)


def test_trace_rejects_future_and_nonmonotonic_timestamps() -> None:
    source = trace()
    with pytest.raises(ValueError, match="increase"):
        LocalizationTrace(
            query_timestamp_s=np.array([[0.0], [1.0], [1.0]]),
            truth_pose=source.truth_pose,
            estimate_pose=source.estimate_pose,
            estimate_covariance=source.estimate_covariance,
            estimate_status=source.estimate_status,
            estimate_timestamp_s=source.estimate_timestamp_s,
        )
    with pytest.raises(ValueError, match="future"):
        LocalizationTrace(
            query_timestamp_s=source.query_timestamp_s,
            truth_pose=source.truth_pose,
            estimate_pose=source.estimate_pose,
            estimate_covariance=source.estimate_covariance,
            estimate_status=source.estimate_status,
            estimate_timestamp_s=source.query_timestamp_s + 0.1,
        )


def test_real_mujoco_dead_reckoning_trace_reports_nonzero_error() -> None:
    provider = DeadReckoningPoseProvider(
        DeadReckoningCfg(
            seed=901,
            linear_velocity_noise_std=0.02,
            angular_velocity_noise_std=0.01,
            linear_velocity_bias=0.05,
            angular_velocity_bias=0.01,
        ),
        parent_frame="map",
        child_frame="base_link",
    )
    env = DiffDrivePointGoalMujocoEnv(
        cfg=DiffDrivePointGoalCfg(),
        num_envs=4,
        pose_provider=provider,
    )
    state = env.reset_to_initial_conditions(
        np.zeros((4, 3), dtype=np.float32),
        np.tile(np.array([[2.0, 0.0]], dtype=np.float32), (4, 1)),
    )
    policy = HeuristicPointGoalPolicy()
    query = [env.localization_time_s.copy()]
    truth = [state.info["robot_state"].copy()]
    estimate = [state.info["localization_pose"].copy()]
    covariance = [state.info["localization_covariance"].copy()]
    status = [state.info["localization_status"].copy()]
    estimate_time = [state.info["localization_timestamp_s"].copy()]
    for _ in range(30):
        state = env.step(policy(state.obs))
        query.append(env.localization_time_s.copy())
        truth.append(state.info["robot_state"].copy())
        estimate.append(state.info["localization_pose"].copy())
        covariance.append(state.info["localization_covariance"].copy())
        status.append(state.info["localization_status"].copy())
        estimate_time.append(state.info["localization_timestamp_s"].copy())
    report = evaluate_localization_trace(
        LocalizationTrace(
            query_timestamp_s=np.asarray(query),
            truth_pose=np.asarray(truth),
            estimate_pose=np.asarray(estimate),
            estimate_covariance=np.asarray(covariance),
            estimate_status=np.asarray(status),
            estimate_timestamp_s=np.asarray(estimate_time),
        )
    )
    metrics = report["metrics"]
    assert metrics["ate_translation_m"]["rmse"] > 0.01
    assert metrics["rpe_translation_m"]["rmse"] > 0.0
    assert metrics["dropout_rate"] == 0.0
    assert metrics["covariance_calibrated_sample_count"] > 0
    env.close()
