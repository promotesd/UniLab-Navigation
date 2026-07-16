"""Transport-free TurtleBot 4 command, scan, and episode bridge tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from unilab.bridges.ros2 import (
    EpisodeResetRequest,
    EpisodeState,
    LaserScanBridgeCfg,
    Ros2LocalizationBridge,
    Ros2LocalizationBridgeCfg,
    Ros2NodeTransport,
    TurtleBot4CommandBridge,
    TurtleBot4EpisodeBridge,
    TurtleBot4LaserScanBridge,
    turtlebot4_real_robot_profile,
    turtlebot4_simulation_profile,
)
from unilab.envs.navigation.diff_drive import (
    DiffDrivePointGoalCfg,
    DiffDrivePointGoalMujocoEnv,
)


@dataclass
class FakeTransport:
    commands: list = field(default_factory=list)
    resets: list = field(default_factory=list)

    def publish_twist(self, command) -> None:
        self.commands.append(command)

    def request_reset(self, service, request) -> None:
        self.resets.append((service, request))


def laser_scan(
    *,
    timestamp_s: float = 1.0,
    frame_id: str = "rplidar_link",
    beam_count: int = 360,
):
    total_nanoseconds = int(round(timestamp_s * 1.0e9))
    sec, nanosec = divmod(total_nanoseconds, 1_000_000_000)
    ranges = np.linspace(0.1, 4.0, beam_count)
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
            frame_id=frame_id,
        ),
        angle_min=-np.pi,
        angle_increment=2.0 * np.pi / beam_count,
        range_min=0.05,
        range_max=5.0,
        ranges=ranges.tolist(),
    )


def test_simulation_and_real_robot_profiles_are_explicitly_separate() -> None:
    simulation = turtlebot4_simulation_profile("robot1")
    real_robot = turtlebot4_real_robot_profile("robot1")
    assert simulation.mode == "simulation"
    assert simulation.reset_service == "/robot1/reset_episode"
    assert simulation.command_topic == "/robot1/cmd_vel"
    assert real_robot.mode == "real_robot"
    assert real_robot.reset_service is None
    assert real_robot.max_linear_velocity < simulation.max_linear_velocity
    assert real_robot.command_timeout_s < simulation.command_timeout_s


def test_twist_mapping_matches_point_goal_actions_and_watchdog_stops() -> None:
    bridge = TurtleBot4CommandBridge(turtlebot4_simulation_profile())
    command = bridge.submit_normalized_action(
        np.array([0.0, 0.5]),
        timestamp_s=1.0,
    )
    assert command.linear_x == pytest.approx(0.25)
    assert command.angular_z == pytest.approx(0.75)
    assert not command.timed_out
    assert bridge.command(now_s=1.2) == command
    timed_out = bridge.command(now_s=1.3)
    assert timed_out.timed_out
    assert timed_out.linear_x == 0.0
    assert timed_out.angular_z == 0.0
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        bridge.submit_normalized_action(np.array([1.1, 0.0]), timestamp_s=2.0)


def test_laser_scan_resamples_fixed_shape_and_marks_invalid_returns() -> None:
    profile = turtlebot4_simulation_profile()
    bridge = TurtleBot4LaserScanBridge(
        profile,
        LaserScanBridgeCfg(beam_count=16),
    )
    message = laser_scan()
    message.ranges[0] = float("inf")
    packet = bridge.convert(message, received_time_s=1.1)
    assert packet.ranges.shape == (16,)
    assert packet.normalized_ranges.shape == (16,)
    assert packet.valid.shape == (16,)
    assert not packet.valid[0]
    assert np.all(packet.ranges >= 0.05)
    assert np.all(packet.ranges <= 5.0)
    assert np.all(np.isfinite(packet.normalized_ranges))


def test_laser_scan_rejects_stale_future_frame_and_coverage_mismatch() -> None:
    bridge = TurtleBot4LaserScanBridge(
        turtlebot4_real_robot_profile(),
        LaserScanBridgeCfg(beam_count=16),
    )
    with pytest.raises(ValueError, match="stale"):
        bridge.convert(laser_scan(timestamp_s=1.0), received_time_s=1.31)
    with pytest.raises(ValueError, match="future"):
        bridge.convert(laser_scan(timestamp_s=2.0), received_time_s=1.0)
    with pytest.raises(ValueError, match="frame"):
        bridge.convert(laser_scan(frame_id="laser"), received_time_s=1.0)
    narrow = laser_scan(beam_count=8)
    narrow.angle_min = -0.2
    narrow.angle_increment = 0.05
    with pytest.raises(ValueError, match="coverage"):
        bridge.convert(narrow, received_time_s=1.0)


def test_episode_bridge_publishes_commands_watchdog_stop_and_sim_reset() -> None:
    profile = turtlebot4_simulation_profile()
    transport = FakeTransport()
    bridge = TurtleBot4EpisodeBridge(profile, transport)
    bridge.start("episode-1")
    assert bridge.state == EpisodeState.RUNNING
    bridge.publish_action(np.array([1.0, -1.0]), timestamp_s=0.0)
    assert transport.commands[-1].linear_x == profile.max_linear_velocity
    bridge.enforce_watchdog(now_s=0.3)
    assert transport.commands[-1].timed_out
    request = EpisodeResetRequest("episode-2", timestamp_s=0.4, seed=7)
    bridge.reset(request)
    assert transport.commands[-1].linear_x == 0.0
    assert transport.resets == [(profile.reset_service, request)]
    assert bridge.state == EpisodeState.IDLE


def test_real_robot_reset_requires_operator_confirmation() -> None:
    transport = FakeTransport()
    bridge = TurtleBot4EpisodeBridge(turtlebot4_real_robot_profile(), transport)
    request = EpisodeResetRequest("real-1", timestamp_s=1.0)
    with pytest.raises(PermissionError, match="operator"):
        bridge.reset(request)
    bridge.reset(request, operator_confirmed=True)
    assert bridge.state == EpisodeState.IDLE
    assert not transport.resets


class FakePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class FakeClient:
    def __init__(self) -> None:
        self.requests = []

    def call_async(self, request) -> None:
        self.requests.append(request)


class FakeNode:
    def __init__(self) -> None:
        self.publisher = FakePublisher()
        self.client = FakeClient()

    def create_publisher(self, message_type, topic, depth):
        self.publisher_args = (message_type, topic, depth)
        return self.publisher

    def create_client(self, service_type, service):
        self.client_args = (service_type, service)
        return self.client


class FakeTwist:
    def __init__(self) -> None:
        self.linear = SimpleNamespace(x=0.0)
        self.angular = SimpleNamespace(z=0.0)


class FakeReset:
    class Request:
        def __init__(self) -> None:
            self.episode_id = ""
            self.seed = 0


def test_injected_node_transport_requires_no_rclpy_import() -> None:
    profile = turtlebot4_simulation_profile()
    node = FakeNode()
    transport = Ros2NodeTransport(
        node,
        profile,
        twist_message_type=FakeTwist,
        reset_service_type=FakeReset,
    )
    episode = TurtleBot4EpisodeBridge(profile, transport)
    episode.start("episode")
    episode.publish_action(np.array([0.0, 0.5]), timestamp_s=0.0)
    assert node.publisher.messages[-1].linear.x == pytest.approx(0.25)
    assert node.publisher.messages[-1].angular.z == pytest.approx(0.75)
    episode.reset(EpisodeResetRequest("next", timestamp_s=0.1, seed=9))
    assert node.client.requests[-1].episode_id == "next"
    assert node.client.requests[-1].seed == 9


def test_real_mujoco_velocity_contract_matches_twist_bridge() -> None:
    profile = turtlebot4_simulation_profile()
    bridge = TurtleBot4CommandBridge(profile)
    cfg = DiffDrivePointGoalCfg(
        max_linear_velocity=profile.max_linear_velocity,
        max_angular_velocity=profile.max_angular_velocity,
    )
    env = DiffDrivePointGoalMujocoEnv(cfg=cfg, num_envs=1)
    env.reset_to_initial_conditions(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    action = np.array([0.0, 0.5], dtype=np.float32)
    command = bridge.submit_normalized_action(action, timestamp_s=0.0)
    env.step(action[None, :])
    np.testing.assert_allclose(
        env.velocity_commands,
        [[command.linear_x, command.angular_z]],
    )
    env.close()


def test_transport_free_turtlebot4_command_odom_scan_reset_flow() -> None:
    profile = turtlebot4_simulation_profile("tb4")
    transport = FakeTransport()
    episode = TurtleBot4EpisodeBridge(profile, transport)
    episode.start("flow-1")
    command = episode.publish_action(np.array([0.2, -0.4]), timestamp_s=1.0)
    assert command.linear_x > 0.0

    scan = TurtleBot4LaserScanBridge(
        profile,
        LaserScanBridgeCfg(beam_count=16),
    ).convert(
        laser_scan(timestamp_s=1.0, frame_id=profile.scan_frame),
        received_time_s=1.0,
    )
    assert scan.valid.shape == (16,)

    localization = Ros2LocalizationBridge(
        Ros2LocalizationBridgeCfg(
            parent_frame=profile.parent_frame,
            child_frame=profile.child_frame,
        ),
        1,
    )
    odom = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=1, nanosec=0),
            frame_id=profile.parent_frame,
        ),
        child_frame_id=profile.child_frame,
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=SimpleNamespace(x=0.5, y=0.0, z=0.0),
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            covariance=np.zeros(36).tolist(),
        ),
    )
    localization.ingest_odometry(0, odom, received_time_s=1.0)
    estimate = localization.estimate(np.array([1.0]))
    np.testing.assert_allclose(estimate.pose[0], [0.5, 0.0, 0.0])

    reset = EpisodeResetRequest("flow-2", timestamp_s=1.1, seed=5)
    episode.reset(reset)
    assert transport.commands[-1].linear_x == 0.0
    assert transport.resets[-1] == (profile.reset_service, reset)
