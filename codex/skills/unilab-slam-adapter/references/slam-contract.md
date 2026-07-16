# SLAM and Localization Contract

## Minimal estimate

A localization estimate should contain:

```python
@dataclass(frozen=True)
class PoseEstimate:
    timestamp_sec: float
    parent_frame: str
    child_frame: str
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    covariance: np.ndarray | None
    status: EstimateStatus
```

For planar navigation, derive `[x, y, yaw]` from the SE(3) estimate at the observation boundary. Preserve the full estimate for logging and future 3D tasks.

## Sensor packets

Prefer typed packets:

- `LidarScan`;
- `PointCloud`;
- `ImuSample`;
- `WheelOdometry`;
- `GroundTruthPose`.

Every packet contains a timestamp and frame.

## Provider protocol

```python
class LocalizationProvider(Protocol):
    def reset(self, env_indices: np.ndarray) -> None: ...
    def update(self, packet_batch: SensorPacketBatch) -> None: ...
    def get_estimate(self) -> PoseEstimateBatch: ...
```

Batch shape must include the environment dimension.

## Offline replay

An offline provider must define:

- trajectory file format;
- timestamp interpolation policy;
- out-of-range behavior;
- frame transform;
- missing sample behavior.

Never match samples only by array index when timestamps exist.

## ROS2 boundary

ROS2 conversion belongs in a bridge package. Core environment code should not import `rclpy`.

Validate:

- `header.stamp`;
- `frame_id`;
- child frame;
- TF availability;
- QoS;
- message age.

## Map contract

Separate:

- occupancy/grid or geometric map representation;
- localization estimate;
- navigation goal;
- local sensor observation.

Do not make a reward function own the map.
