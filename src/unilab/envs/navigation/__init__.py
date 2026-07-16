"""Navigation environment registry bootstrap."""

from .localization import (
    DeadReckoningCfg,
    DeadReckoningPoseProvider,
    GroundTruthPosePacket,
    GroundTruthPoseProvider,
    LocalizationCfg,
    LocalizationPacket,
    NoisyPoseCfg,
    NoisyPoseProvider,
    PoseEstimate,
    PoseProvider,
    PoseStatus,
    WheelOdometryPacket,
    create_pose_provider,
)

__unilab_registry_modules__ = (
    "unilab.envs.navigation.diff_drive",
)

__all__ = [
    "GroundTruthPoseProvider",
    "GroundTruthPosePacket",
    "DeadReckoningCfg",
    "DeadReckoningPoseProvider",
    "LocalizationPacket",
    "LocalizationCfg",
    "NoisyPoseCfg",
    "NoisyPoseProvider",
    "PoseEstimate",
    "PoseProvider",
    "PoseStatus",
    "WheelOdometryPacket",
    "create_pose_provider",
]
