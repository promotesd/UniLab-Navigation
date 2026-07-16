"""Navigation environment registry bootstrap."""

from .localization import (
    GroundTruthPoseProvider,
    PoseEstimate,
    PoseProvider,
    PoseStatus,
)

__unilab_registry_modules__ = (
    "unilab.envs.navigation.diff_drive",
)

__all__ = [
    "GroundTruthPoseProvider",
    "PoseEstimate",
    "PoseProvider",
    "PoseStatus",
]
