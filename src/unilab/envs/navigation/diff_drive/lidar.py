"""Backend-independent vectorized planar LiDAR geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unilab.dtype_config import get_global_dtype


@dataclass
class PlanarLidarCfg:
    """Fixed-shape planar LiDAR configuration."""

    beam_count: int = 16
    angle_min: float = -np.pi
    angle_max: float = np.pi
    min_range: float = 0.05
    max_range: float = 5.0
    noise_std: float = 0.0
    noise_seed: int = 1

    def validate(self) -> None:
        if self.beam_count <= 0:
            raise ValueError("beam_count must be positive")
        if not np.isfinite(self.angle_min) or not np.isfinite(self.angle_max):
            raise ValueError("LiDAR angles must be finite")
        if self.angle_max <= self.angle_min:
            raise ValueError("angle_max must be greater than angle_min")
        if not np.isfinite(self.min_range) or not np.isfinite(self.max_range):
            raise ValueError("LiDAR ranges must be finite")
        if self.min_range < 0.0:
            raise ValueError("min_range must be non-negative")
        if self.max_range <= self.min_range:
            raise ValueError("max_range must be greater than min_range")
        if not np.isfinite(self.noise_std) or self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        if self.noise_seed < 0:
            raise ValueError("noise_seed must be non-negative")


class PlanarLidarProvider:
    """Compute clipped planar ray ranges against axis-aligned boxes."""

    def __init__(self, cfg: PlanarLidarCfg) -> None:
        cfg.validate()
        self.cfg = cfg
        self._beam_angles = np.linspace(
            cfg.angle_min,
            cfg.angle_max,
            cfg.beam_count,
            endpoint=False,
            dtype=np.float64,
        )
        self._rng = np.random.default_rng(cfg.noise_seed)

    @property
    def beam_angles(self) -> np.ndarray:
        """Return beam angles relative to the robot heading in radians."""
        return self._beam_angles.copy()

    def scan(
        self,
        robot_states: np.ndarray,
        obstacle_centers: np.ndarray,
        obstacle_half_extents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return physical and normalized ranges for every environment and beam.

        Obstacle arrays may be shared with shape ``(obstacles, 2)`` or batched
        with shape ``(environments, obstacles, 2)``. A normalized range of zero
        denotes ``min_range`` and one denotes ``max_range``. All rays are valid:
        a ray without a hit reports ``max_range``.
        """
        states = np.asarray(robot_states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError("robot_states must have shape (environments, 3)")
        if not np.all(np.isfinite(states)):
            raise ValueError("robot_states must contain only finite values")

        centers = self._broadcast_obstacles(obstacle_centers, len(states), "obstacle_centers")
        half_extents = self._broadcast_obstacles(
            obstacle_half_extents,
            len(states),
            "obstacle_half_extents",
        )
        if centers.shape != half_extents.shape:
            raise ValueError("obstacle_centers and obstacle_half_extents must match")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(half_extents)):
            raise ValueError("obstacle geometry must contain only finite values")
        if np.any(half_extents <= 0.0):
            raise ValueError("obstacle_half_extents must be positive")
        if centers.shape[1] == 0:
            ranges = np.full(
                (len(states), self.cfg.beam_count),
                self.cfg.max_range,
                dtype=np.float64,
            )
            return self._finalize_ranges(ranges)

        world_angles = states[:, 2, None] + self._beam_angles[None, :]
        directions = np.stack((np.cos(world_angles), np.sin(world_angles)), axis=-1)
        origins = states[:, None, None, :2]
        lower = centers[:, None, :, :] - half_extents[:, None, :, :]
        upper = centers[:, None, :, :] + half_extents[:, None, :, :]
        ray_directions = directions[:, :, None, :]

        parallel = np.abs(ray_directions) <= np.finfo(np.float64).eps
        safe_directions = np.where(parallel, 1.0, ray_directions)
        first = (lower - origins) / safe_directions
        second = (upper - origins) / safe_directions
        near = np.minimum(first, second)
        far = np.maximum(first, second)

        inside_slab = (origins >= lower) & (origins <= upper)
        near = np.where(parallel & inside_slab, -np.inf, near)
        far = np.where(parallel & inside_slab, np.inf, far)
        near = np.where(parallel & ~inside_slab, np.inf, near)
        far = np.where(parallel & ~inside_slab, -np.inf, far)

        entering = np.max(near, axis=-1)
        exiting = np.min(far, axis=-1)
        hit = (exiting >= np.maximum(entering, self.cfg.min_range)) & (exiting >= 0.0)
        candidates = np.where(hit, np.maximum(entering, self.cfg.min_range), np.inf)
        ranges = np.min(candidates, axis=2)
        ranges = np.minimum(ranges, self.cfg.max_range)

        return self._finalize_ranges(ranges)

    def _finalize_ranges(self, ranges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.noise_std > 0.0:
            ranges = ranges + self._rng.normal(
                0.0,
                self.cfg.noise_std,
                size=ranges.shape,
            )
        ranges = np.clip(ranges, self.cfg.min_range, self.cfg.max_range)
        normalized = (ranges - self.cfg.min_range) / (
            self.cfg.max_range - self.cfg.min_range
        )
        dtype = get_global_dtype()
        return ranges.astype(dtype), normalized.astype(dtype)

    @staticmethod
    def _broadcast_obstacles(
        values: np.ndarray,
        num_envs: int,
        name: str,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 2 and array.shape[1] == 2:
            return np.broadcast_to(array, (num_envs,) + array.shape)
        if array.ndim == 3 and array.shape[0] == num_envs and array.shape[2] == 2:
            return array
        raise ValueError(
            f"{name} must have shape (obstacles, 2) or "
            f"(environments, obstacles, 2)"
        )
