"""Tests for backend-independent planar LiDAR geometry."""

import numpy as np
import pytest

from unilab.envs.navigation.diff_drive import PlanarLidarCfg, PlanarLidarProvider


def four_beam_provider(**overrides: float | int) -> PlanarLidarProvider:
    values = {
        "beam_count": 4,
        "angle_min": 0.0,
        "angle_max": 2.0 * np.pi,
        "min_range": 0.1,
        "max_range": 5.0,
        **overrides,
    }
    return PlanarLidarProvider(PlanarLidarCfg(**values))


def test_axis_aligned_box_ranges_are_analytical_and_clipped() -> None:
    provider = four_beam_provider()
    ranges, normalized = provider.scan(
        np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[2.0, 0.0]], dtype=np.float32),
        np.array([[0.5, 0.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(ranges, [[1.5, 5.0, 5.0, 5.0]], atol=1.0e-6)
    np.testing.assert_allclose(normalized[0, 0], (1.5 - 0.1) / (5.0 - 0.1))
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))


def test_heading_rotates_beams_without_changing_world_geometry() -> None:
    provider = four_beam_provider()
    ranges, _ = provider.scan(
        np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2]],
            dtype=np.float32,
        ),
        np.array([[2.0, 0.0]], dtype=np.float32),
        np.array([[0.5, 0.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(ranges[0], [1.5, 5.0, 5.0, 5.0], atol=1.0e-6)
    np.testing.assert_allclose(ranges[1], [5.0, 5.0, 5.0, 1.5], atol=1.0e-6)


def test_batched_obstacle_layouts_produce_fixed_finite_shape() -> None:
    provider = four_beam_provider()
    ranges, normalized = provider.scan(
        np.zeros((3, 3), dtype=np.float32),
        np.array([[[2.0, 0.0]], [[20.0, 0.0]], [[0.0, 2.0]]]),
        np.full((3, 1, 2), 0.5),
    )
    assert ranges.shape == normalized.shape == (3, 4)
    assert np.all(np.isfinite(ranges))
    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(ranges[1], 5.0)


def test_empty_scene_reports_finite_max_range_without_a_validity_mask() -> None:
    provider = four_beam_provider()
    ranges, normalized = provider.scan(
        np.zeros((2, 3), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0, 2), dtype=np.float32),
    )
    np.testing.assert_allclose(ranges, 5.0)
    np.testing.assert_allclose(normalized, 1.0)


def test_seeded_noise_is_reproducible_and_range_clipped() -> None:
    first = four_beam_provider(noise_std=0.2, noise_seed=91)
    second = four_beam_provider(noise_std=0.2, noise_seed=91)
    inputs = (
        np.zeros((8, 3), dtype=np.float32),
        np.array([[2.0, 0.0]], dtype=np.float32),
        np.array([[0.5, 0.5]], dtype=np.float32),
    )
    first_ranges, first_normalized = first.scan(*inputs)
    second_ranges, second_normalized = second.scan(*inputs)
    np.testing.assert_array_equal(first_ranges, second_ranges)
    np.testing.assert_array_equal(first_normalized, second_normalized)
    assert np.all(first_ranges >= first.cfg.min_range)
    assert np.all(first_ranges <= first.cfg.max_range)


@pytest.mark.parametrize(
    "cfg, message",
    [
        (PlanarLidarCfg(beam_count=0), "beam_count"),
        (PlanarLidarCfg(angle_min=1.0, angle_max=1.0), "angle_max"),
        (PlanarLidarCfg(min_range=2.0, max_range=1.0), "max_range"),
        (PlanarLidarCfg(max_range=np.inf), "ranges must be finite"),
        (PlanarLidarCfg(noise_std=-1.0), "noise_std"),
        (PlanarLidarCfg(noise_std=np.nan), "noise_std"),
    ],
)
def test_invalid_lidar_config_is_rejected(cfg: PlanarLidarCfg, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PlanarLidarProvider(cfg)
