"""Tests for continuum subtraction — continuum_subtract.py."""

import numpy as np
import pytest

from astraios.core.continuum_subtract import (
    ContinuumSubtractParams,
    _compute_pedestal,
    _estimate_star_gain,
    _normalize_gain,
    subtract_continuum,
)
from astraios.core.masks import Mask


def _star_field(h: int = 128, w: int = 128, n_stars: int = 15, seed: int = 0):
    """Background + a handful of Gaussian point sources ("stars")."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    img = np.zeros((h, w), dtype=np.float32)
    positions = []
    ys = rng.integers(15, h - 15, n_stars)
    xs = rng.integers(15, w - 15, n_stars)
    for y, x in zip(ys, xs):
        amp = float(rng.uniform(0.4, 0.9))
        sigma = 1.5
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        positions.append((int(x), int(y)))
    return img.astype(np.float32), positions


def _nebula_patch(h: int = 128, w: int = 128, amplitude: float = 0.3):
    """A smooth emission patch confined to a corner, disjoint from stars."""
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h - 25, w - 25
    patch = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12.0**2))
    return patch.astype(np.float32)


def _synthetic_pair(h=128, w=128, n_stars=15, seed=0, star_gain=1.0, nebula_amp=0.3):
    """continuum = background + stars; narrowband = star_gain*continuum + nebula."""
    background = 0.10
    stars, positions = _star_field(h, w, n_stars, seed)
    continuum = np.clip(background + stars, 0.0, 1.0).astype(np.float32)
    nebula = _nebula_patch(h, w, nebula_amp)
    narrowband = np.clip(star_gain * continuum + nebula, 0.0, 1.0).astype(np.float32)
    return narrowband, continuum, positions, nebula


class TestSubtractContinuum:
    def test_removes_stars_keeps_nebula(self):
        narrowband, continuum, positions, nebula = _synthetic_pair(star_gain=1.0)
        params = ContinuumSubtractParams(scale_factor=1.0)
        result = subtract_continuum(narrowband, continuum, params)

        assert result.shape == narrowband.shape

        star_peak_before = np.array([narrowband[y, x] for x, y in positions])
        star_peak_after = np.array([result[y, x] for x, y in positions])
        background_level = np.median(result)

        # Stars should be pulled down close to background, far below their
        # original peak amplitude.
        assert star_peak_after.mean() < star_peak_before.mean() * 0.5
        assert abs(star_peak_after.mean() - background_level) < 0.15

        # The nebula patch (present only in the narrowband) should survive.
        nebula_mask = nebula > 0.05
        assert result[nebula_mask].mean() > 0.5 * nebula[nebula_mask].mean()

    def test_scale_factor_zero_is_noop(self):
        narrowband, continuum, _, _ = _synthetic_pair()
        params = ContinuumSubtractParams(
            scale_factor=0.0,
            scaling_method="manual",
            background_pedestal=False,
        )
        result = subtract_continuum(narrowband, continuum, params)
        np.testing.assert_allclose(result, np.clip(narrowband, 0.0, 1.0), atol=1e-6)

    def test_scale_factor_extreme_over_subtracts(self):
        narrowband, continuum, positions, _ = _synthetic_pair(star_gain=1.0)
        params = ContinuumSubtractParams(
            scale_factor=5.0,
            scaling_method="manual",
            background_pedestal=False,
        )
        result = subtract_continuum(narrowband, continuum, params)
        # Heavy over-subtraction should clip star cores to the floor.
        star_vals = np.array([result[y, x] for x, y in positions])
        assert np.any(star_vals <= 1e-6)
        assert result.min() >= 0.0

    def test_mono_and_color_shapes(self):
        narrowband, continuum, _, _ = _synthetic_pair()
        mono_result = subtract_continuum(narrowband, continuum)
        assert mono_result.shape == narrowband.shape
        assert mono_result.dtype == np.float32

        color_nb = np.stack([narrowband, narrowband * 0.9, narrowband * 1.1], axis=0)
        color_cont = np.stack([continuum, continuum * 0.9, continuum * 1.1], axis=0)
        color_result = subtract_continuum(color_nb, color_cont)
        assert color_result.shape == color_nb.shape
        assert color_result.dtype == np.float32

    def test_color_broadcasts_mono_continuum(self):
        narrowband, continuum, _, _ = _synthetic_pair()
        color_nb = np.stack([narrowband, narrowband, narrowband], axis=0)
        result = subtract_continuum(color_nb, continuum)
        assert result.shape == color_nb.shape

    def test_mask_confines_processing(self):
        narrowband, continuum, _, _ = _synthetic_pair()
        h, w = narrowband.shape
        mask_data = np.zeros((h, w), dtype=np.float32)
        mask_data[:, w // 2 :] = 1.0  # only process right half
        mask = Mask(data=mask_data)

        result = subtract_continuum(narrowband, continuum, mask=mask)

        # Left half (masked out) must be untouched.
        np.testing.assert_array_equal(result[:, : w // 2], narrowband[:, : w // 2])
        # Right half should generally differ (continuum was subtracted).
        assert not np.allclose(result[:, w // 2 :], narrowband[:, w // 2 :])

    def test_output_in_range(self):
        narrowband, continuum, _, _ = _synthetic_pair()
        result = subtract_continuum(narrowband, continuum)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_invalid_scaling_method_raises(self):
        narrowband, continuum, _, _ = _synthetic_pair()
        bad_params = ContinuumSubtractParams(scaling_method="bogus")
        with pytest.raises(ValueError):
            subtract_continuum(narrowband, continuum, bad_params)

    def test_shape_mismatch_raises(self):
        narrowband, continuum, _, _ = _synthetic_pair()
        with pytest.raises(ValueError):
            subtract_continuum(narrowband, continuum[:-1, :])

    def test_progress_callback_invoked(self):
        narrowband, continuum, _, _ = _synthetic_pair()
        seen = []
        subtract_continuum(narrowband, continuum, progress=lambda f, m: seen.append(f))
        assert seen
        assert seen[-1] == 1.0
        assert all(0.0 <= f <= 1.0 for f in seen)


class TestNormalizeGain:
    def test_matches_mad_and_median(self):
        rng = np.random.default_rng(1)
        cont = (0.3 + 0.05 * rng.standard_normal((64, 64))).astype(np.float32)
        cont = np.clip(cont, 0, 1)
        nb = (cont * 2.0 + 0.1).astype(np.float32)  # different gain/offset
        nb = np.clip(nb, 0, 1)

        out = _normalize_gain(nb, cont)
        assert abs(np.median(out) - np.median(cont)) < 0.02


class TestComputePedestal:
    def test_uniform_background_yields_near_zero_pedestal(self):
        nb = np.full((80, 80), 0.2, dtype=np.float32)
        cont = np.full((80, 80), 0.2, dtype=np.float32)
        rng = np.random.default_rng(0)
        ped_nb, ped_cont = _compute_pedestal(
            nb, cont, num_boxes=50, box_size=10, iterations=5, rng=rng
        )
        assert abs(ped_nb) < 1e-6
        assert abs(ped_cont) < 1e-6

    def test_syncs_continuum_darker_patch_to_narrowband(self):
        nb = np.full((80, 80), 0.3, dtype=np.float32)
        cont = np.full((80, 80), 0.3, dtype=np.float32)
        cont[30:50, 30:50] = 0.1  # continuum has a darker sky patch than narrowband
        rng = np.random.default_rng(0)
        ped_nb, ped_cont = _compute_pedestal(
            nb, cont, num_boxes=80, box_size=10, iterations=10, rng=rng
        )
        # Continuum's dark patch should get pulled up more than narrowband's.
        assert ped_cont >= ped_nb


class TestEstimateStarGain:
    def test_matches_known_gain(self):
        stars, _ = _star_field(n_stars=20, seed=2)
        continuum = np.clip(0.1 + stars, 0, 1).astype(np.float32)
        narrowband = np.clip(0.1 + 2.0 * stars, 0, 1).astype(np.float32)
        params = ContinuumSubtractParams()
        gain = _estimate_star_gain(narrowband, continuum, params)
        assert gain == pytest.approx(2.0, rel=0.35)

    def test_too_few_stars_falls_back_to_one(self):
        nb = np.full((32, 32), 0.2, dtype=np.float32)
        cont = np.full((32, 32), 0.2, dtype=np.float32)
        params = ContinuumSubtractParams()
        gain = _estimate_star_gain(nb, cont, params)
        assert gain == 1.0
