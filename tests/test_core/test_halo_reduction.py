"""Tests for Halo-B-Gon bright-star halo reduction."""

import numpy as np
import pytest

from astraios.core.device_manager import get_device_manager
from astraios.core.halo_reduction import (
    HaloReductionLevel,
    HaloReductionParams,
    _reduce_halos_cpu,
    _reduce_halos_gpu,
    reduce_halos,
)
from astraios.core.masks import Mask


def _star_halo_image(color: bool = True, size: int = 160):
    """A synthetic bright star with a broad low halo on a dim background."""
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size // 2
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    background = 0.05
    halo = 0.35 * np.exp(-dist_sq / (2 * 30.0**2))
    core = 0.90 * np.exp(-dist_sq / (2 * 3.0**2))

    mono = np.clip(background + halo + core, 0.0, 1.0).astype(np.float32)
    if not color:
        return mono, dist_sq
    return np.stack([mono, mono, mono], axis=0), dist_sq


def _region_means(image, dist_sq, core_radius=3, halo_lo=12, halo_hi=28):
    core_mask = dist_sq <= core_radius**2
    halo_mask = (dist_sq >= halo_lo**2) & (dist_sq <= halo_hi**2)
    if image.ndim == 3:
        lum = image.mean(axis=0)
    else:
        lum = image
    return lum[core_mask].mean(), lum[halo_mask].mean()


class TestHaloSuppression:
    def test_halo_reduced_more_than_core(self):
        img, dist_sq = _star_halo_image()
        core_before, halo_before = _region_means(img, dist_sq)

        params = HaloReductionParams(reduction_level=HaloReductionLevel.HIGH)
        result = reduce_halos(img, params)
        core_after, halo_after = _region_means(result, dist_sq)

        # The halo should shrink proportionally more than the core.
        assert (halo_after / halo_before) < (core_after / core_before)
        # The core must still be clearly present, not blown out to near-zero.
        assert core_after > 0.15

    def test_higher_level_suppresses_halo_more(self):
        img, dist_sq = _star_halo_image()

        low = reduce_halos(img, HaloReductionParams(reduction_level=HaloReductionLevel.EXTRA_LOW))
        high = reduce_halos(img, HaloReductionParams(reduction_level=HaloReductionLevel.HIGH))

        _, halo_low = _region_means(low, dist_sq)
        _, halo_high = _region_means(high, dist_sq)
        assert halo_high < halo_low

    def test_is_linear_flag_runs_and_changes_output(self):
        img, _ = _star_halo_image()
        out_normal = reduce_halos(img, HaloReductionParams(is_linear=False))
        out_linear = reduce_halos(img, HaloReductionParams(is_linear=True))
        assert out_normal.shape == img.shape == out_linear.shape
        assert np.isfinite(out_linear).all()
        assert not np.allclose(out_normal, out_linear)


class TestMonoHandling:
    def test_mono_star_halo_is_processed(self):
        mono, dist_sq = _star_halo_image(color=False)
        result = reduce_halos(mono, HaloReductionParams(reduction_level=HaloReductionLevel.HIGH))
        assert result.shape == mono.shape
        core_before, halo_before = _region_means(mono, dist_sq)
        core_after, halo_after = _region_means(result, dist_sq)
        assert (halo_after / halo_before) < (core_after / core_before)


class TestMaskBlend:
    def test_mask_protects_original(self):
        img, dist_sq = _star_halo_image()
        h, w = img.shape[1:]
        mask_data = np.zeros((h, w), dtype=np.float32)
        mask_data[dist_sq <= 40**2] = 1.0  # only the star's local area is editable
        mask = Mask(data=mask_data)

        params = HaloReductionParams(reduction_level=HaloReductionLevel.HIGH)
        result = reduce_halos(img, params, mask=mask)
        protected = dist_sq > 40**2
        np.testing.assert_array_equal(result[:, protected], img[:, protected])


class TestGpuCpuAgreement:
    def test_gpu_matches_cpu(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("no GPU available")
        img, _ = _star_halo_image()
        level = int(HaloReductionLevel.HIGH)
        cpu_out = _reduce_halos_cpu(img, level, is_linear=False)
        gpu_out = _reduce_halos_gpu(img, level, is_linear=False, dm=dm)
        assert np.abs(cpu_out - gpu_out).mean() < 0.02

    def test_gpu_matches_cpu_linear(self):
        dm = get_device_manager()
        if not dm.is_gpu:
            pytest.skip("no GPU available")
        img, _ = _star_halo_image()
        level = int(HaloReductionLevel.MEDIUM)
        cpu_out = _reduce_halos_cpu(img, level, is_linear=True)
        gpu_out = _reduce_halos_gpu(img, level, is_linear=True, dm=dm)
        assert np.abs(cpu_out - gpu_out).mean() < 0.02
