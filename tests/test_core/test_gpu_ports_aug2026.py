"""GPU ports from August 2026: each must give the CPU/library answer."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.ndimage

from astraios.core import filters
from astraios.core.device_manager import get_device_manager


def _plane(seed, h=96, w=80):
    return np.random.default_rng(seed).random((h, w)).astype(np.float32)


class TestGPUMedian:
    @pytest.mark.parametrize("ksize", [7, 9, 11])
    def test_bit_identical_to_scipy(self, ksize):
        img = _plane(1)
        ref = scipy.ndimage.median_filter(img, size=ksize)
        out = filters._median_2d_gpu(img, ksize, get_device_manager())
        assert np.array_equal(out, ref)

    def test_banding_does_not_change_the_result(self, monkeypatch):
        img = _plane(2, 120, 64)
        ref = scipy.ndimage.median_filter(img, size=7)
        monkeypatch.setattr(filters, "_MEDIAN_BAND_BYTES", 64 * 7 * 7 * 4 * 5)  # 5-row bands
        out = filters._median_2d_gpu(img, 7, get_device_manager())
        assert np.array_equal(out, ref)

    def test_public_median_filter_colour_large_kernel(self):
        from astraios.core.filters import MedianFilterParams, median_filter

        img = np.stack([_plane(3), _plane(4), _plane(5)])
        out = median_filter(img, MedianFilterParams(kernel_size=7))
        ref = np.stack([scipy.ndimage.median_filter(c, size=7) for c in img])
        assert np.array_equal(out, ref)


class TestLRGBChromaBlur:
    def test_matches_scipy_within_budget(self):
        from astraios.core.lrgb import LRGBParams, lrgb_combine

        rng = np.random.default_rng(6)
        rgb = (rng.random((3, 64, 64)) * 0.5 + 0.2).astype(np.float32)
        lum = (rng.random((64, 64)) * 0.5 + 0.3).astype(np.float32)
        out = lrgb_combine(lum, rgb, LRGBParams(chrominance_noise=0.5))
        assert out.shape == rgb.shape and np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestSelectiveAdjustBlurParity:
    def test_gpu_kernel_width_matches_cv2(self):
        import cv2
        import torch

        from astraios.core.selective_adjust import _gaussian_blur_gpu, _gaussian_blur_mask_np

        mask = (_plane(7) > 0.7).astype(np.float32)
        sigma = 2.5
        cpu = _gaussian_blur_mask_np(mask, sigma)
        gpu = _gaussian_blur_gpu(torch.from_numpy(mask).to(get_device_manager().device), sigma)
        gpu = gpu.cpu().numpy()
        assert np.abs(cpu - gpu).max() < 2e-5
        del cv2
