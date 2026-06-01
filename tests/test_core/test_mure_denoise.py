"""Tests for MureDenoise noise estimation."""

import numpy as np
import pytest

from cosmica.core.mure_denoise import (
    _patch_std_map,
    estimate_noise,
    estimate_noise_from_dark,
    snr_estimate,
)


class TestEstimateNoise:
    def test_grayscale_flat_with_noise(self):
        rng = np.random.default_rng(42)
        true_std = 0.05
        clean = np.ones((128, 128), dtype=np.float32) * 0.5
        noisy = clean + rng.normal(0, true_std, (128, 128)).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)

        est = estimate_noise(noisy)

        assert est == pytest.approx(true_std, rel=0.2)

    def test_rgb_flat_with_noise(self):
        rng = np.random.default_rng(42)
        true_std = 0.03
        clean = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        noisy = clean + rng.normal(0, true_std, (64, 64, 3)).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)

        est_r, est_g, est_b = estimate_noise(noisy)

        assert est_r == pytest.approx(true_std, rel=0.2)
        assert est_g == pytest.approx(true_std, rel=0.2)
        assert est_b == pytest.approx(true_std, rel=0.2)

    def test_noiseless_image(self):
        clean = np.ones((64, 64), dtype=np.float32) * 0.5

        est = estimate_noise(clean)

        assert est < 1e-6

    def test_different_patch_size(self):
        rng = np.random.default_rng(42)
        true_std = 0.04
        clean = np.ones((96, 96), dtype=np.float32) * 0.5
        noisy = clean + rng.normal(0, true_std, (96, 96)).astype(np.float32)
        noisy = np.clip(noisy, 0, 1)

        est = estimate_noise(noisy, patch_size=16)

        assert est == pytest.approx(true_std, rel=0.25)

    def test_image_smaller_than_patch(self):
        small = np.ones((4, 4), dtype=np.float32)
        est = estimate_noise(small, patch_size=8)
        assert est == pytest.approx(0.0, abs=1e-6)

    def test_rgb_output_type(self):
        rng = np.random.default_rng(42)
        img = rng.normal(0.5, 0.05, (32, 32, 3)).astype(np.float32)
        img = np.clip(img, 0, 1)
        result = estimate_noise(img)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_grayscale_output_type(self):
        rng = np.random.default_rng(42)
        img = rng.normal(0.5, 0.05, (32, 32)).astype(np.float32)
        img = np.clip(img, 0, 1)
        result = estimate_noise(img)
        assert isinstance(result, float)


class TestPatchStdMap:
    def test_output_shape_grayscale(self):
        img = np.random.default_rng(42).normal(0.5, 0.1, (64, 64)).astype(np.float32)
        result = _patch_std_map(img, 8)
        assert result.shape == (8, 8)

    def test_output_shape_rgb(self):
        img = np.random.default_rng(42).normal(0.5, 0.1, (64, 64, 3)).astype(np.float32)
        result = _patch_std_map(img, 8)
        assert result.shape == (8, 8, 3)

    def test_non_divisible_dimensions(self):
        img = np.random.default_rng(42).normal(0.5, 0.1, (70, 70)).astype(np.float32)
        result = _patch_std_map(img, 8)
        assert result.shape == (8, 8)


class TestEstimateNoiseFromDark:
    def test_grayscale_stack(self):
        rng = np.random.default_rng(42)
        true_std = 0.02
        stack = rng.normal(0, true_std, (16, 64, 64)).astype(np.float32)

        est = estimate_noise_from_dark(stack)

        assert est == pytest.approx(true_std, rel=0.2)

    def test_rgb_stack(self):
        rng = np.random.default_rng(42)
        true_std = 0.015
        stack = rng.normal(0, true_std, (16, 64, 64, 3)).astype(np.float32)

        est = estimate_noise_from_dark(stack)

        assert est == pytest.approx(true_std, rel=0.2)


class TestSnrEstimate:
    def test_basic_snr(self):
        img = np.ones((32, 32), dtype=np.float32) * 0.5
        noise_std = 0.1
        expected = 0.5 / 0.1

        result = snr_estimate(img, noise_std)

        assert result == pytest.approx(expected)

    def test_zero_noise(self):
        img = np.ones((32, 32), dtype=np.float32) * 0.5

        result = snr_estimate(img, 0.0)

        assert result == float("inf")

    def test_zero_mean(self):
        img = np.zeros((32, 32), dtype=np.float32)
        result = snr_estimate(img, 0.1)
        assert result == pytest.approx(0.0)
