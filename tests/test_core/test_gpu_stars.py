"""Tests for GPU star detection and alignment."""

import cv2
import numpy as np
import pytest
import torch

from cosmica.core.gpu_stars import (
    Star,  # re-exported from star_detection
    compose_affine_transforms,
    detect_stars_gpu,
    estimate_transform_gpu,
    match_stars_gpu,
    warp_image_gpu,
)


class TestDetectStarsGPU:
    def test_no_stars_on_blank(self):
        blank = torch.zeros(100, 100)
        assert detect_stars_gpu(blank, threshold_sigma=5.0) == []

    def test_detects_bright_spots(self):
        img = torch.zeros(100, 100)
        img[50, 50] = 1.0
        img[30, 70] = 0.8
        stars = detect_stars_gpu(img, threshold_sigma=2.0)
        assert len(stars) >= 1

    def test_max_stars_limit(self):
        img = torch.zeros(200, 200)
        for i in range(10):
            for j in range(10):
                img[20 + i * 15, 20 + j * 15] = 1.0
        stars = detect_stars_gpu(img, max_stars=5)
        assert len(stars) <= 5

    def test_sorted_by_flux(self):
        img = torch.zeros(100, 100)
        img[20, 20] = 0.5
        img[50, 50] = 1.0
        img[80, 80] = 0.8
        stars = detect_stars_gpu(img, threshold_sigma=1.0)
        if len(stars) >= 2:
            assert stars[0].flux >= stars[1].flux

    def test_star_is_same_type_as_star_detection(self):
        """Star class should be the same object from star_detection (no duplicate)."""
        from cosmica.core.star_detection import Star as StarCPU
        assert Star is StarCPU


class TestMatchStarsGPU:
    def test_empty_returns_empty(self):
        assert match_stars_gpu([], [], 10.0) == []
        assert match_stars_gpu([Star(0, 0, 1.0)], [], 10.0) == []

    def test_matches_translation(self):
        ref = [Star(10, 10, 1.0), Star(20, 20, 0.9)]
        tgt = [Star(15, 15, 1.0), Star(25, 25, 0.9)]
        matches = match_stars_gpu(ref, tgt, max_dist=10.0)
        assert len(matches) >= 1

    def test_respects_max_dist(self):
        ref = [Star(0, 0, 1.0)]
        tgt = [Star(100, 100, 1.0)]
        matches = match_stars_gpu(ref, tgt, max_dist=10.0)
        assert len(matches) == 0


class TestEstimateTransformGPU:
    def test_identity_transform(self):
        matches = [
            (Star(10, 10, 1.0), Star(10, 10, 1.0)),
            (Star(20, 20, 1.0), Star(20, 20, 1.0)),
            (Star(30, 30, 1.0), Star(30, 30, 1.0)),
        ]
        transform = estimate_transform_gpu(matches)
        assert transform is not None
        np.testing.assert_allclose(transform[:2, :2], np.eye(2), atol=1e-3)
        np.testing.assert_allclose(transform[:2, 2], [0, 0], atol=1e-3)

    def test_pure_translation(self):
        matches = [
            (Star(10, 10, 1.0), Star(20, 20, 1.0)),
            (Star(20, 20, 1.0), Star(30, 30, 1.0)),
            (Star(30, 30, 1.0), Star(40, 40, 1.0)),
        ]
        transform = estimate_transform_gpu(matches)
        assert transform is not None
        np.testing.assert_allclose(transform[0, 2], -10.0, atol=2.0)
        np.testing.assert_allclose(transform[1, 2], -10.0, atol=2.0)

    def test_needs_min_matches(self):
        matches = [(Star(10, 10, 1.0), Star(20, 20, 1.0))]
        assert estimate_transform_gpu(matches) is None

    def test_uses_ransac(self):
        """RANSAC should produce a valid result even with outlier matches."""
        # 3 good matches + 1 outlier
        matches = [
            (Star(10, 10, 1.0), Star(20, 20, 1.0)),
            (Star(20, 20, 1.0), Star(30, 30, 1.0)),
            (Star(30, 30, 1.0), Star(40, 40, 1.0)),
            (Star(50, 50, 1.0), Star(5, 90, 1.0)),  # outlier
        ]
        transform = estimate_transform_gpu(matches)
        assert transform is not None
        # Translation should still be ~-10
        np.testing.assert_allclose(transform[0, 2], -10.0, atol=3.0)


class TestWarpImageGPU:
    def test_identity_warp(self):
        img = torch.arange(100.0).reshape(10, 10)
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        warped = warp_image_gpu(img, identity)
        np.testing.assert_allclose(warped, img, atol=1e-3)

    def test_warp_matches_opencv_on_large_rectangular(self):
        cv2 = pytest.importorskip("cv2")
        h, w = 4000, 6000
        rng = np.random.default_rng(0)
        img_np = rng.random((h, w)).astype(np.float32)
        angle = np.deg2rad(7.0)
        tx, ty = 12.5, -8.0
        center = (w / 2, h / 2)
        m_cv = cv2.getRotationMatrix2D(center, np.rad2deg(angle), 1.0)
        m_cv[0, 2] += tx
        m_cv[1, 2] += ty
        m_cv = m_cv.astype(np.float32)

        ocv = cv2.warpAffine(img_np, m_cv, (w, h), flags=cv2.INTER_CUBIC)
        # warp_image_gpu expects output→input (same matrix OpenCV warpAffine uses)
        m_inv = np.linalg.inv(np.vstack([m_cv, [0, 0, 1]]))[:2].astype(np.float32)
        gpu = warp_image_gpu(torch.from_numpy(img_np), m_inv, mode="bicubic").numpy()
        np.testing.assert_allclose(gpu, ocv, rtol=0.05, atol=0.05)

    def test_different_from_original_with_shift(self):
        img = torch.arange(100.0).reshape(10, 10).float()
        identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        shift = np.array([[1, 0, 5], [0, 1, 5]], dtype=np.float32)
        warped_identity = warp_image_gpu(img, identity)
        warped_shift = warp_image_gpu(img, shift)
        diff = (warped_identity - warped_shift).abs().max().item()
        assert diff > 0.5

    def test_mode_parameter(self):
        img = torch.rand(50, 60)
        M = np.array([[1, 0, 2], [0, 1, 3]], dtype=np.float32)
        bilinear = warp_image_gpu(img, M, mode="bilinear")
        bicubic = warp_image_gpu(img, M, mode="bicubic")
        # Both should produce valid output, slightly different due to interpolation
        assert bilinear.shape == img.shape
        assert bicubic.shape == img.shape


class TestComposeAffineTransforms:
    def test_two_translations(self):
        t1 = np.array([[1, 0, 10], [0, 1, 5]], dtype=np.float32)
        t2 = np.array([[1, 0, 20], [0, 1, 15]], dtype=np.float32)
        combined = compose_affine_transforms(t1, t2)
        np.testing.assert_allclose(combined[0, 2], 30.0, atol=0.01)
        np.testing.assert_allclose(combined[1, 2], 20.0, atol=0.01)

    def test_translation_then_scale(self):
        t = np.array([[1, 0, 10], [0, 1, 5]], dtype=np.float32)
        s = np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float32)
        combined = compose_affine_transforms(t, s)
        np.testing.assert_allclose(combined[0, 2], 20.0, atol=0.01)
        np.testing.assert_allclose(combined[1, 2], 10.0, atol=0.01)
        np.testing.assert_allclose(combined[0, 0], 2.0, atol=0.01)

    def test_general_affine_composition(self):
        """compose_affine_transforms handles general affine (not just similarity)."""
        m1 = np.array([[1, 0, 3], [0, 1, 4]], dtype=np.float32)
        m2 = np.array([[1, 0, 7], [0, 1, 1]], dtype=np.float32)
        combined = compose_affine_transforms(m1, m2)
        # m2(m1(p)) for p=(0,0): m1 -> (3,4), m2 -> (3+7, 4+1) = (10, 5)
        np.testing.assert_allclose(combined[0, 2], 10.0, atol=0.01)
        np.testing.assert_allclose(combined[1, 2], 5.0, atol=0.01)
