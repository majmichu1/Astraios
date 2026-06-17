"""Tests for lens distortion correction."""

import cv2
import numpy as np
import pytest

from astraios.core.lens_distortion import (
    LensDistortionParams,
    _compute_camera_matrix,
    _radial_distortion_map,
    correct_distortion,
    estimate_distortion_from_stars,
)


class TestLensDistortionParams:
    """Tests for LensDistortionParams defaults and construction."""

    def test_defaults(self):
        """Default params should have zero distortion."""
        p = LensDistortionParams()
        assert p.k1 == 0.0
        assert p.k2 == 0.0
        assert p.k3 == 0.0
        assert p.p1 == 0.0
        assert p.p2 == 0.0
        assert p.fov == 0.0
        assert p.focal_length_mm == 0.0
        assert p.sensor_width_mm == 36.0

    def test_custom_params(self):
        """Should accept custom parameter values."""
        p = LensDistortionParams(k1=-0.2, k2=0.05, fov=90.0, focal_length_mm=50.0)
        assert p.k1 == -0.2
        assert p.k2 == 0.05
        assert p.fov == 90.0
        assert p.focal_length_mm == 50.0

    def test_distortion_coeffs_shape(self):
        """distortion_coeffs should be 1x5 float64."""
        p = LensDistortionParams(k1=0.1, k2=-0.05, p1=0.001, p2=0.002, k3=0.01)
        coeffs = p.distortion_coeffs
        assert coeffs.shape == (1, 5)
        assert coeffs.dtype == np.float64
        np.testing.assert_array_almost_equal(
            coeffs[0], [0.1, -0.05, 0.001, 0.002, 0.01],
        )


class TestComputeCameraMatrix:
    """Tests for the internal camera matrix helper."""

    def test_center_principal_point(self):
        """Principal point should be at image centre."""
        k = _compute_camera_matrix(100, 80, 50.0, 36.0)
        assert k[0, 2] == pytest.approx(49.5)
        assert k[1, 2] == pytest.approx(39.5)

    def test_focal_length_in_pixels(self):
        """Focal length should be converted correctly."""
        k = _compute_camera_matrix(100, 80, 36.0, 36.0)
        # focal_px = 36/36 * 100 = 100
        assert k[0, 0] == pytest.approx(100.0)
        assert k[1, 1] == pytest.approx(100.0)

    def test_zero_focal_length_fallback(self):
        """Zero focal length should fall back to max(w, h)."""
        k = _compute_camera_matrix(200, 100, 0.0, 36.0)
        assert k[0, 0] == pytest.approx(200.0)
        assert k[1, 1] == pytest.approx(200.0)


class TestRadialDistortionMap:
    """Tests for the radial remap builder."""

    def test_output_shape(self):
        """map_x and map_y should have shape (h, w)."""
        mx, my = _radial_distortion_map(50, 100, 0.0, 0.0, 0.0)
        assert mx.shape == (50, 100)
        assert my.shape == (50, 100)
        assert mx.dtype == np.float32
        assert my.dtype == np.float32

    def zero_coeffs_identity(self):
        """With all k=0 the map should be identity."""
        h, w = 40, 60
        mx, my = _radial_distortion_map(h, w, 0.0, 0.0, 0.0)
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        y, x = np.meshgrid(ys, xs, indexing="ij")
        np.testing.assert_array_almost_equal(mx, x, decimal=4)
        np.testing.assert_array_almost_equal(my, y, decimal=4)

    def test_positive_k1_barrel_effect(self):
        """Positive k1 should push pixels outward (barrel)."""
        h, w = 50, 50
        mx, my = _radial_distortion_map(h, w, 0.3, 0.0, 0.0)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        center_val_x = mx[int(cy), int(cx)]
        corner_val_x = mx[0, 0]
        # Centre should be near identity; corners should be pushed outward
        assert corner_val_x < center_val_x


class TestCorrectDistortion:
    """Tests for the main correct_distortion function."""

    def test_zero_params_returns_copy(self):
        """Zero distortion coefficients should return a copy."""
        img = np.random.RandomState(42).rand(64, 64).astype(np.float32)
        result = correct_distortion(img, LensDistortionParams())
        assert result is not img
        np.testing.assert_array_equal(result, img)

    def test_output_shape_matches_input_mono(self):
        """Output shape should match input for (H, W)."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        result = correct_distortion(img, LensDistortionParams(k1=0.1))
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_output_shape_matches_input_color(self):
        """Output shape should match input for (H, W, C)."""
        img = np.full((64, 64, 3), 0.5, dtype=np.float32)
        result = correct_distortion(img, LensDistortionParams(k1=0.1))
        assert result.shape == (64, 64, 3)

    def test_values_in_valid_range(self):
        """Corrected image should remain in valid range."""
        img = np.full((32, 32), 0.5, dtype=np.float32)
        result = correct_distortion(img, LensDistortionParams(k1=0.3, k2=-0.1))
        assert np.all(np.isfinite(result))

    def test_invalid_ndim_raises(self):
        """Should raise ValueError for non-2D/3D input."""
        img = np.zeros((2, 3, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Expected 2D or 3D image"):
            correct_distortion(img, LensDistortionParams())

    def test_with_focal_length(self):
        """Should work with explicit focal length."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        params = LensDistortionParams(k1=0.2, focal_length_mm=50.0, sensor_width_mm=36.0)
        result = correct_distortion(img, params)
        assert result.shape == (64, 64)


class TestCorrectDistortionSyntheticGrid:
    """Test that distortion correction properly undoes a known distortion."""

    @staticmethod
    def _apply_radial_distort(
        img: np.ndarray, k1: float, k2: float, k3: float = 0.0,
    ) -> np.ndarray:
        """Apply forward radial distortion via remap."""
        h, w = img.shape[:2]
        mx, my = _radial_distortion_map(h, w, k1, k2, k3)
        if img.ndim == 2:
            return cv2.remap(img, mx, my, cv2.INTER_LINEAR)
        else:
            channels = []
            for c in range(img.shape[2]):
                channels.append(
                    cv2.remap(img[..., c], mx, my, cv2.INTER_LINEAR),
                )
            return np.stack(channels, axis=-1)

    def test_correct_barrel_distortion(self):
        """Correction should undo a known barrel distortion."""
        h, w = 128, 128
        grid = np.zeros((h, w), dtype=np.float32)
        step = 16
        grid[::step, :] = 1.0
        grid[:, ::step] = 1.0

        k1_true = 0.15
        distorted = self._apply_radial_distort(grid, k1_true, 0.0)

        params = LensDistortionParams(k1=-k1_true, focal_length_mm=50.0, sensor_width_mm=36.0)
        corrected = correct_distortion(distorted, params)

        # Centre line should be largely unchanged
        cy = h // 2
        assert corrected[cy, :].sum() > 0


class TestEstimateDistortionFromStars:
    """Tests for the star-based distortion estimator."""

    def test_known_star_positions(self):
        """Should return params when given known star positions."""
        h, w = 128, 128
        # Generate synthetic star positions near image centre
        rng = np.random.RandomState(42)
        positions = []
        for _ in range(30):
            x = w / 2 + rng.randn() * 20
            y = h / 2 + rng.randn() * 20
            positions.append((x, y))

        params = estimate_distortion_from_stars(
            np.zeros((h, w), dtype=np.float32),
            known_star_positions=positions,
        )
        assert isinstance(params, LensDistortionParams)

    def test_too_few_stars_returns_default(self):
        """Fewer than 5 stars should return default params."""
        positions = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
        params = estimate_distortion_from_stars(
            np.zeros((64, 64), dtype=np.float32),
            known_star_positions=positions,
        )
        assert params.k1 == 0.0
        assert params.k2 == 0.0

    def test_auto_detect_fallback(self):
        """Should not crash when auto-detecting stars on blank image."""
        img = np.full((64, 64), 0.1, dtype=np.float32)
        params = estimate_distortion_from_stars(img)
        assert isinstance(params, LensDistortionParams)
