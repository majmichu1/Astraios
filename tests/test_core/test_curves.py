"""Tests for curves transformation."""

import numpy as np
import pytest

from astraios.core.curves import (
    CurvePoints,
    CurvesParams,
    apply_curve_lut,
    curves_transform,
)
from astraios.core.masks import Mask


class TestCurvePoints:
    """Tests for CurvePoints control point management."""

    def test_default_points(self):
        cp = CurvePoints()
        assert len(cp.points) == 2
        assert cp.points[0] == (0.0, 0.0)
        assert cp.points[1] == (1.0, 1.0)

    def test_add_point(self):
        cp = CurvePoints()
        cp.add_point(0.5, 0.7)
        assert len(cp.points) == 3
        assert cp.points[1] == (0.5, 0.7)

    def test_add_point_sorted(self):
        cp = CurvePoints()
        cp.add_point(0.8, 0.9)
        cp.add_point(0.3, 0.4)
        xs = [p[0] for p in cp.points]
        assert xs == sorted(xs)

    def test_remove_point(self):
        cp = CurvePoints()
        cp.add_point(0.5, 0.5)
        assert len(cp.points) == 3
        cp.remove_point(1)
        assert len(cp.points) == 2

    def test_cannot_remove_endpoints(self):
        cp = CurvePoints()
        cp.remove_point(0)  # should be no-op
        assert len(cp.points) == 2
        cp.remove_point(1)  # last point
        assert len(cp.points) == 2

    def test_build_lut_identity(self):
        cp = CurvePoints()  # default (0,0) -> (1,1)
        lut = cp.build_lut()
        assert lut.shape[0] == 65536
        # Should be approximately identity
        np.testing.assert_allclose(lut[0], 0.0, atol=0.01)
        np.testing.assert_allclose(lut[-1], 1.0, atol=0.01)
        np.testing.assert_allclose(lut[32768], 0.5, atol=0.02)

    def test_build_lut_brightened(self):
        cp = CurvePoints()
        cp.add_point(0.5, 0.75)
        lut = cp.build_lut()
        # Midtones should be brighter
        assert lut[32768] > 0.6


class TestApplyCurveLut:
    """Tests for LUT application."""

    def test_identity_lut(self):
        lut = np.linspace(0, 1, 65536, dtype=np.float32)
        channel = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
        result = apply_curve_lut(channel, lut)
        np.testing.assert_array_almost_equal(result, channel, decimal=3)

    def test_inverted_lut(self):
        lut = np.linspace(1, 0, 65536, dtype=np.float32)
        channel = np.array([[0.0, 1.0]], dtype=np.float32)
        result = apply_curve_lut(channel, lut)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=0.01)
        np.testing.assert_allclose(result[0, 1], 0.0, atol=0.01)


class TestCurvesTransform:
    """Tests for the full curves transformation."""

    def test_identity_transform(self):
        image = np.random.rand(100, 100).astype(np.float32)
        result = curves_transform(image)
        np.testing.assert_array_almost_equal(result, image, decimal=3)

    def test_color_image(self):
        image = np.random.rand(3, 50, 50).astype(np.float32)
        result = curves_transform(image)
        assert result.shape == image.shape

    def test_custom_curve(self):
        image = np.ones((50, 50), dtype=np.float32) * 0.5
        params = CurvesParams()
        params.master.add_point(0.5, 0.8)
        result = curves_transform(image, params)
        # All pixels should be brighter
        assert result.mean() > 0.6

    def test_mask_support(self):
        image = np.ones((50, 50), dtype=np.float32) * 0.5
        params = CurvesParams()
        params.master.add_point(0.5, 0.9)

        mask_data = np.zeros((50, 50), dtype=np.float32)
        mask_data[25:, :] = 1.0
        mask = Mask(data=mask_data)

        result = curves_transform(image, params, mask=mask)
        # Top half should be unchanged
        assert abs(result[:25, :].mean() - 0.5) < 0.05
        # Bottom half should be brightened
        assert result[25:, :].mean() > 0.7
