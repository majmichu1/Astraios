"""Tests for banding reduction."""

import numpy as np

from astraios.core.banding import BandingParams, banding_reduction
from astraios.core.masks import Mask


class TestBandingReduction:
    """Tests for banding reduction."""

    def test_horizontal_banding_removed(self):
        """Should detect and remove row-level offsets."""
        image = np.ones((100, 100), dtype=np.float32) * 0.3
        # Add horizontal banding: every 10th row has an offset
        for i in range(0, 100, 10):
            image[i, :] += 0.05

        params = BandingParams(horizontal=True, vertical=False, amount=1.0)
        result = banding_reduction(image, params)

        # Banding rows should now be closer to the mean
        banding_rows = result[::10, :].mean()
        normal_rows = result[5::10, :].mean()
        assert abs(banding_rows - normal_rows) < abs(0.05)

    def test_vertical_banding_removed(self):
        """Should detect and remove column-level offsets."""
        image = np.ones((100, 100), dtype=np.float32) * 0.3
        for j in range(0, 100, 10):
            image[:, j] += 0.05

        params = BandingParams(horizontal=False, vertical=True, amount=1.0)
        result = banding_reduction(image, params)

        banding_cols = result[:, ::10].mean()
        normal_cols = result[:, 5::10].mean()
        assert abs(banding_cols - normal_cols) < abs(0.05)

    def test_no_banding_no_params(self):
        """Disabled should return image unchanged."""
        image = np.random.rand(50, 50).astype(np.float32)
        params = BandingParams(horizontal=False, vertical=False)
        result = banding_reduction(image, params)
        np.testing.assert_array_equal(result, image)

    def test_color_image(self):
        """Should work with multi-channel images."""
        image = np.random.rand(3, 50, 50).astype(np.float32) * 0.3 + 0.1
        result = banding_reduction(image, BandingParams(horizontal=True))
        assert result.shape == (3, 50, 50)

    def test_output_in_range(self):
        """Output should be clipped to [0, 1]."""
        image = np.random.rand(100, 100).astype(np.float32)
        result = banding_reduction(image, BandingParams(horizontal=True, vertical=True))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_amount_controls_strength(self):
        """Lower amount should correct less."""
        image = np.ones((100, 100), dtype=np.float32) * 0.3
        for i in range(0, 100, 5):
            image[i, :] += 0.05

        r_full = banding_reduction(image, BandingParams(amount=1.0))
        r_half = banding_reduction(image, BandingParams(amount=0.5))
        # Full correction should have less variance than half
        assert r_full.std() <= r_half.std()

    def test_mask_support(self):
        """Mask should limit corrections."""
        image = np.ones((100, 100), dtype=np.float32) * 0.3
        for i in range(0, 100, 10):
            image[i, :] += 0.05

        mask_data = np.zeros((100, 100), dtype=np.float32)
        mask_data[50:, :] = 1.0
        mask = Mask(data=mask_data)

        result = banding_reduction(image, BandingParams(horizontal=True), mask=mask)
        # Top half should be unchanged (banding still present)
        np.testing.assert_allclose(result[:50, :], image[:50, :], atol=0.01)
