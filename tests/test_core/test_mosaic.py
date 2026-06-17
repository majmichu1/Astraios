"""Tests for mosaic stitching."""

import numpy as np
import pytest

from astraios.core.mosaic import MosaicParams, MosaicResult, mosaic_stitch


class TestMosaicStitch:
    def test_two_identical_panels(self):
        panel = np.random.rand(50, 50).astype(np.float32) * 0.5
        result = mosaic_stitch([panel, panel])
        assert isinstance(result, MosaicResult)
        assert result.n_panels == 2

    def test_output_in_range(self):
        p1 = np.random.rand(50, 50).astype(np.float32) * 0.5
        p2 = np.random.rand(50, 50).astype(np.float32) * 0.5
        result = mosaic_stitch([p1, p2])
        assert result.data.min() >= 0.0
        assert result.data.max() <= 1.0

    def test_color_panels(self):
        p1 = np.random.rand(3, 50, 50).astype(np.float32) * 0.5
        p2 = np.random.rand(3, 50, 50).astype(np.float32) * 0.5
        result = mosaic_stitch([p1, p2])
        assert result.data.ndim == 3
        assert result.data.shape[0] == 3

    def test_too_few_panels_raises(self):
        with pytest.raises(ValueError):
            mosaic_stitch([np.zeros((50, 50), dtype=np.float32)])

    def test_three_panels(self):
        panels = [np.random.rand(50, 50).astype(np.float32) * 0.5 for _ in range(3)]
        result = mosaic_stitch(panels)
        assert result.n_panels == 3
