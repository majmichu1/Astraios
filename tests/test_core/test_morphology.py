"""Tests for morphological operations."""

import numpy as np

from astraios.core.masks import Mask
from astraios.core.morphology import (
    MorphOp,
    MorphologyParams,
    StructuringElement,
    morphology_mask,
    morphology_transform,
)


class TestMorphologyTransform:
    def test_dilate_expands(self):
        data = np.zeros((50, 50), dtype=np.float32)
        data[25, 25] = 1.0
        params = MorphologyParams(operation=MorphOp.DILATE, kernel_size=3)
        result = morphology_transform(data, params)
        # Dilated dot should be larger
        assert (result > 0).sum() > 1

    def test_erode_shrinks(self):
        data = np.zeros((50, 50), dtype=np.float32)
        data[20:30, 20:30] = 1.0  # 10x10 square
        params = MorphologyParams(operation=MorphOp.ERODE, kernel_size=3)
        result = morphology_transform(data, params)
        assert (result > 0).sum() < (data > 0).sum()

    def test_open_removes_noise(self):
        data = np.zeros((50, 50), dtype=np.float32)
        data[20:30, 20:30] = 1.0
        data[5, 5] = 1.0  # isolated noise pixel
        params = MorphologyParams(operation=MorphOp.OPEN, kernel_size=3)
        result = morphology_transform(data, params)
        # Isolated pixel should be removed
        assert result[5, 5] == 0.0

    def test_close_fills_holes(self):
        data = np.zeros((50, 50), dtype=np.float32)
        data[20:30, 20:30] = 1.0
        data[25, 25] = 0.0  # hole in center
        params = MorphologyParams(operation=MorphOp.CLOSE, kernel_size=3)
        result = morphology_transform(data, params)
        # Hole should be filled
        assert result[25, 25] > 0.5

    def test_color_image(self):
        data = np.random.rand(3, 50, 50).astype(np.float32)
        params = MorphologyParams(operation=MorphOp.DILATE)
        result = morphology_transform(data, params)
        assert result.shape == (3, 50, 50)

    def test_output_in_range(self):
        data = np.random.rand(50, 50).astype(np.float32)
        params = MorphologyParams(operation=MorphOp.DILATE, kernel_size=5)
        result = morphology_transform(data, params)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_structuring_elements(self):
        data = np.zeros((50, 50), dtype=np.float32)
        data[25, 25] = 1.0
        for elem in StructuringElement:
            params = MorphologyParams(
                operation=MorphOp.DILATE, element=elem, kernel_size=5
            )
            result = morphology_transform(data, params)
            assert (result > 0).sum() > 1


class TestMorphologyMask:
    def test_dilate_mask(self):
        mask = Mask(
            data=np.zeros((50, 50), dtype=np.float32),
            name="test",
        )
        mask.data[25, 25] = 1.0
        params = MorphologyParams(operation=MorphOp.DILATE, kernel_size=3)
        result = morphology_mask(mask, params)
        assert isinstance(result, Mask)
        assert (result.data > 0).sum() > 1

    def test_mask_name_updated(self):
        mask = Mask(data=np.ones((50, 50), dtype=np.float32), name="Original")
        params = MorphologyParams(operation=MorphOp.ERODE)
        result = morphology_mask(mask, params)
        assert "ERODE" in result.name
