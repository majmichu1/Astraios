"""Tests for memory-bounded tiled pixel-wise application."""

import numpy as np

from astraios.core.color_tools import ColorAdjustParams, SCNRParams, color_adjust, scnr
from astraios.core.tiling import apply_pixelwise_tiled, iter_tiles, should_tile


def test_should_tile_threshold():
    small = np.zeros((1000, 1000), dtype=np.float32)   # 1 MP
    big = np.zeros((3, 6000, 6000), dtype=np.float32)   # 36 MP
    assert not should_tile(small)
    assert should_tile(big)


def test_iter_tiles_covers_grid_exactly():
    covered = np.zeros((130, 170), dtype=int)
    for y0, y1, x0, x1 in iter_tiles(130, 170, tile=64):
        covered[y0:y1, x0:x1] += 1
    # Every pixel covered exactly once (no gaps, no overlaps).
    assert covered.min() == 1 and covered.max() == 1


def test_tiled_pixelwise_identical_mono():
    rng = np.random.default_rng(0)
    img = rng.random((200, 260)).astype(np.float32)
    ref = img * 0.5 + 0.1  # pixel-wise reference
    out = apply_pixelwise_tiled(img.copy(), lambda t: t * 0.5 + 0.1, tile=64)
    np.testing.assert_array_equal(out, ref)


def test_tiled_pixelwise_identical_color():
    rng = np.random.default_rng(1)
    img = rng.random((3, 200, 260)).astype(np.float32)
    # A per-channel pixel-wise op.
    gains = np.array([1.1, 0.9, 1.2], dtype=np.float32)[:, None, None]
    ref = np.clip(img * gains, 0, 1)
    out = apply_pixelwise_tiled(img.copy(), lambda t: np.clip(t * gains, 0, 1), tile=64)
    np.testing.assert_array_equal(out, ref)


def test_mutates_in_place():
    img = np.ones((3, 80, 90), dtype=np.float32)
    returned = apply_pixelwise_tiled(img, lambda t: t * 0.0, tile=32)
    assert returned is img            # same object
    assert float(img.max()) == 0.0    # mutated


def test_tiled_color_adjust_matches_full_frame():
    # The real OOM stage: color_adjust is pixel-wise, so tiling it must be
    # identical to the full-frame result (no seams).
    rng = np.random.default_rng(2)
    img = rng.random((3, 220, 300)).astype(np.float32)
    params = ColorAdjustParams(saturation=1.4, vibrance=0.2)

    full = color_adjust(img.copy(), params)
    tiled = apply_pixelwise_tiled(img.copy(), lambda t: color_adjust(t, params), tile=96)

    # GPU/CPU HSV round-trips carry tiny float error; require visually identical.
    assert np.max(np.abs(tiled - full)) < 1e-4


def test_tiled_scnr_matches_full_frame():
    rng = np.random.default_rng(3)
    img = rng.random((3, 220, 300)).astype(np.float32)
    params = SCNRParams()
    full = scnr(img.copy(), params)
    tiled = apply_pixelwise_tiled(img.copy(), lambda t: scnr(t, params), tile=96)
    assert np.max(np.abs(tiled - full)) < 1e-4
