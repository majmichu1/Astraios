"""Tests for the built-in (morphological + diffusion-inpaint) star remover."""

import numpy as np

from cosmica.ai.inference.star_removal import _diffuse_inpaint, remove_stars_builtin


def _nebula_with_stars(h=160, w=160):
    yy, xx = np.mgrid[0:h, 0:w]
    nebula = (0.15 * np.exp(-(((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * 45 ** 2)))).astype(np.float32) + 0.05
    img = nebula.copy()
    stars = [(40, 40), (120, 120), (80, 25), (50, 130)]
    for sy, sx in stars:
        img[sy - 2:sy + 3, sx - 2:sx + 3] += 0.7
    return np.clip(img, 0, 1).astype(np.float32), nebula, stars


class TestDiffuseInpaint:
    def test_fills_hole_from_surroundings(self):
        field = np.full((40, 40), 0.3, np.float32)
        mask = np.zeros((40, 40), bool)
        mask[18:22, 18:22] = True
        corrupted = field.copy()
        corrupted[mask] = 1.0  # bright blob to remove
        out = _diffuse_inpaint(corrupted, mask)
        # The hole should be filled with the surrounding value, not the blob.
        assert abs(float(out[19, 19]) - 0.3) < 0.02
        # Known pixels untouched.
        assert abs(float(out[0, 0]) - 0.3) < 1e-6

    def test_no_mask_is_noop(self):
        field = np.full((16, 16), 0.4, np.float32)
        out = _diffuse_inpaint(field, np.zeros((16, 16), bool))
        np.testing.assert_allclose(out, field, atol=1e-6)


class TestRemoveStarsBuiltin:
    def test_removes_stars_and_reconstructs_nebula(self):
        img, nebula, stars = _nebula_with_stars()
        out = remove_stars_builtin(img, threshold=0.5)
        assert out.shape == img.shape
        # Peak drops (brightest stars gone), nebula peak (~0.2) remains.
        assert out.max() < 0.4
        # Under each star, the reconstructed value matches the true nebula.
        for sy, sx in stars:
            assert abs(float(out[sy, sx]) - float(nebula[sy, sx])) < 0.05

    def test_preserves_starless_nebula(self):
        img, nebula, _ = _nebula_with_stars()
        out = remove_stars_builtin(img, threshold=0.5)
        # A star-free patch must be untouched.
        patch = (slice(95, 115), slice(95, 115))
        np.testing.assert_allclose(out[patch], img[patch], atol=0.02)

    def test_color_image(self):
        img, _, _ = _nebula_with_stars(96, 96)
        color = np.stack([img, img * 0.8, img * 0.6]).astype(np.float32)
        out = remove_stars_builtin(color, threshold=0.5)
        assert out.shape == color.shape
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestDoesNotDestroyImage:
    """Regressions for the catastrophic 'whole image blown to white / dark hole
    in the nebula core' bugs.

    1. On a smooth background the positive-clipped residual collapsed the noise
       estimate to ~0, the star mask covered the whole frame, and the inpainter
       flooded everything with a saturated core's value (~1.0).
    2. A large bright nebula core was classified as a giant 'star' and inpainted
       away, leaving a dark hole.
    """

    def test_saturated_core_does_not_blow_background(self):
        rng = np.random.default_rng(0)
        img = np.clip(rng.normal(0.1, 0.01, (300, 300)).astype(np.float32), 0, 1)
        img[140:160, 140:160] = 0.99  # saturated core blob
        out = remove_stars_builtin(img, threshold=0.5)
        # Background corner must stay dark, not flood toward white.
        assert float(out[:40, :40].mean()) < 0.2
        assert float(np.median(out)) < 0.2

    def test_large_bright_core_preserved(self):
        h, w = 600, 800
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
        img = np.full((h, w), 0.08, np.float32)
        img += np.exp(-r2 / (2 * 50 ** 2)) * 0.9  # large bright core (~100px)
        rng = np.random.default_rng(1)
        for _ in range(60):
            sy, sx = int(rng.integers(20, h - 20)), int(rng.integers(20, w - 20))
            img[sy - 1:sy + 2, sx - 1:sx + 2] += 0.7
        img = np.clip(img, 0, 1).astype(np.float32)
        out = remove_stars_builtin(img, threshold=0.5)
        core = r2 < 25 ** 2
        # The bright core must survive (not become a dark hole).
        assert float(out[core].mean()) > 0.6

    def test_smooth_background_returned_unchanged(self):
        rng = np.random.default_rng(2)
        img = np.clip(rng.normal(0.05, 0.008, (200, 200)).astype(np.float32), 0, 1)
        out = remove_stars_builtin(img, threshold=0.5)
        # No stars present → median must not move.
        assert abs(float(np.median(out)) - float(np.median(img))) < 0.01


class TestNoiseSpecksDoNotFloodBackground:
    """Regression: on a frame with a dark, noisy background and a bright object,
    background grain threw tens of thousands of 1-2px specks above threshold.
    The size filter only removed LARGE blobs, so the specks survived; dilation
    then merged them into a mask covering most of the frame, and the inpainter
    flooded the dark sky with the bright object's median (~0.45). Result: the
    'starless' background was 6x brighter than the input.
    """

    @staticmethod
    def _dark_bg_bright_object(h=700, w=1000, seed=23):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
        img = np.clip(rng.normal(0.07, 0.02, (h, w)).astype(np.float32), 0, 1)
        img[r2 < (0.22 * h) ** 2] = 0.5  # bright object core (~16% of frame)
        for _ in range(80):
            sy, sx = int(rng.integers(15, h - 15)), int(rng.integers(15, w - 15))
            img[sy - 1:sy + 2, sx - 1:sx + 2] = 0.9
        return np.clip(img, 0, 1).astype(np.float32)

    def test_dark_background_not_inflated(self):
        img = self._dark_bg_bright_object()
        out = remove_stars_builtin(img, threshold=0.5)
        # The starless background median must stay near the input, not balloon
        # toward the bright object's value.
        assert abs(float(np.median(out)) - float(np.median(img))) < 0.04
