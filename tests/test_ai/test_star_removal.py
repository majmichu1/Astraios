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
