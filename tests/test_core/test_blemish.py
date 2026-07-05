"""Tests for blemish healing and clone stamp."""

import numpy as np

from astraios.core.blemish import (
    BlemishParams,
    CloneStampParams,
    clone_stamp,
    heal_spot,
)


class TestHealSpot:
    """Tests for the Blemish Blaster port (heal_spot)."""

    def _smooth_background(self, size=120, value=0.3):
        rng = np.random.default_rng(0)
        base = np.full((size, size), value, dtype=np.float32)
        base += rng.normal(0.0, 0.002, size=base.shape).astype(np.float32)
        return np.clip(base, 0.0, 1.0)

    def test_removes_dark_spot_mono(self):
        """Healing a synthetic dark spot on a smooth background removes it."""
        img = self._smooth_background()
        cx, cy = 60, 60
        yy, xx = np.mgrid[0:120, 0:120]
        spot = (xx - cx) ** 2 + (yy - cy) ** 2 <= 8**2
        img[spot] = 0.0

        healed = heal_spot(img, cx, cy, BlemishParams(radius=10, feather=0.3, opacity=1.0))

        assert healed.shape == img.shape
        assert np.isfinite(healed).all()
        # The center of the (formerly) dark spot should now be close to the
        # surrounding background value, not black.
        assert healed[cy, cx] > 0.2

    def test_surroundings_unchanged_beyond_radius(self):
        """Pixels well outside the brush radius must be untouched."""
        img = self._smooth_background()
        cx, cy = 60, 60
        img[cy, cx] = 0.0

        params = BlemishParams(radius=8, feather=0.3, opacity=1.0)
        healed = heal_spot(img, cx, cy, params)

        far_y, far_x = cy + 40, cx + 40
        assert healed[far_y, far_x] == img[far_y, far_x]

    def test_color_image_all_channels(self):
        """A (C, H, W) color image should be healed on every channel."""
        base = np.stack([self._smooth_background(value=v) for v in (0.2, 0.3, 0.4)], axis=0)
        cx, cy = 50, 50
        yy, xx = np.mgrid[0:120, 0:120]
        spot = (xx - cx) ** 2 + (yy - cy) ** 2 <= 6**2
        base[:, spot] = 0.0

        healed = heal_spot(base, cx, cy, BlemishParams(radius=8, feather=0.3, opacity=1.0))

        assert healed.shape == base.shape
        for c in range(3):
            assert healed[c, cy, cx] > 0.1

    def test_edge_click_does_not_crash(self):
        """Clicking at/near the image border must not raise or corrupt data."""
        img = self._smooth_background(size=40)
        for x, y in [(0, 0), (39, 39), (0, 39), (39, 0), (-5, -5), (100, 100)]:
            out = heal_spot(img, x, y, BlemishParams(radius=15))
            assert out.shape == img.shape
            assert np.isfinite(out).all()

    def test_out_of_bounds_is_noop(self):
        """A click point outside the canvas returns the image unmodified."""
        img = self._smooth_background(size=32)
        out = heal_spot(img, -10, -10, BlemishParams(radius=5))
        np.testing.assert_array_equal(out, img)

    def test_opacity_zero_is_noop(self):
        img = self._smooth_background(size=64)
        img[32, 32] = 0.0
        out = heal_spot(img, 32, 32, BlemishParams(radius=6, opacity=0.0))
        np.testing.assert_allclose(out, img, atol=1e-6)


class TestCloneStamp:
    """Tests for the Clone Stamp port."""

    def test_copies_source_patch(self):
        """Cloning should paint the source region onto the destination."""
        img = np.zeros((100, 100), dtype=np.float32)
        # A bright square patch used as the clone source.
        img[10:30, 10:30] = 0.8

        src = (20, 20)
        dst = (70, 70)
        out = clone_stamp(img, src, dst, CloneStampParams(radius=8, feather=0.0, opacity=1.0))

        # The destination point should now resemble the bright source patch.
        assert out[70, 70] > 0.5
        # Untouched region (far from both source and destination) unchanged.
        assert out[5, 5] == img[5, 5]

    def test_color_image(self):
        img = np.zeros((3, 80, 80), dtype=np.float32)
        img[:, 10:25, 10:25] = np.array([0.9, 0.1, 0.1]).reshape(3, 1, 1)

        params = CloneStampParams(radius=6, feather=0.0, opacity=1.0)
        out = clone_stamp(img, (17, 17), (60, 60), params)
        assert out.shape == img.shape
        assert out[0, 60, 60] > 0.5
        assert out[1, 60, 60] < 0.5

    def test_edge_click_does_not_crash(self):
        img = np.random.default_rng(1).random((50, 50)).astype(np.float32)
        for src, dst in [((0, 0), (49, 49)), ((-5, -5), (60, 60)), ((25, 25), (0, 0))]:
            out = clone_stamp(img, src, dst, CloneStampParams(radius=10))
            assert out.shape == img.shape
            assert np.isfinite(out).all()

    def test_opacity_zero_is_noop(self):
        img = np.random.default_rng(2).random((60, 60)).astype(np.float32)
        out = clone_stamp(img, (10, 10), (40, 40), CloneStampParams(radius=8, opacity=0.0))
        np.testing.assert_allclose(out, img, atol=1e-6)
