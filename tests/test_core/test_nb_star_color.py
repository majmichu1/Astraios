"""Tests for narrowband-to-RGB star color recombination."""

import numpy as np

from astraios.core.nb_star_color import NBStarColorParams, recombine_star_color


def _star_mask(size, centers, radius=3):
    yy, xx = np.mgrid[0:size, 0:size]
    m = np.zeros((size, size), dtype=bool)
    for cx, cy in centers:
        m |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    return m


class TestRecombineStarColor:
    def test_shape_and_finite(self):
        size = 48
        ha = np.full((size, size), 0.2, dtype=np.float32)
        stars = np.zeros((3, size, size), dtype=np.float32)
        out = recombine_star_color(ha, stars)
        assert out.shape == (3, size, size)
        assert np.isfinite(out).all()
        assert out.dtype == np.float32

    def test_mono_ha_only_star_color_replaces_odd_star_hue(self):
        """With Ha-only NB input (no color info) and a stars-only broadband
        frame, R and B should come straight from the broadband star frame
        (SASpro's algorithm: sii/oiii both fall back to the broadband
        channels when absent), i.e. natural star color replaces the
        colorless narrowband star.
        """
        size = 64
        centers = [(20, 20), (45, 40)]
        stars_mask = _star_mask(size, centers)

        # Ha-only composite: nebulosity plus "stars" that carry no color
        # information at all (same value everywhere -> would render gray).
        ha = np.full((size, size), 0.15, dtype=np.float32)
        ha[stars_mask] = 0.95

        # Broadband stars-only frame: zero background, natural blue-white
        # stars.
        r = np.zeros((size, size), dtype=np.float32)
        g = np.zeros((size, size), dtype=np.float32)
        b = np.zeros((size, size), dtype=np.float32)
        r[stars_mask] = 0.85
        g[stars_mask] = 0.20
        b[stars_mask] = 0.95
        rgb_stars = np.stack([r, g, b], axis=0)

        params = NBStarColorParams(
            ratio=0.3, enable_star_stretch=False, apply_scnr=False, saturation=1.0
        )
        out = recombine_star_color(ha, rgb_stars, params)

        cx, cy = centers[0]
        # Star pixel: R and B channels are a direct pass-through of the
        # broadband star color (sii falls back to r, oiii falls back to b).
        np.testing.assert_allclose(out[0, cy, cx], r[cy, cx], atol=1e-5)
        np.testing.assert_allclose(out[2, cy, cx], b[cy, cx], atol=1e-5)
        # Green channel blends Ha with broadband green by `ratio`.
        expected_g = 0.3 * ha[cy, cx] + 0.7 * g[cy, cx]
        np.testing.assert_allclose(out[1, cy, cx], expected_g, atol=1e-5)

        # The star is no longer colorless/gray: channels differ noticeably.
        assert abs(float(out[0, cy, cx]) - float(out[1, cy, cx])) > 0.05

    def test_nebulosity_preserved_in_green_channel(self):
        """Away from stars (where the broadband frame is zero), the Ha
        structure must survive (scaled by `ratio`) in the green channel
        rather than being wiped out by the star frame.
        """
        size = 64
        centers = [(20, 20)]
        stars_mask = _star_mask(size, centers)

        yy, xx = np.mgrid[0:size, 0:size]
        ha = (0.1 + 0.05 * np.sin(xx / 5.0) * np.cos(yy / 7.0)).astype(np.float32)
        ha = np.clip(ha, 0.0, 1.0)
        ha_neb_only = ha.copy()
        ha[stars_mask] = 0.95

        rgb_stars = np.zeros((3, size, size), dtype=np.float32)
        rgb_stars[:, stars_mask] = 0.9

        params = NBStarColorParams(ratio=0.4, enable_star_stretch=False, apply_scnr=False)
        out = recombine_star_color(ha, rgb_stars, params)

        neb = ~stars_mask
        expected_g_neb = 0.4 * ha_neb_only[neb]
        np.testing.assert_allclose(out[1][neb], expected_g_neb, atol=1e-5)
        # Nebulosity green channel is not flat/zero -- structure preserved.
        assert np.std(out[1][neb]) > 1e-3

    def test_stacked_ha_oiii_sii_input(self):
        """A (3, H, W) nb_image stack (Ha, OIII, SII) is accepted and each
        channel participates per the documented formula.
        """
        size = 32
        ha = np.full((size, size), 0.5, dtype=np.float32)
        oiii = np.full((size, size), 0.1, dtype=np.float32)
        sii = np.full((size, size), 0.6, dtype=np.float32)
        nb = np.stack([ha, oiii, sii], axis=0)

        rgb_stars = np.full((3, size, size), 0.2, dtype=np.float32)

        out = recombine_star_color(
            nb, rgb_stars, NBStarColorParams(ratio=0.5, enable_star_stretch=False, apply_scnr=False)
        )

        expected_r = 0.5 * 0.2 + 0.5 * 0.6
        expected_g = 0.5 * 0.5 + 0.5 * 0.2
        expected_b = 0.1  # OIII provided directly -> used as-is.
        np.testing.assert_allclose(out[0], expected_r, atol=1e-5)
        np.testing.assert_allclose(out[1], expected_g, atol=1e-5)
        np.testing.assert_allclose(out[2], expected_b, atol=1e-5)

    def test_star_stretch_boosts_and_stays_bounded(self):
        size = 24
        ha = np.full((size, size), 0.3, dtype=np.float32)
        rgb_stars = np.full((3, size, size), 0.3, dtype=np.float32)

        plain = recombine_star_color(
            ha, rgb_stars, NBStarColorParams(enable_star_stretch=False, apply_scnr=False)
        )
        stretched = recombine_star_color(
            ha,
            rgb_stars,
            NBStarColorParams(enable_star_stretch=True, stretch_factor=5.0, apply_scnr=False),
        )
        assert np.all(stretched >= plain - 1e-6)
        assert stretched.max() <= 1.0 + 1e-6
        assert stretched.min() >= 0.0 - 1e-6

    def test_scnr_reduces_green_excess(self):
        size = 16
        ha = np.zeros((size, size), dtype=np.float32)
        # Broadband green much higher than red/blue everywhere.
        rgb_stars = np.zeros((3, size, size), dtype=np.float32)
        rgb_stars[1] = 0.9
        rgb_stars[0] = 0.1
        rgb_stars[2] = 0.1

        no_scnr = recombine_star_color(
            ha, rgb_stars, NBStarColorParams(ratio=0.0, enable_star_stretch=False, apply_scnr=False)
        )
        with_scnr = recombine_star_color(
            ha,
            rgb_stars,
            NBStarColorParams(
                ratio=0.0, enable_star_stretch=False, apply_scnr=True, scnr_amount=1.0
            ),
        )
        assert no_scnr[1].mean() > with_scnr[1].mean()
        # Full-strength average-neutral SCNR clamps green to (r+b)/2.
        np.testing.assert_allclose(with_scnr[1], (with_scnr[0] + with_scnr[2]) / 2.0, atol=1e-5)
