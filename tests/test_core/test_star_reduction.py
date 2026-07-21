"""Tests for star reduction and star mask generation."""

import numpy as np

from astraios.core.masks import MaskType
from astraios.core.star_reduction import (
    StarReductionParams,
    create_star_mask,
    reduce_stars,
)


def _star_image(n_stars=5):
    """Create a synthetic image with bright stars."""
    image = np.ones((200, 200), dtype=np.float32) * 0.05
    yy, xx = np.mgrid[0:200, 0:200]
    positions = [(40, 40), (160, 40), (100, 100), (40, 160), (160, 160)]
    for sx, sy in positions[:n_stars]:
        dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
        star = 0.9 * np.exp(-dist_sq / (2 * 3.0**2))
        image += star.astype(np.float32)
    return np.clip(image, 0, 1)


class TestCreateStarMask:
    def test_creates_mask(self):
        image = _star_image()
        mask = create_star_mask(image)
        assert mask.mask_type == MaskType.STAR
        assert mask.data.shape == (200, 200)

    def test_stars_are_bright_in_mask(self):
        image = _star_image()
        mask = create_star_mask(image)
        # Star positions should have high mask values
        assert mask.data[40, 40] > 0.3
        assert mask.data[100, 100] > 0.3

    def test_background_is_dark(self):
        image = _star_image()
        mask = create_star_mask(image)
        # Far from any star should be near zero
        assert mask.data[0, 0] < 0.1

    def test_color_image(self):
        mono = _star_image()
        color = np.stack([mono, mono * 0.8, mono * 0.6], axis=0)
        mask = create_star_mask(color)
        assert mask.data.shape == (200, 200)


class TestReduceStars:
    def test_reduces_star_brightness(self):
        image = _star_image()
        result = reduce_stars(image, params=StarReductionParams(amount=1.0, iterations=3))
        # The star as a whole must lose flux. Note this measures the summed
        # star, not the peak pixel: protect_core defaults to True and its whole
        # job is to hold that peak at its original value.
        region = (slice(90, 111), slice(90, 111))
        assert result[region].sum() < image[region].sum()

    def test_peak_is_dimmed_when_core_protection_is_off(self):
        image = _star_image()
        result = reduce_stars(
            image,
            params=StarReductionParams(amount=1.0, iterations=3, protect_core=False),
        )
        assert result[100, 100] < image[100, 100]

    def test_preserves_background(self):
        image = _star_image()
        result = reduce_stars(image)
        # Background far from stars should be similar
        bg_orig = image[0:10, 0:10].mean()
        bg_result = result[0:10, 0:10].mean()
        assert abs(bg_orig - bg_result) < 0.05

    def test_amount_controls_strength(self):
        image = _star_image()
        r_low = reduce_stars(image, params=StarReductionParams(amount=0.2))
        r_high = reduce_stars(image, params=StarReductionParams(amount=1.0))
        # Compare summed star flux rather than the peak pixel, which the
        # default core protection deliberately holds steady.
        region = (slice(90, 111), slice(90, 111))
        assert r_high[region].sum() < r_low[region].sum()

    def test_amount_controls_strength_at_the_peak_without_protection(self):
        image = _star_image()
        r_low = reduce_stars(
            image, params=StarReductionParams(amount=0.2, protect_core=False)
        )
        r_high = reduce_stars(
            image, params=StarReductionParams(amount=1.0, protect_core=False)
        )
        assert r_high[100, 100] < r_low[100, 100]

    def test_color_image(self):
        mono = _star_image()
        color = np.stack([mono, mono, mono], axis=0)
        result = reduce_stars(color)
        assert result.shape == (3, 200, 200)

    def test_output_in_range(self):
        image = _star_image()
        result = reduce_stars(image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestProtectCore:
    """`protect_core` was declared, set by the UI checkbox and the Smart
    Processor, and then never read by the algorithm -- a knob that did
    nothing while its tooltip promised it avoided hollow, doughnut stars.
    """

    @staticmethod
    def _field():
        import numpy as np

        h = w = 160
        yy, xx = np.mgrid[0:h, 0:w]
        bg = 0.02
        img = np.full((h, w), bg, np.float32)
        stars = [(50, 50, 0.95, 3.0), (110, 110, 0.7, 4.0)]
        for x, y, amp, fwhm in stars:
            s = fwhm / 2.3548
            img += (amp * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * s * s)))).astype(
                np.float32
            )
        return np.clip(img, 0, 1), stars, bg

    def test_flag_actually_changes_the_result(self):
        import numpy as np

        from astraios.core.star_reduction import StarReductionParams, reduce_stars

        img, _, _ = self._field()
        on = reduce_stars(img, params=StarReductionParams(amount=0.9, protect_core=True))
        off = reduce_stars(img, params=StarReductionParams(amount=0.9, protect_core=False))
        assert not np.allclose(on, off), "protect_core is being ignored again"

    def test_core_is_preserved_but_star_still_reduced(self):
        import numpy as np

        from astraios.core.star_reduction import StarReductionParams, reduce_stars

        img, stars, bg = self._field()
        on = reduce_stars(img, params=StarReductionParams(amount=0.9, iterations=3,
                                                         protect_core=True))
        off = reduce_stars(img, params=StarReductionParams(amount=0.9, iterations=3,
                                                           protect_core=False))

        def peak(a):
            return float(np.mean([a[y, x] for x, y, _, _ in stars]))

        def flux(a):
            yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            total = 0.0
            for x, y, _, _ in stars:
                sel = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) < 10
                total += float((a[sel] - bg).clip(0).sum())
            return total

        # the core survives far better with protection on ...
        assert peak(on) / peak(img) > peak(off) / peak(img) + 0.15
        assert peak(on) / peak(img) > 0.85
        # ... while the star is still genuinely reduced overall
        assert flux(on) < flux(img) * 0.95

    def test_output_stays_in_range(self):
        import numpy as np

        from astraios.core.star_reduction import StarReductionParams, reduce_stars

        img, _, _ = self._field()
        out = reduce_stars(img, params=StarReductionParams(protect_core=True))
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_color_image_supported(self):
        import numpy as np

        from astraios.core.star_reduction import StarReductionParams, reduce_stars

        img, _, _ = self._field()
        rgb = np.stack([img, img * 0.9, img * 0.8])
        out = reduce_stars(rgb, params=StarReductionParams(protect_core=True))
        assert out.shape == rgb.shape
        assert np.isfinite(out).all()
