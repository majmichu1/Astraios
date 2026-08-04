"""Tests for mosaic stitching."""

import numpy as np
import pytest

from astraios.core.mosaic import (
    BlendMethod,
    MosaicParams,
    MosaicResult,
    _multiband_blend_plane,
    _panel_weight,
    _pyramid_levels,
    mosaic_stitch,
)


def _star_panel(shape=(50, 50), seed: int = 0) -> np.ndarray:
    """Panel with synthetic stars — mosaic registration needs detectable,
    matchable star patterns, so stitch tests cannot use pure noise."""
    rng = np.random.default_rng(seed)
    img = np.full(shape, 0.1, dtype=np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for _ in range(8):
        x = rng.uniform(6, shape[1] - 6)
        y = rng.uniform(6, shape[0] - 6)
        img += (0.6 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.5)).astype(np.float32)
    return img.astype(np.float32)


class TestMosaicStitch:
    def test_two_identical_panels(self):
        panel = _star_panel()
        result = mosaic_stitch([panel, panel.copy()])
        assert isinstance(result, MosaicResult)
        assert result.n_panels == 2

    def test_output_in_range(self):
        p1 = _star_panel(seed=1)
        result = mosaic_stitch([p1, p1.copy()])
        assert result.data.min() >= 0.0
        assert result.data.max() <= 1.0

    def test_color_panels(self):
        mono = _star_panel(seed=2)
        p1 = np.stack([mono, mono * 0.9, mono * 0.8]).astype(np.float32)
        result = mosaic_stitch([p1, p1.copy()])
        assert result.data.ndim == 3
        assert result.data.shape[0] == 3

    def test_too_few_panels_raises(self):
        with pytest.raises(ValueError):
            mosaic_stitch([np.zeros((50, 50), dtype=np.float32)])

    def test_three_panels(self):
        base = _star_panel(seed=3)
        result = mosaic_stitch([base, base.copy(), base.copy()])
        assert result.n_panels == 3

    def test_unregistrable_panel_raises(self):
        """Regression: a panel that could not be registered used to get the
        identity transform and was silently blended at its raw pixel
        position, producing a visibly wrong mosaic."""
        flat = [np.zeros((40, 40), dtype=np.float32) for _ in range(2)]
        with pytest.raises(ValueError, match="could not be registered"):
            mosaic_stitch(flat)

    def test_transitive_registration_links_corner_panel(self):
        """Regression: panels were matched only against panel 0, so a panel
        overlapping just an intermediate panel (e.g. the far corner of a
        2x2 mosaic) failed registration. It must be linked transitively."""
        from astraios.core.mosaic import _compute_pairwise_transforms

        rng = np.random.default_rng(7)
        img = np.full((90, 240), 0.1, dtype=np.float32)
        yy, xx = np.mgrid[0:90, 0:240]
        for _ in range(30):
            x, y = rng.uniform(6, 234), rng.uniform(6, 84)
            img += (0.6 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 2.5)).astype(np.float32)
        img = img.astype(np.float32)

        p0 = img[:, 0:100]    # left
        p1 = img[:, 70:170]   # middle — overlaps both neighbours
        p2 = img[:, 140:240]  # right — overlaps p1 only, not p0

        transforms = _compute_pairwise_transforms([p0, p1, p2], lambda f, m: None)
        identity = np.eye(2, 3, dtype=np.float32)
        assert not np.allclose(transforms[2], identity, atol=1e-2), (
            "corner panel fell back to identity instead of being linked "
            "through the middle panel"
        )


class TestBlendMethods:
    """`blend_method` was offered in the mosaic dialog but ignored by the
    stitcher, which always feathered. All three modes are real now.
    """

    @staticmethod
    def _two_panels(overlap=64, offset=0.06):
        """Two half-canvas panels with a brightness step between them."""
        h, w = 128, 256
        yy, xx = np.mgrid[0:h, 0:w]
        base = (0.3 + 0.2 * np.sin(xx / 40.0) + 0.1 * np.cos(yy / 30.0)).astype(np.float32)
        mid, half = w // 2, overlap // 2
        mask_a = (xx < mid + half).astype(np.float32)
        mask_b = (xx > mid - half).astype(np.float32)
        return (base * mask_a).astype(np.float32), \
               ((base + offset) * mask_b).astype(np.float32), mask_a, mask_b

    def _blend(self, method, overlap=64):
        pa, pb, ma, mb = self._two_panels(overlap)
        params = MosaicParams(blend_method=method, feather_width=50)
        wa, wb = _panel_weight(ma, params), _panel_weight(mb, params)
        if method == BlendMethod.MULTIBAND:
            return _multiband_blend_plane([pa, pb], [wa, wb],
                                          _pyramid_levels(pa.shape))
        num, den = pa * wa + pb * wb, wa + wb
        return np.where(den > 0, num / np.maximum(den, 1e-6), 0).astype(np.float32)

    def test_the_three_methods_differ(self):
        avg = self._blend(BlendMethod.AVERAGE)
        fea = self._blend(BlendMethod.FEATHER)
        mbd = self._blend(BlendMethod.MULTIBAND)
        assert not np.allclose(avg, fea), "AVERAGE and FEATHER are identical"
        assert not np.allclose(fea, mbd), "MULTIBAND is not doing anything"

    def test_feather_beats_a_hard_average_at_the_seam(self):
        def seam(img):
            return float(np.abs(np.diff(img[:, 68:188], axis=1)).max())

        assert seam(self._blend(BlendMethod.FEATHER)) < \
               seam(self._blend(BlendMethod.AVERAGE))

    def test_multiband_wins_on_a_narrow_overlap(self):
        """The case feathering cannot handle: too little room to ramp."""
        def seam(img):
            return float(np.abs(np.diff(img[:, 68:188], axis=1)).max())

        narrow = 8
        assert seam(self._blend(BlendMethod.MULTIBAND, narrow)) < \
               seam(self._blend(BlendMethod.FEATHER, narrow))

    def test_multiband_output_is_finite_and_bounded(self):
        out = self._blend(BlendMethod.MULTIBAND)
        assert np.isfinite(out).all()
        assert out.min() > -0.5 and out.max() < 1.5

    def test_stitch_runs_with_every_blend_method(self):
        base = _star_panel(shape=(80, 80), seed=4)
        panels = [base, base.copy()]
        for method in (BlendMethod.AVERAGE, BlendMethod.FEATHER, BlendMethod.MULTIBAND):
            res = mosaic_stitch(panels, MosaicParams(blend_method=method))
            assert np.isfinite(res.data).all()
