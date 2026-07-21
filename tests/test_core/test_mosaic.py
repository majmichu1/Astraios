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
        rng = np.random.default_rng(0)
        base = rng.random((80, 80)).astype(np.float32) * 0.5 + 0.2
        panels = [base, base.copy()]
        for method in (BlendMethod.AVERAGE, BlendMethod.FEATHER, BlendMethod.MULTIBAND):
            res = mosaic_stitch(panels, MosaicParams(blend_method=method))
            assert np.isfinite(res.data).all()
