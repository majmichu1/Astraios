"""Tests for narrowband processing."""

import numpy as np

from astraios.core.narrowband import (
    NarrowbandPalette,
    NarrowbandParams,
    combine_narrowband,
    continuum_subtraction,
)


class TestCombineNarrowband:
    def test_sho_palette(self):
        ha = np.ones((100, 100), dtype=np.float32) * 0.5
        oiii = np.ones((100, 100), dtype=np.float32) * 0.3
        sii = np.ones((100, 100), dtype=np.float32) * 0.2

        result = combine_narrowband(
            {"ha": ha, "oiii": oiii, "sii": sii},
            NarrowbandParams(palette=NarrowbandPalette.SHO),
        )
        assert result.shape == (3, 100, 100)
        # SHO: R=SII, G=Ha, B=OIII
        # After normalization, G should be brightest (Ha=0.5)
        assert result[1].mean() > result[0].mean()

    def test_hoo_palette(self):
        ha = np.ones((100, 100), dtype=np.float32) * 0.8
        oiii = np.ones((100, 100), dtype=np.float32) * 0.4

        result = combine_narrowband(
            {"ha": ha, "oiii": oiii},
            NarrowbandParams(palette=NarrowbandPalette.HOO),
        )
        # HOO: R=Ha, G=B=OIII
        assert result.shape == (3, 100, 100)

    def test_output_normalized(self):
        ha = np.ones((100, 100), dtype=np.float32) * 0.9
        oiii = np.ones((100, 100), dtype=np.float32) * 0.5
        sii = np.ones((100, 100), dtype=np.float32) * 0.3

        result = combine_narrowband({"ha": ha, "oiii": oiii, "sii": sii})
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_missing_sii(self):
        ha = np.ones((100, 100), dtype=np.float32) * 0.5
        oiii = np.ones((100, 100), dtype=np.float32) * 0.3

        result = combine_narrowband({"ha": ha, "oiii": oiii})
        assert result.shape == (3, 100, 100)

    def test_ha_required(self):
        import pytest
        oiii = np.ones((100, 100), dtype=np.float32) * 0.3
        with pytest.raises(ValueError):
            combine_narrowband({"oiii": oiii})


class TestContinuumSubtraction:
    def test_subtracts_continuum(self):
        narrowband = np.ones((100, 100), dtype=np.float32) * 0.8
        broadband = np.ones((100, 100), dtype=np.float32) * 0.3

        result = continuum_subtraction(narrowband, broadband, scale=1.0)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.01)

    def test_output_clipped(self):
        narrowband = np.ones((100, 100), dtype=np.float32) * 0.2
        broadband = np.ones((100, 100), dtype=np.float32) * 0.5

        result = continuum_subtraction(narrowband, broadband)
        assert result.min() >= 0.0

    def test_scale_factor(self):
        narrowband = np.ones((100, 100), dtype=np.float32) * 0.8
        broadband = np.ones((100, 100), dtype=np.float32) * 0.4

        r1 = continuum_subtraction(narrowband, broadband, scale=0.5)
        r2 = continuum_subtraction(narrowband, broadband, scale=1.0)
        assert r1.mean() > r2.mean()  # less subtracted with lower scale


class TestExtendedPalettes:
    """Perfect-Palette-Picker additions: extra linear maps + dynamic palettes."""

    @staticmethod
    def _chans():
        ha = np.full((32, 40), 0.6, np.float32)
        oiii = np.full((32, 40), 0.3, np.float32)
        sii = np.full((32, 40), 0.1, np.float32)
        return {"ha": ha, "oiii": oiii, "sii": sii}

    def test_linear_palettes_map_channels(self):
        # With normalize=False the mapping is exact and checkable.
        ch = self._chans()
        cases = {
            NarrowbandPalette.HSO: (ch["ha"], ch["sii"], ch["oiii"]),
            NarrowbandPalette.OHS: (ch["oiii"], ch["ha"], ch["sii"]),
            NarrowbandPalette.OSH: (ch["oiii"], ch["sii"], ch["ha"]),
        }
        for palette, (r, g, b) in cases.items():
            out = combine_narrowband(ch, NarrowbandParams(palette=palette, normalize=False))
            assert out.shape == (3, 32, 40)
            np.testing.assert_allclose(out[0], r, atol=1e-6)
            np.testing.assert_allclose(out[1], g, atol=1e-6)
            np.testing.assert_allclose(out[2], b, atol=1e-6)

    def test_foraxx_dynamic_runs_and_blends(self):
        ch = self._chans()
        out = combine_narrowband(
            ch, NarrowbandParams(palette=NarrowbandPalette.FORAXX, normalize=False)
        )
        assert out.shape == (3, 32, 40)
        assert out.min() >= 0.0 and out.max() <= 1.0
        # R=Ha, B=OIII exactly; G is a blend strictly between them here.
        np.testing.assert_allclose(out[0], ch["ha"], atol=1e-6)
        np.testing.assert_allclose(out[2], ch["oiii"], atol=1e-6)
        g = float(out[1].mean())
        assert min(0.3, 0.6) - 1e-6 <= g <= max(0.3, 0.6) + 1e-6
        # Green must differ from a flat SHO (which would just be Ha=0.6).
        assert abs(g - 0.6) > 1e-3

    def test_dynamic_sho_uses_sii_for_red(self):
        ch = self._chans()
        out = combine_narrowband(
            ch, NarrowbandParams(palette=NarrowbandPalette.DYNAMIC_SHO, normalize=False)
        )
        np.testing.assert_allclose(out[0], ch["sii"], atol=1e-6)
        np.testing.assert_allclose(out[2], ch["oiii"], atol=1e-6)

    def test_dynamic_green_tracks_oiii_extremes(self):
        # factor = OIII^(1-OIII): at OIII=1 -> green=Ha; at OIII=0 -> green=OIII=0.
        ha = np.full((8, 8), 0.7, np.float32)
        hi = combine_narrowband(
            {"ha": ha, "oiii": np.ones((8, 8), np.float32)},
            NarrowbandParams(palette=NarrowbandPalette.FORAXX, normalize=False),
        )
        lo = combine_narrowband(
            {"ha": ha, "oiii": np.zeros((8, 8), np.float32)},
            NarrowbandParams(palette=NarrowbandPalette.FORAXX, normalize=False),
        )
        np.testing.assert_allclose(hi[1], ha, atol=1e-5)
        np.testing.assert_allclose(lo[1], np.zeros((8, 8), np.float32), atol=1e-5)
