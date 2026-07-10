"""Tests for the Perfect Palette Picker (narrowband -> RGB false-color palettes)."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.palette_picker import (
    PALETTE_LABELS,
    Palette,
    PalettePickerParams,
    apply_palette,
)

_SIZE = 32


def _plane(value: float) -> np.ndarray:
    return np.full((_SIZE, _SIZE), value, dtype=np.float32)


def _random_plane(rng: np.random.Generator) -> np.ndarray:
    return rng.random((_SIZE, _SIZE)).astype(np.float32)


# All non-custom palettes exercised by the "every palette" sweep.
_ALL_NON_CUSTOM = [p for p in Palette if p is not Palette.CUSTOM]


class TestEveryPaletteShapeAndRange:
    @pytest.mark.parametrize("palette", _ALL_NON_CUSTOM, ids=lambda p: p.name)
    def test_shape_and_range(self, palette):
        rng = np.random.default_rng(0)
        ha, oo, si = _random_plane(rng), _random_plane(rng), _random_plane(rng)
        params = PalettePickerParams(palette=palette, linear_input=False)
        out = apply_palette(ha, oo, si, params=params)
        assert out.shape == (3, _SIZE, _SIZE)
        assert out.dtype == np.float32
        assert np.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_all_labels_and_descriptions_present(self):
        from astraios.core.palette_picker import PALETTE_DESCRIPTIONS

        for p in Palette:
            assert p in PALETTE_LABELS
            assert p in PALETTE_DESCRIPTIONS


class TestSHORouting:
    """SHO must route SII->R, Ha->G, OIII->B exactly (SASpro ground truth)."""

    def test_sho_single_channel_routing(self):
        ha = _plane(0.5)
        oo = _plane(0.7)
        si = _plane(0.3)
        params = PalettePickerParams(palette=Palette.SHO, linear_input=False, normalize=False)
        out = apply_palette(ha, oo, si, params=params)
        r, g, b = out[0], out[1], out[2]
        assert np.allclose(r, 0.3, atol=1e-5)  # R = SII
        assert np.allclose(g, 0.5, atol=1e-5)  # G = Ha
        assert np.allclose(b, 0.7, atol=1e-5)  # B = OIII

    @pytest.mark.parametrize(
        "palette,expect",
        [
            (Palette.HOO, (0.5, 0.7, 0.7)),
            (Palette.HSO, (0.5, 0.3, 0.7)),
            (Palette.HOS, (0.5, 0.7, 0.3)),
            (Palette.OSS, (0.7, 0.3, 0.3)),
            (Palette.OHH, (0.7, 0.5, 0.5)),
            (Palette.OSH, (0.7, 0.3, 0.5)),
            (Palette.OHS, (0.7, 0.5, 0.3)),
            (Palette.HSS, (0.5, 0.3, 0.3)),
        ],
    )
    def test_basic_palette_routing(self, palette, expect):
        ha, oo, si = _plane(0.5), _plane(0.7), _plane(0.3)
        params = PalettePickerParams(palette=palette, linear_input=False, normalize=False)
        out = apply_palette(ha, oo, si, params=params)
        for ch, val in zip(out, expect, strict=True):
            assert np.allclose(ch, val, atol=1e-5)


class TestSpecialPalettes:
    def test_realistic_1_formula(self):
        ha, oo, si = _plane(0.4), _plane(0.6), _plane(0.2)
        params = PalettePickerParams(
            palette=Palette.REALISTIC_1, linear_input=False, normalize=False
        )
        out = apply_palette(ha, oo, si, params=params)
        assert np.allclose(out[0], 0.5 * 0.4 + 0.5 * 0.2, atol=1e-5)
        assert np.allclose(out[1], 0.3 * 0.4 + 0.7 * 0.6, atol=1e-5)
        assert np.allclose(out[2], 0.9 * 0.6 + 0.1 * 0.4, atol=1e-5)

    def test_realistic_2_formula(self):
        ha, oo, si = _plane(0.4), _plane(0.6), _plane(0.2)
        params = PalettePickerParams(
            palette=Palette.REALISTIC_2, linear_input=False, normalize=False
        )
        out = apply_palette(ha, oo, si, params=params)
        assert np.allclose(out[0], 0.7 * 0.4 + 0.3 * 0.2, atol=1e-5)
        assert np.allclose(out[1], 0.3 * 0.2 + 0.7 * 0.6, atol=1e-5)
        assert np.allclose(out[2], 0.6, atol=1e-5)

    def test_foraxx_formula(self):
        ha, oo, si = _plane(0.4), _plane(0.6), _plane(0.2)
        params = PalettePickerParams(palette=Palette.FORAXX, linear_input=False, normalize=False)
        out = apply_palette(ha, oo, si, params=params)

        oo_c = np.clip(0.6, 1e-6, 1.0)
        t = oo_c ** (1.0 - oo_c)
        expect_r = t * 0.2 + (1.0 - t) * 0.4
        t2 = 0.4 * 0.6
        expect_g = (t2 ** (1.0 - t2)) * 0.4 + (1.0 - (t2 ** (1.0 - t2))) * 0.6

        assert np.allclose(out[0], expect_r, atol=1e-5)
        assert np.allclose(out[1], expect_g, atol=1e-5)
        assert np.allclose(out[2], 0.6, atol=1e-5)

    def test_foraxx_ha_oiii_only_matches_sii_substituted_formula(self):
        """No SII loaded: SASpro substitutes SII=Ha before the Foraxx branch,
        so Ha+OIII-only Foraxx must equal the 3-channel formula with si=ha."""
        ha, oo = _plane(0.4), _plane(0.6)
        params = PalettePickerParams(palette=Palette.FORAXX, linear_input=False, normalize=False)
        out_two = apply_palette(ha, oo, sii=None, params=params)
        out_three = apply_palette(ha, oo, sii=ha, params=params)
        assert np.allclose(out_two, out_three, atol=1e-6)


class TestCustomWeights:
    def test_custom_identity_matches_hos(self):
        ha, oo, si = _plane(0.5), _plane(0.7), _plane(0.3)
        custom = np.eye(3, dtype=np.float32)  # R=Ha, G=OIII, B=SII == HOS
        params_custom = PalettePickerParams(
            palette=Palette.CUSTOM, custom_weights=custom, linear_input=False, normalize=False
        )
        params_hos = PalettePickerParams(
            palette=Palette.HOS, linear_input=False, normalize=False
        )
        out_custom = apply_palette(ha, oo, si, params=params_custom)
        out_hos = apply_palette(ha, oo, si, params=params_hos)
        assert np.allclose(out_custom, out_hos, atol=1e-6)

    def test_custom_arbitrary_matrix(self):
        ha, oo, si = _plane(1.0), _plane(0.0), _plane(0.0)
        # R = 0.5*Ha + 0.5*OIII, G = SII, B = 0
        custom = np.array(
            [[0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32
        )
        params = PalettePickerParams(
            palette=Palette.CUSTOM, custom_weights=custom, linear_input=False, normalize=False
        )
        out = apply_palette(ha, oo, si, params=params)
        assert np.allclose(out[0], 0.5, atol=1e-5)
        assert np.allclose(out[1], 0.0, atol=1e-5)
        assert np.allclose(out[2], 0.0, atol=1e-5)


class TestMissingChannels:
    def test_missing_sii_substitutes_ha(self):
        ha, oo = _plane(0.6), _plane(0.4)
        params = PalettePickerParams(palette=Palette.SHO, linear_input=False, normalize=False)
        out = apply_palette(ha, oo, sii=None, params=params)
        # SHO: R=SII(->Ha)=0.6, G=Ha=0.6, B=OIII=0.4
        assert np.allclose(out[0], 0.6, atol=1e-5)
        assert np.allclose(out[1], 0.6, atol=1e-5)
        assert np.allclose(out[2], 0.4, atol=1e-5)

    def test_missing_ha_substitutes_sii(self):
        oo, si = _plane(0.4), _plane(0.9)
        params = PalettePickerParams(palette=Palette.SHO, linear_input=False, normalize=False)
        out = apply_palette(None, oo, sii=si, params=params)
        # SHO: R=SII=0.9, G=Ha(->SII)=0.9, B=OIII=0.4
        assert np.allclose(out[0], 0.9, atol=1e-5)
        assert np.allclose(out[1], 0.9, atol=1e-5)
        assert np.allclose(out[2], 0.4, atol=1e-5)

    def test_missing_oiii_raises(self):
        ha = _plane(0.5)
        with pytest.raises(ValueError, match="OIII"):
            apply_palette(ha, None, sii=_plane(0.2))

    def test_missing_ha_and_sii_raises(self):
        with pytest.raises(ValueError):
            apply_palette(None, _plane(0.5), sii=None)

    def test_shape_mismatch_raises(self):
        ha = _plane(0.5)
        oo = np.zeros((16, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            apply_palette(ha, oo, sii=None)


class TestStarsBlend:
    def test_stars_opacity_zero_is_noop(self):
        rng = np.random.default_rng(1)
        ha, oo, si = _random_plane(rng), _random_plane(rng), _random_plane(rng)
        stars = rng.random((3, _SIZE, _SIZE)).astype(np.float32)
        params_no_stars = PalettePickerParams(
            palette=Palette.SHO, linear_input=False, stars_opacity=0.0
        )
        out_without = apply_palette(ha, oo, si, params=params_no_stars)
        out_with_zero_opacity = apply_palette(ha, oo, si, stars=stars, params=params_no_stars)
        assert np.allclose(out_without, out_with_zero_opacity)

    def test_stars_blend_changes_output_and_stays_in_range(self):
        rng = np.random.default_rng(2)
        ha, oo, si = _random_plane(rng), _random_plane(rng), _random_plane(rng)
        stars = rng.random((3, _SIZE, _SIZE)).astype(np.float32)
        params_base = PalettePickerParams(palette=Palette.SHO, linear_input=False)
        params_stars = PalettePickerParams(
            palette=Palette.SHO, linear_input=False, stars_opacity=1.0
        )
        out_base = apply_palette(ha, oo, si, params=params_base)
        out_stars = apply_palette(ha, oo, si, stars=stars, params=params_stars)
        assert out_stars.shape == (3, _SIZE, _SIZE)
        assert out_stars.min() >= 0.0
        assert out_stars.max() <= 1.0
        assert not np.allclose(out_base, out_stars)

    def test_stars_mono_broadcasts(self):
        rng = np.random.default_rng(3)
        ha, oo, si = _random_plane(rng), _random_plane(rng), _random_plane(rng)
        stars_mono = rng.random((_SIZE, _SIZE)).astype(np.float32)
        params = PalettePickerParams(palette=Palette.SHO, linear_input=False, stars_opacity=0.5)
        out = apply_palette(ha, oo, si, stars=stars_mono, params=params)
        assert out.shape == (3, _SIZE, _SIZE)

    def test_stars_shape_mismatch_raises(self):
        ha, oo, si = _plane(0.5), _plane(0.5), _plane(0.5)
        stars = np.zeros((3, 8, 8), dtype=np.float32)
        params = PalettePickerParams(palette=Palette.SHO, stars_opacity=1.0)
        with pytest.raises(ValueError):
            apply_palette(ha, oo, si, stars=stars, params=params)


class TestDeterminism:
    @pytest.mark.parametrize(
        "palette", [Palette.SHO, Palette.FORAXX, Palette.REALISTIC_1, Palette.CUSTOM]
    )
    def test_repeated_calls_match(self, palette):
        rng = np.random.default_rng(4)
        ha, oo, si = _random_plane(rng), _random_plane(rng), _random_plane(rng)
        params = PalettePickerParams(palette=palette, linear_input=True)
        out1 = apply_palette(ha, oo, si, params=params)
        out2 = apply_palette(ha, oo, si, params=params)
        assert np.array_equal(out1, out2)


class TestLinearInputStretch:
    def test_linear_input_true_changes_result_vs_false(self):
        rng = np.random.default_rng(5)
        ha, oo, si = _random_plane(rng), _random_plane(rng), _random_plane(rng)
        params_raw = PalettePickerParams(palette=Palette.SHO, linear_input=False)
        params_stretched = PalettePickerParams(palette=Palette.SHO, linear_input=True)
        out_raw = apply_palette(ha, oo, si, params=params_raw)
        out_stretched = apply_palette(ha, oo, si, params=params_stretched)
        assert not np.allclose(out_raw, out_stretched)
        assert out_stretched.min() >= 0.0
        assert out_stretched.max() <= 1.0
