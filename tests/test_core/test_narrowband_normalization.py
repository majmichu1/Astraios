"""Tests for narrowband channel normalization (SHO/HSO/HOS/HOO)."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.narrowband_normalization import (
    MissingChannelsError,
    NBNParams,
    _adev,
    _apply_hl_reduction_and_brightness_and_recover,
    _compute_m_e0,
    _mtf,
    _normalize_channel,
    _rescale,
    normalize_narrowband,
)

# Finishing-stage settings that make `_apply_hl_reduction_and_brightness_and_recover`
# an identity (aside from clipping) — see module math: with hlreduct=brightness=
# hlrecover=1.0, the midtone pivot lands exactly at 0.5, where the MTF is the
# identity function. This lets tests inspect the normalization core directly
# through the public API.
_IDENTITY_FINISH = {"hlrecover": 1.0, "hlreduct": 1.0, "brightness": 1.0}


def _synthetic_channel(rng, size, median_level, scale=0.05):
    """A smooth-ish synthetic mono channel clustered around `median_level`."""
    base = rng.normal(loc=median_level, scale=scale, size=(size, size)).astype(np.float32)
    return np.clip(base, 0.0, 1.0)


class TestLowLevelMath:
    def test_mtf_formula(self):
        # m=0.25, x=0.5 -> (m-1)*x / ((2m-1)*x - m) = -0.375 / -0.5 = 0.75
        assert _mtf(0.25, np.array([0.5], dtype=np.float32))[0] == pytest.approx(0.75, abs=1e-5)

    def test_mtf_midpoint_is_identity(self):
        # m=0.5 is a singular-looking case in the raw formula but simplifies to identity.
        x = np.array([0.0, 0.1, 0.37, 0.5, 0.9, 1.0], dtype=np.float32)
        np.testing.assert_allclose(_mtf(0.5, x), x, atol=1e-5)

    def test_rescale_formula(self):
        out = _rescale(np.array([0.6], dtype=np.float32), 0.2, 1.0)
        assert out[0] == pytest.approx(0.5, abs=1e-6)

    def test_rescale_clips(self):
        out = _rescale(np.array([-1.0, 2.0], dtype=np.float32), 0.0, 1.0)
        np.testing.assert_allclose(out, [0.0, 1.0])

    def test_adev_matches_manual_median_absolute_deviation(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert _adev(arr) == pytest.approx(1.0, abs=1e-6)

    def test_compute_m_e0_matches_manual_formula(self):
        ch0 = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        ch1 = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        m, e0 = _compute_m_e0((ch0, ch1), blackpoint=0.5)
        # ch0: min=0, med=0.4, mean=0.4, adev = median(|x-0.4|) = median([0.4,0.2,0,0.2,0.4]) = 0.2
        expected_m0 = 0.0 + 0.5 * (0.4 - 0.0)
        expected_e0_0 = (0.2 / 1.2533) + 0.4 - expected_m0
        assert m[0] == pytest.approx(expected_m0, abs=1e-5)
        assert e0[0] == pytest.approx(expected_e0_0, abs=1e-4)
        # ch1 is flat: min=med=mean=0.1, adev=0 -> M=0.1, E0=0.1-0.1=0
        assert m[1] == pytest.approx(0.1, abs=1e-6)
        assert e0[1] == pytest.approx(0.0, abs=1e-5)

    def test_finishing_stage_is_identity_at_default_settings(self):
        x = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
        params = NBNParams(scenario="SHO", mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH)
        out = _apply_hl_reduction_and_brightness_and_recover(x, params)
        np.testing.assert_allclose(out, x, atol=1e-5)


class TestScenarioValidation:
    def test_hoo_requires_ha_and_oiii(self):
        params = NBNParams(scenario="HOO", mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH)
        with pytest.raises(MissingChannelsError):
            normalize_narrowband(None, np.zeros((4, 4), dtype=np.float32), None, params)

    def test_sho_requires_all_three_channels(self):
        params = NBNParams(scenario="SHO", mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH)
        ha = np.full((4, 4), 0.3, dtype=np.float32)
        oiii = np.full((4, 4), 0.3, dtype=np.float32)
        with pytest.raises(MissingChannelsError):
            normalize_narrowband(ha, oiii, None, params)

    def test_shape_mismatch_raises(self):
        params = NBNParams(scenario="SHO", mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH)
        ha = np.full((4, 4), 0.3, dtype=np.float32)
        oiii = np.full((8, 8), 0.3, dtype=np.float32)
        sii = np.full((4, 4), 0.3, dtype=np.float32)
        with pytest.raises(ValueError):
            normalize_narrowband(ha, oiii, sii, params)

    def test_unknown_scenario_raises(self):
        params = NBNParams(scenario="XYZ", mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH)
        ha = np.full((4, 4), 0.3, dtype=np.float32)
        with pytest.raises(ValueError):
            normalize_narrowband(ha, ha, ha, params)


class TestOutputShapeAndRange:
    @pytest.mark.parametrize("scenario", ["SHO", "HSO", "HOS", "HOO"])
    def test_output_is_chw_rgb_in_range(self, scenario):
        rng = np.random.default_rng(0)
        ha = _synthetic_channel(rng, 32, 0.4)
        oiii = _synthetic_channel(rng, 32, 0.5)
        sii = _synthetic_channel(rng, 32, 0.3)
        params = NBNParams(
            scenario=scenario, mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH
        )
        out = normalize_narrowband(ha, oiii, sii, params)
        assert out.shape == (3, 32, 32)
        assert out.dtype == np.float32
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_hoo_ignores_sii(self):
        rng = np.random.default_rng(0)
        ha = _synthetic_channel(rng, 16, 0.4)
        oiii = _synthetic_channel(rng, 16, 0.5)
        params = NBNParams(scenario="HOO", mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH)
        out_without_sii = normalize_narrowband(ha, oiii, None, params)
        out_with_sii = normalize_narrowband(ha, oiii, _synthetic_channel(rng, 16, 0.9), params)
        np.testing.assert_allclose(out_without_sii, out_with_sii, atol=1e-5)


class TestNormalizeChannelTargetStatistic:
    """`_normalize_channel` (the rescale+MTF core every scenario calls) is
    anchored at two exact statistics regardless of the boost-derived pivot
    `e1`: a channel's own blackpoint `M` always maps to `M`, and its own
    full-scale value (1.0) always maps to 1.0. This is the documented target
    invariant of PixInsight's rescale()/mtf() combination that SASpro's
    formulas are built from — verified analytically in
    `TestLowLevelMath.test_mtf_midpoint_is_identity` territory, and here
    directly against `_normalize_channel` and the end-to-end public API.
    """

    @pytest.mark.parametrize("e1", [0.1, 0.5, 0.9])
    def test_blackpoint_is_a_fixed_point(self, e1):
        m_val = 0.35
        out = _normalize_channel(np.array([m_val], dtype=np.float32), m_val, e1)
        assert out[0] == pytest.approx(m_val, abs=1e-5)

    @pytest.mark.parametrize("e1", [0.1, 0.5, 0.9])
    def test_full_scale_maps_to_one(self, e1):
        m_val = 0.35
        out = _normalize_channel(np.array([1.0], dtype=np.float32), m_val, e1)
        assert out[0] == pytest.approx(1.0, abs=1e-5)

    def test_below_blackpoint_never_exceeds_blackpoint(self):
        m_val = 0.4
        t = np.array([0.0, 0.1, 0.2, 0.39], dtype=np.float32)
        out = _normalize_channel(t, m_val, e1=0.5)
        assert np.all(out <= m_val + 1e-5)

    def test_end_to_end_preserves_blackpoint_and_saturation_fixed_points(self):
        """Through the public API (SHO, identity finishing): a channel pixel
        sitting exactly at that channel's own M keeps that exact value, and a
        fully-saturated pixel (1.0) stays fully saturated, regardless of the
        boost sliders."""
        rng = np.random.default_rng(5)
        ha = _synthetic_channel(rng, 16, 0.4)
        oiii = _synthetic_channel(rng, 16, 0.5)
        sii = _synthetic_channel(rng, 16, 0.3)
        sii[0, 0] = 1.0  # force one SII pixel to its own theoretical max

        params = NBNParams(
            scenario="SHO",
            mode=0,
            lightness=0,
            blackpoint=0.25,
            siiboost=1.7,
            oiiiboost2=0.6,
            **_IDENTITY_FINISH,
        )
        out = normalize_narrowband(ha, oiii, sii, params)
        assert out[0, 0, 0] == pytest.approx(1.0, abs=1e-4)  # R = normalized SII


class TestSCNR:
    def test_scnr_caps_green_channel(self):
        rng = np.random.default_rng(1)
        ha = _synthetic_channel(rng, 24, 0.5)
        oiii = _synthetic_channel(rng, 24, 0.5)
        sii = _synthetic_channel(rng, 24, 0.5)
        params = NBNParams(
            scenario="SHO", mode=0, lightness=0, blackpoint=0.25, scnr=True, **_IDENTITY_FINISH
        )
        out = normalize_narrowband(ha, oiii, sii, params)
        # Finishing is identity at these settings, so out[1] (G) must satisfy
        # the exact SCNR cap: G = min((R+B)/2, Ha).
        expected_g = np.minimum((out[0] + out[2]) * 0.5, ha)
        np.testing.assert_allclose(out[1], expected_g, atol=1e-5)

    def test_no_scnr_green_is_raw_ha(self):
        rng = np.random.default_rng(1)
        ha = _synthetic_channel(rng, 24, 0.5)
        oiii = _synthetic_channel(rng, 24, 0.5)
        sii = _synthetic_channel(rng, 24, 0.5)
        params = NBNParams(
            scenario="SHO", mode=0, lightness=0, blackpoint=0.25, scnr=False, **_IDENTITY_FINISH
        )
        out = normalize_narrowband(ha, oiii, sii, params)
        np.testing.assert_allclose(out[1], ha, atol=1e-5)


class TestHOOBlendModes:
    def test_blend_modes_produce_different_results(self):
        rng = np.random.default_rng(2)
        ha = _synthetic_channel(rng, 24, 0.4)
        oiii = _synthetic_channel(rng, 24, 0.6)
        outs = []
        for blendmode in (0, 1, 2):
            params = NBNParams(
                scenario="HOO",
                mode=0,
                lightness=0,
                blackpoint=0.25,
                blendmode=blendmode,
                hablend=0.6,
                **_IDENTITY_FINISH,
            )
            outs.append(normalize_narrowband(ha, oiii, None, params))
        assert not np.allclose(outs[0], outs[1])
        assert not np.allclose(outs[1], outs[2])

    def test_hablend_zero_and_one_are_pure_endpoints(self):
        rng = np.random.default_rng(3)
        ha = _synthetic_channel(rng, 24, 0.4)
        oiii = _synthetic_channel(rng, 24, 0.6)
        params_full_ha = NBNParams(
            scenario="HOO", mode=0, lightness=0, blackpoint=0.25,
            blendmode=0, hablend=1.0, **_IDENTITY_FINISH,
        )
        out = normalize_narrowband(ha, oiii, None, params_full_ha)
        # blendmode 0: E4 = t0*hb + E3*(1-hb); hablend=1.0 -> G == Ha exactly.
        np.testing.assert_allclose(out[1], ha, atol=1e-5)


class TestFlatChannelDoesNotCrash:
    def test_all_equal_channel_is_finite(self):
        flat = np.full((8, 8), 0.5, dtype=np.float32)
        params = NBNParams(scenario="SHO", mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH)
        out = normalize_narrowband(flat, flat, flat, params)
        assert np.all(np.isfinite(out))

    def test_zero_channel_is_finite(self):
        zero = np.zeros((8, 8), dtype=np.float32)
        params = NBNParams(scenario="SHO", mode=0, lightness=0, blackpoint=0.25, **_IDENTITY_FINISH)
        out = normalize_narrowband(zero, zero, zero, params)
        assert np.all(np.isfinite(out))


class TestNonLinearMode:
    @pytest.mark.parametrize("scenario", ["SHO", "HSO", "HOS", "HOO"])
    @pytest.mark.parametrize("lightness", [0, 1, 2, 3, 4])
    def test_nonlinear_lightness_options_run_without_error(self, scenario, lightness):
        if scenario == "HOO" and lightness > 3:
            pytest.skip("HOO only has lightness options 0-3")
        rng = np.random.default_rng(7)
        ha = _synthetic_channel(rng, 20, 0.4)
        oiii = _synthetic_channel(rng, 20, 0.5)
        sii = _synthetic_channel(rng, 20, 0.3)
        params = NBNParams(
            scenario=scenario, mode=1, lightness=lightness, blackpoint=0.25, **_IDENTITY_FINISH
        )
        out = normalize_narrowband(ha, oiii, sii if scenario != "HOO" else None, params)
        assert out.shape == (3, 20, 20)
        assert np.all(np.isfinite(out))
