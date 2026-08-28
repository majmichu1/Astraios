"""Controls that used to be drawn but not read (August 2026 audit)."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.ui.panels.tools_panel import ToolsPanel


@pytest.fixture
def panel(qtbot):
    # qtbot.addWidget is unusable here (pytest-qt binding mismatch); the
    # widget is never shown, so nothing needs cleaning up.
    return ToolsPanel()


def test_scnr_target_channel_is_read(panel):
    panel._scnr_target_combo.setCurrentText("Blue")
    assert panel.get_scnr_params().channel == 2
    panel._scnr_target_combo.setCurrentText("Red")
    assert panel.get_scnr_params().channel == 0


def test_background_box_size_is_read(panel):
    panel._bg_box_size_spin.setValue(96)
    assert panel.get_background_params().box_size == 96


def test_alignment_controls_reach_stacking_params(panel):
    panel._star_sens_spin.setValue(7.5)
    panel._max_shift_spin.setValue(80)
    panel._ransac_thresh_spin.setValue(2.0)
    panel._comet_radius_spin.setValue(25)
    a = panel.get_alignment_params()
    assert a["star_sensitivity"] == 7.5 and a["max_shift"] == 80
    assert a["ransac_threshold"] == 2.0 and a["comet_nucleus_radius"] == 25
    s = panel.get_stacking_params()
    assert s.star_sigma_threshold == 7.5 and s.star_max_match_dist == 80.0
    assert s.ransac_threshold == 2.0 and s.comet_nucleus_radius == 25


def test_wiener_selects_the_wiener_method(panel):
    panel._deconv_method_combo.setCurrentText("Wiener")
    assert panel.get_deconvolution_params().method == "wiener"
    panel._deconv_method_combo.setCurrentText("Richardson-Lucy")
    assert panel.get_deconvolution_params().method == "rl"


def test_tiled_inference_checkbox_turns_tiling_off(panel):
    panel._ai_tile_combo.setCurrentText("256")
    panel._ai_tiled_check.setChecked(False)
    assert panel.get_ai_denoise_params().tile_size == 0
    panel._ai_tiled_check.setChecked(True)
    assert panel.get_ai_denoise_params().tile_size == 256


def test_lrgb_controls_reach_params(panel):
    panel._lrgb_lum_weight.setValue(0.6)
    panel._lrgb_sat_boost.setValue(1.5)
    panel._lrgb_chroma_nr.setValue(0.3)
    p = panel.get_lrgb_params()
    assert (p.luminance_weight, p.saturation_boost, p.chrominance_noise) == pytest.approx((0.6, 1.5, 0.3))


def test_denoise_simple_methods_grey_out_the_unused_sliders(panel):
    panel._denoise_method_combo.setCurrentText("TGV Denoise")
    assert not panel._denoise_lum.isEnabled() and not panel._denoise_chrom.isEnabled()
    panel._denoise_method_combo.setCurrentText("Wavelet Denoise")
    assert panel._denoise_lum.isEnabled()
    panel._denoise_chrom.setChecked(True)
    assert panel.get_denoise_params().chrominance_only is True


def test_calibration_pickers_feed_sources(panel, tmp_path):
    master = tmp_path / "master_dark.fits"
    master.write_bytes(b"")
    panel._cal_sources["dark"] = {"paths": [], "master": master}
    panel._set_cal_label("dark", master.name)
    src = panel.get_calibration_sources()
    assert src["dark_master"] == master and src["bias_paths"] == []
    assert "master_dark.fits" in panel._cal_dark_label.text()


def test_scnr_core_neutralises_the_chosen_channel():
    from astraios.core.color_tools import SCNRParams, scnr

    img = np.full((3, 8, 8), 0.3, np.float32)
    img[0] += 0.2  # red cast
    out = scnr(img, SCNRParams(channel=0, amount=1.0, preserve_luminance=False))
    assert out[0].mean() < img[0].mean() and np.allclose(out[1], img[1])


def test_wiener_deconvolution_sharpens_a_blurred_star():
    from astraios.core.deconvolution import DeconvolutionParams, wiener_deconvolve

    yy, xx = np.mgrid[0:64, 0:64]
    star = np.exp(-((yy - 32) ** 2 + (xx - 32) ** 2) / (2 * 2.0**2)).astype(np.float32) * 0.6
    out = wiener_deconvolve(star, DeconvolutionParams(psf_fwhm=3.0, method="wiener", regularization=0.01))
    assert np.isfinite(out).all() and out.shape == star.shape
    assert out[32, 32] > star[32, 32]


def test_spcc_panel_builds_a_real_sfcc_configuration(panel):
    from astraios.core.sfcc import FILTER_CURVES, SENSOR_QE_CURVES, SFCCParams

    panel._spcc_filter_combo.setCurrentText("OSC (Bayer RGB)")
    panel._spcc_sensor_combo.setCurrentText("CCD (Kodak KAF)")
    panel._spcc_lp_combo.setCurrentText("UV/IR Cut (generic, 400-700nm)")
    p = panel.get_spcc_params()
    assert isinstance(p, SFCCParams)
    assert p.filter_r in FILTER_CURVES and p.filter_g in FILTER_CURVES and p.filter_b in FILTER_CURVES
    assert p.sensor in SENSOR_QE_CURVES and p.lp_filter_1 in FILTER_CURVES
    panel._spcc_filter_combo.setCurrentText("Broadband (LRGB interference filters)")
    assert panel.get_spcc_params().filter_b == "Broadband-B (generic LRGB interference)"


def test_ignore_black_pixels_checkbox_reaches_stacking_params(panel):
    assert panel.get_stacking_params().ignore_black_pixels is True
    panel._ignore_black_check.setChecked(False)
    assert panel.get_stacking_params().ignore_black_pixels is False
