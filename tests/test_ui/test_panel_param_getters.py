"""Regression: several tools-panel param getters were missing or passed wrong
types, so the corresponding operations crashed when invoked from the GUI
(median filter, MLT, chromatic aberration, morphology, color adjust, stretch/
GHS reset). These pin the getters down."""

from __future__ import annotations

import pytest


@pytest.fixture
def panel(qtbot):
    from cosmica.ui.panels.tools_panel import ToolsPanel

    return ToolsPanel()


def test_morphology_params_are_enums(panel):
    from cosmica.core.morphology import MorphOp, StructuringElement

    p = panel.get_morphology_params()
    assert isinstance(p.operation, MorphOp)
    assert isinstance(p.element, StructuringElement)
    # .name must work (this used to be a 'str has no attribute name' crash).
    assert p.operation.name and p.element.name


def test_median_filter_params(panel):
    from cosmica.core.filters import MedianFilterParams

    p = panel.get_median_filter_params()
    assert isinstance(p, MedianFilterParams)
    assert p.kernel_size % 2 == 1 and p.kernel_size >= 3


def test_mlt_params_have_noise_thresholds(panel):
    from cosmica.core.wavelets import WaveletParams

    p = panel.get_mlt_params()
    assert isinstance(p, WaveletParams)
    assert hasattr(p, "noise_thresholds")


def test_ca_params(panel):
    from cosmica.core.chromatic_aberration import CAParams

    assert isinstance(panel.get_ca_params(), CAParams)


def test_color_adjust_params_neutral_default(panel):
    from cosmica.core.color_tools import ColorAdjustParams

    p = panel.get_color_adjust_params()
    assert isinstance(p, ColorAdjustParams)
    # Sliders centred on 0 must map to a neutral (1.0) saturation, not 0 (grey).
    assert abs(p.saturation - 1.0) < 1e-6
    assert hasattr(p, "hue_shift") and not hasattr(p, "hue")


def test_reset_methods_exist(panel):
    panel.reset_stretch_params()
    panel.reset_ghs_params()
