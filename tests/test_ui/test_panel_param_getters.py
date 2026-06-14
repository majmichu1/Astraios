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


def test_curves_params_assigns_to_channel(panel):
    from cosmica.core.curves import CurvesParams, CurvePoints

    p = panel.get_curves_params()
    assert isinstance(p, CurvesParams)
    assert isinstance(p.master, CurvePoints)
    assert not hasattr(p, "channel")  # used to pass a bogus 'channel' kwarg


def test_stacking_params_normalization_valid(panel):
    from cosmica.core.stacking import NormalizationMethod, StackingParams

    p = panel.get_stacking_params()
    assert isinstance(p, StackingParams)
    assert isinstance(p.normalization, NormalizationMethod)


def test_every_zero_arg_getter_is_callable(panel):
    """No tools-panel get_*/is_* method should crash with default UI state —
    this is the bug class that broke color_adjust/morphology/median/mlt/ca/
    curves/stacking via wrong kwargs or missing enum members."""
    import inspect

    broken = []
    for name in dir(panel):
        if not (name.startswith("get_") or name.startswith("is_")):
            continue
        m = getattr(panel, name)
        if not callable(m):
            continue
        sig = inspect.signature(m)
        if any(
            pr.default is pr.empty
            and pr.kind in (pr.POSITIONAL_OR_KEYWORD, pr.POSITIONAL_ONLY)
            for pr in sig.parameters.values()
        ):
            continue
        try:
            m()
        except Exception as e:  # noqa: BLE001
            broken.append(f"{name}: {type(e).__name__}: {e}")
    assert not broken, "broken getters: " + "; ".join(broken)


def test_geometric_params_use_enums(panel):
    """rotate/flip/bin handlers call params.<field>.name, so the getters must
    return enum values, not raw ints/strings."""
    from cosmica.core.transforms import BinMode, CropParams, FlipAxis, RotateAngle

    assert isinstance(panel.get_rotate_params().angle, RotateAngle)
    assert isinstance(panel.get_flip_params().axis, FlipAxis)
    assert isinstance(panel.get_bin_params().mode, BinMode)
    # Crop width/height must be ints (0 = full), never None.
    cp = panel.get_crop_params()
    assert isinstance(cp, CropParams)
    assert cp.width is not None and cp.height is not None
