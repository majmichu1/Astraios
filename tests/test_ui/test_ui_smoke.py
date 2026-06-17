"""Smoke tests for Astraios UI components — no Qt dependency in tests."""

from __future__ import annotations

import numpy as np
import pytest

from astraios.core.image_io import ImageData
from astraios.core.stretch import StretchParams


def _make_test_image(w=256, h=256, channels=3) -> ImageData:
    """Create a synthetic gradient test image."""
    if channels == 1:
        data = np.linspace(0, 1, w * h, dtype=np.float32).reshape(h, w)
    else:
        r = np.linspace(0, 1, w * h, dtype=np.float32).reshape(h, w)
        g = np.linspace(1, 0, w * h, dtype=np.float32).reshape(h, w)
        b = np.full((h, w), 0.5, dtype=np.float32)
        data = np.stack([r, g, b], axis=0)
    return ImageData(data=data)


# ── Core dataclasses (no Qt needed) ──────────────────────────────

class TestStretchParams:
    def test_defaults(self):
        params = StretchParams()
        assert params.shadow_clip == -2.8
        assert params.midtone == 0.25
        assert params.linked is True


class TestImageData:
    def test_mono_image(self):
        data = np.zeros((64, 64), dtype=np.float32)
        img = ImageData(data=data)
        assert not img.is_color
        assert img.channels == 1
        assert img.height == 64
        assert img.width == 64

    def test_color_image(self):
        data = np.zeros((3, 64, 64), dtype=np.float32)
        img = ImageData(data=data)
        assert img.is_color
        assert img.channels == 3
        assert "3ch" in img.shape_str

    def test_to_display_mono(self):
        data = np.full((64, 64), 0.5, dtype=np.float32)
        img = ImageData(data=data)
        rgb = img.to_display(stretch=False)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_to_display_color(self):
        data = np.full((3, 64, 64), 0.5, dtype=np.float32)
        img = ImageData(data=data)
        rgb = img.to_display(stretch=False)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_image_properties(self):
        data = np.zeros((3, 100, 200), dtype=np.float32)
        header = {"EXPTIME": 120.0, "CCD-TEMP": -20.0}
        img = ImageData(data=data, header=header)
        assert img.exposure == 120.0
        assert img.temperature == -20.0
        assert img.width == 200
        assert img.height == 100


# ── UI widgets — tested via import + instantiation only ──────────
# pytest-qt has a PySide6 vs PyQt6 conflict, so we test that the
# widgets can be imported and their non-GUI methods work.

class TestUIImports:
    """Verify all UI modules can be imported without errors."""

    def test_import_histogram(self):
        from astraios.ui.widgets.histogram import HistogramWidget
        assert HistogramWidget is not None

    def test_import_curves_widget(self):
        from astraios.ui.widgets.curves_widget import CurveEditor
        assert CurveEditor is not None

    def test_import_image_canvas(self):
        from astraios.ui.widgets.image_canvas import ImageCanvas
        assert ImageCanvas is not None

    def test_import_log_panel(self):
        from astraios.ui.widgets.log_panel import LogPanel
        assert LogPanel is not None

    def test_import_mask_controls(self):
        from astraios.ui.widgets.mask_controls import MaskSelector
        assert MaskSelector is not None

    def test_import_export_dialog(self):
        from astraios.ui.dialogs.export_dialog import ExportDialog
        assert ExportDialog is not None

    def test_import_preferences_dialog(self):
        from astraios.ui.dialogs.preferences_dialog import PreferencesDialog
        assert PreferencesDialog is not None

    def test_import_main_window(self):
        from astraios.ui.main_window import MainWindow
        assert MainWindow is not None

    def test_import_theme(self):
        from astraios.ui.theme import get_dark_theme, DARK_THEME
        assert isinstance(DARK_THEME, str)
        assert len(DARK_THEME) > 0
        assert "QMainWindow" in DARK_THEME

    def test_import_tools_panel(self):
        from astraios.ui.panels.tools_panel import ToolsPanel
        assert ToolsPanel is not None

    def test_import_project_panel(self):
        from astraios.ui.panels.project_panel import ProjectPanel
        assert ProjectPanel is not None


class TestHistogramData:
    """Test histogram computation (core, no Qt)."""

    def test_compute_histogram(self):
        from astraios.core.stretch import compute_histogram
        data = np.full((3, 100, 100), 0.5, dtype=np.float32)
        hist = compute_histogram(data)
        assert "red" in hist
        assert "green" in hist
        assert "blue" in hist
        assert "luminance" in hist
        assert len(hist["red"]) == 256


class TestCurvesCore:
    """Test curves computation (core, no Qt)."""

    def test_curves_lut_generation(self):
        from astraios.core.curves import CurvePoints
        points = CurvePoints(points=[(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)])
        lut = points.build_lut()
        assert lut.shape[0] > 0
        assert lut[0] == pytest.approx(0.0, abs=0.01)
        assert lut[-1] == pytest.approx(1.0, abs=0.01)


class TestExportDialogParams:
    """Test export dialog parameter logic without Qt."""

    def test_extension_detection(self):
        from astraios.ui.dialogs.export_dialog import FORMAT_FILTERS
        assert ".tif" in FORMAT_FILTERS
        assert ".png" in FORMAT_FILTERS
        assert ".jpg" in FORMAT_FILTERS


class TestPreferencesDefaults:
    """Test preferences default values without Qt."""

    def test_default_values(self):
        from astraios.ui.dialogs.preferences_dialog import DEFAULTS
        assert DEFAULTS["processing/use_gpu"] is True
        assert DEFAULTS["processing/tile_size"] == 1024
        assert DEFAULTS["update/check_on_startup"] is True
