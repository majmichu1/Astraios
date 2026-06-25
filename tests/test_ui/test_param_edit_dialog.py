"""Tests for the generic step parameter editor."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from astraios.core.denoise import DenoiseMethod  # noqa: E402
from astraios.ui.dialogs.param_edit_dialog import ParamEditDialog  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_preserves_types_and_enum(app):
    params = {
        "method": DenoiseMethod.WAVELET,   # enum
        "strength": 0.5,                   # float
        "wavelet_levels": 4,               # int
        "linked": True,                    # bool
        "wavelet": "db4",                  # str
    }
    dlg = ParamEditDialog(None, "Denoise", params)
    out = dlg.get_params()
    # Unchanged read-back keeps every value and type.
    assert out["method"] is DenoiseMethod.WAVELET
    assert isinstance(out["strength"], float) and out["strength"] == 0.5
    assert isinstance(out["wavelet_levels"], int) and out["wavelet_levels"] == 4
    assert out["linked"] is True
    assert out["wavelet"] == "db4"


def test_edits_apply(app):
    params = {"strength": 0.5, "linked": True, "method": DenoiseMethod.WAVELET}
    dlg = ParamEditDialog(None, "Denoise", params)
    dlg._editors["strength"].setValue(0.9)
    dlg._editors["linked"].setChecked(False)
    # Switch the enum combo to a different member.
    combo = dlg._editors["method"]
    nlm_idx = combo.findData(DenoiseMethod.NLM)
    combo.setCurrentIndex(nlm_idx)
    out = dlg.get_params()
    assert out["strength"] == pytest.approx(0.9)
    assert out["linked"] is False
    assert out["method"] is DenoiseMethod.NLM


def test_number_list_round_trip(app):
    params = {"scale_weights": [1.5, 1.2, 1.0, 1.0]}
    dlg = ParamEditDialog(None, "Wavelet", params)
    out = dlg.get_params()
    assert out["scale_weights"] == [1.5, 1.2, 1.0, 1.0]
    dlg._editors["scale_weights"].setText("2.0, 1.0, 0.5")
    assert dlg.get_params()["scale_weights"] == [2.0, 1.0, 0.5]


def test_empty_params(app):
    dlg = ParamEditDialog(None, "Invert", {})
    assert dlg.get_params() == {}
