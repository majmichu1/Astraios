"""Regression: dialogs must construct without NameError/AttributeError.

ChannelCombineDialog referenced an undefined 'options_group' in __init__, so
opening it crashed. It wasn't covered by the menu-handler smoke (it's opened via
a different path), hence this direct construction guard over all dialog classes.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import numpy as np
import pytest

import astraios.ui.dialogs as dialogs_pkg

_IMG = np.clip(np.random.default_rng(0).random((3, 32, 32)) * 0.4 + 0.05, 0, 1).astype(np.float32)

# Covered by TestSpecificConstructorDialogs below with their real
# constructor arguments — excluded from the generic guess-the-kwargs pass
# so it cannot skip them into invisibility.
_SPECIFICALLY_TESTED = {
    "AddStarsDialog", "ContinuumSubtractDialog", "FITSHeaderDialog",
    "ImageCombineDialog", "LuminanceRecombineDialog", "NBStarColorDialog",
    "ParamEditDialog", "ProcessingGraphDialog", "StatisticsDialog",
    "TransientHunterDialog",
}
_CANDIDATE_KWARGS = [
    {}, {"parent": None}, {"image_data": _IMG}, {"current_image": _IMG},
    {"image": _IMG}, {"base_image": _IMG}, {"frame_paths": []}, {"paths": []},
]


def _all_dialog_classes():
    out = []
    for mod in pkgutil.iter_modules(dialogs_pkg.__path__):
        m = importlib.import_module(f"astraios.ui.dialogs.{mod.name}")
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if (
                name.endswith("Dialog")
                and obj.__module__ == m.__name__
                and name not in _SPECIFICALLY_TESTED
            ):
                out.append(obj)
    return out


@pytest.mark.parametrize("dialog_cls", _all_dialog_classes(), ids=lambda c: c.__name__)
def test_dialog_constructs(qtbot, dialog_cls):
    """Constructing with the right args must not raise a NameError/AttributeError
    (a code bug). Dialogs whose constructors need specific args we can't guess
    only raise TypeError; those are skipped, not failed."""
    for kw in _CANDIDATE_KWARGS:
        try:
            dialog_cls(**kw)
            return  # constructed with some arg set
        except TypeError:
            continue  # wrong args — try the next candidate
        except Exception as e:  # NameError / AttributeError = real construction bug
            pytest.fail(f"{dialog_cls.__name__} construction crashed: {type(e).__name__}: {e}")
    pytest.skip(f"{dialog_cls.__name__} needs specific constructor args (no crash)")


def test_channel_combine_dialog_constructs(qtbot):
    from astraios.ui.dialogs.channel_combine_dialog import ChannelCombineDialog

    dlg = ChannelCombineDialog(current_image=_IMG)
    assert len(dlg._channel_rows) >= 1


class TestSpecificConstructorDialogs:
    """The ten dialogs whose constructors need particular arguments.

    Constructed with realistic values so the skip in the parametrized
    test above can never hide a construction crash in them.
    """

    def test_add_stars(self, qtbot):
        from astraios.ui.dialogs.add_stars_dialog import AddStarsDialog
        AddStarsDialog(starless=_IMG)

    def test_continuum_subtract(self, qtbot):
        from astraios.ui.dialogs.continuum_subtract_dialog import ContinuumSubtractDialog
        ContinuumSubtractDialog(narrowband=_IMG)

    def test_fits_header(self, qtbot):
        from astraios.ui.dialogs.fits_header_dialog import FITSHeaderDialog
        FITSHeaderDialog(header={"OBJECT": "M42", "EXPTIME": 300.0})

    def test_image_combine(self, qtbot):
        from astraios.ui.dialogs.image_combine_dialog import ImageCombineDialog
        ImageCombineDialog(image_a=_IMG)

    def test_luminance_recombine(self, qtbot):
        from astraios.ui.dialogs.luminance_recombine_dialog import LuminanceRecombineDialog
        LuminanceRecombineDialog(color_image=_IMG)

    def test_nb_star_color(self, qtbot):
        from astraios.ui.dialogs.nb_star_color_dialog import NBStarColorDialog
        NBStarColorDialog(nb_image=_IMG)

    def test_param_edit(self, qtbot):
        from astraios.ui.dialogs.param_edit_dialog import ParamEditDialog
        ParamEditDialog(
            None, "Test Tool",
            {"sigma": 2.5, "iterations": 3, "name": "preset", "enabled": True},
        )

    def test_processing_graph(self, qtbot):
        from astraios.core.processing_graph import ProcessingGraph
        from astraios.ui.dialogs.processing_graph_dialog import ProcessingGraphDialog
        ProcessingGraphDialog(None, ProcessingGraph())

    def test_statistics(self, qtbot):
        from astraios.core.statistics import compute_image_statistics
        from astraios.ui.dialogs.statistics_dialog import StatisticsDialog
        StatisticsDialog(stats=compute_image_statistics(_IMG))

    def test_transient_hunter(self, qtbot):
        from astraios.ui.dialogs.transient_hunter_dialog import TransientHunterDialog
        TransientHunterDialog(reference_image=_IMG)
