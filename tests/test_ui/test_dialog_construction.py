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

import cosmica.ui.dialogs as dialogs_pkg

_IMG = np.clip(np.random.default_rng(0).random((3, 32, 32)) * 0.4 + 0.05, 0, 1).astype(np.float32)
_CANDIDATE_KWARGS = [
    {}, {"parent": None}, {"image_data": _IMG}, {"current_image": _IMG},
    {"image": _IMG}, {"base_image": _IMG}, {"frame_paths": []}, {"paths": []},
]


def _all_dialog_classes():
    out = []
    for mod in pkgutil.iter_modules(dialogs_pkg.__path__):
        m = importlib.import_module(f"cosmica.ui.dialogs.{mod.name}")
        for name, obj in inspect.getmembers(m, inspect.isclass):
            if name.endswith("Dialog") and obj.__module__ == m.__name__:
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
    from cosmica.ui.dialogs.channel_combine_dialog import ChannelCombineDialog

    dlg = ChannelCombineDialog(current_image=_IMG)
    assert len(dlg._channel_rows) >= 1
