"""Regression: BlinkDialog was constructed as BlinkDialog(self, frame_paths=...)
which bound the parent to frame_paths ('multiple values'), and it crashed on
frame_paths=None ('NoneType not iterable')."""

import pytest


@pytest.fixture
def _app(qtbot):
    return None


def test_blink_dialog_constructs_with_none_and_empty(qtbot):
    from astraios.ui.dialogs.blink_dialog import BlinkDialog

    BlinkDialog()                       # no args
    BlinkDialog(frame_paths=None)       # explicit None
    BlinkDialog(frame_paths=[])         # empty list
    dlg = BlinkDialog(frame_paths=["/tmp/x.fits"], parent=None)
    assert dlg._frame_paths == ["/tmp/x.fits"]
