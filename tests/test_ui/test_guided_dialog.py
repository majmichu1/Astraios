"""Tests for the Guided Processing wizard.

Covers the behaviour a beginner relies on: previews must not commit anything,
Back must rewind exactly, and skipping must not lose the image.
"""

import numpy as np
import pytest

from astraios.ui.dialogs.guided_dialog import GuidedDialog


def _linear_image(color=True):
    rng = np.random.default_rng(0)
    h = w = 96
    yy, xx = np.mgrid[0:h, 0:w]
    neb = 0.012 * np.exp(-(((xx - 48) ** 2 + (yy - 46) ** 2) / (2 * 20.0**2)))
    if color:
        img = np.full((3, h, w), 0.004, np.float32)
        for c, amp in enumerate((1.0, 0.6, 0.55)):
            img[c] += (neb * amp).astype(np.float32)
    else:
        img = (np.full((h, w), 0.004, np.float32) + neb).astype(np.float32)
    for _ in range(10):
        x, y = rng.integers(10, h - 10, 2)
        img = img + (0.2 * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.8**2)))).astype(
            np.float32
        )
    return np.clip(img, 0, 1).astype(np.float32)


@pytest.fixture
def dialog(qtbot):
    # qtbot is requested for the QApplication it provides; this codebase does
    # not register dialogs with addWidget (see test_blink_dialog).
    return GuidedDialog(_linear_image())


class TestConstruction:
    def test_builds_with_the_full_workflow(self, dialog):
        ids = [s.step_id for s in dialog._steps]
        assert ids[0] == "trim"
        assert "stretch" in ids and "gradient" in ids

    def test_mono_image_drops_the_colour_steps(self, qtbot):
        dlg = GuidedDialog(_linear_image(color=False))
        ids = [s.step_id for s in dlg._steps]
        assert "color" not in ids and "saturation" not in ids

    def test_back_is_disabled_on_the_first_step(self, dialog):
        assert not dialog._back_btn.isEnabled()

    def test_controls_are_prefilled_with_suggestions(self, dialog):
        # step 1 (trim) has a control, and it must start at a real value
        assert dialog._controls
        for spin in dialog._controls.values():
            assert spin.minimum() <= spin.value() <= spin.maximum()


class TestNavigation:
    def test_skip_advances_without_changing_the_image(self, dialog):
        before = dialog._current.copy()
        dialog._skip()
        assert dialog._index == 1
        assert np.array_equal(dialog._current, before)

    def test_back_restores_the_previous_image_exactly(self, dialog):
        before = dialog._current.copy()
        dialog._skip()
        dialog._go_back()
        assert dialog._index == 0
        assert np.array_equal(dialog._current, before)

    def test_back_becomes_available_after_advancing(self, dialog):
        dialog._skip()
        assert dialog._back_btn.isEnabled()

    def test_skipping_every_step_still_yields_a_result(self, dialog):
        results = []
        dialog.result_ready.connect(results.append)
        for _ in range(len(dialog._steps)):
            dialog._skip()
        assert results, "finishing must emit a result even if all steps skipped"
        assert np.array_equal(results[0], dialog._original)


class TestAlreadyStretchedWarning:
    def test_warns_when_the_image_is_not_linear(self, qtbot):
        stretched = np.full((3, 64, 64), 0.35, np.float32)
        dlg = GuidedDialog(stretched)
        assert "already been stretched" in dlg._status.text()

    def test_no_warning_for_a_linear_image(self, dialog):
        assert "already been stretched" not in dialog._status.text()


class TestFinishStep:
    """The wizard used to just vanish when the last step was done, leaving a
    beginner with a finished image and no idea what to do next."""

    def _finish(self, dialog):
        for _ in range(len(dialog._steps)):
            dialog._skip()
        return dialog

    def test_finishing_shows_a_summary_instead_of_closing(self, dialog):
        self._finish(dialog)
        assert "ready" in dialog._title.text().lower()

    def test_finish_offers_save(self, dialog):
        self._finish(dialog)
        assert dialog._save_btn is not None
        assert not dialog._save_btn.isHidden()

    def test_save_emits_the_request(self, dialog):
        self._finish(dialog)
        asked = []
        dialog.save_requested.connect(lambda: asked.append(True))
        dialog._save_and_close()
        assert asked == [True]

    def test_step_buttons_are_hidden_on_the_finish_screen(self, dialog):
        self._finish(dialog)
        assert dialog._apply_btn.isHidden()
        assert dialog._skip_btn.isHidden()

    def test_back_from_finish_restores_a_usable_step(self, dialog):
        """Regression: the finish screen hides the step buttons, so stepping
        back showed a step with no way to apply or skip it."""
        self._finish(dialog)
        dialog._go_back()
        assert dialog._index == len(dialog._steps) - 1
        assert not dialog._apply_btn.isHidden()
        assert not dialog._skip_btn.isHidden()
        assert not dialog._controls_box.isHidden()

    def test_summary_names_the_applied_steps(self, dialog):
        dialog._applied = ["Remove the gradient", "The big stretch"]
        dialog._index = len(dialog._steps)
        dialog._finish()
        text = dialog._summary.text().lower()
        assert "gradient" in text and "stretch" in text

    def test_auto_finish_reports_what_it_actually_did(self, dialog):
        """Regression: 'Do the rest for me' bypassed the applied-steps list,
        so the finish screen claimed nothing had been applied right after it
        had processed the entire image."""
        dialog._skip()          # past trim
        dialog._run_rest()
        summary = dialog._summary.text().lower()
        assert "no steps were applied" not in summary
        assert "stretch" in summary
