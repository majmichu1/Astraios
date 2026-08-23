"""Guided Processing — the beginner's front door to Astraios.

A toolbox this size is intimidating: ten tabs and a hundred sliders give a
newcomer no idea what to press first. This dialog answers that by walking the
correct workflow one step at a time, explaining each step in plain language,
suggesting settings measured from the image, and showing the result before it
is committed.

The sequence, the suggestions and the maths all live in
``astraios.core.guided``; this file is only presentation.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from astraios.core.guided import GuidedStep, looks_linear, workflow_for
from astraios.ui.widgets.ui_kit import (
    ACCENT,
    BG_TERTIARY,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    RunBtn,
    help_dot,
    make_label,
    param_help,
)

log = logging.getLogger(__name__)


class _StepWorker(QThread):
    """Run one step off the GUI thread so a slow tool cannot freeze the app."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, step: GuidedStep, image: np.ndarray, params: dict):
        super().__init__()
        self._step, self._image, self._params = step, image, dict(params)

    def run(self):
        try:
            self.done.emit(self._step.apply(self._image, self._params))
        except Exception as exc:
            log.exception("Guided step %s failed", self._step.step_id)
            self.failed.emit(str(exc))


class GuidedDialog(QDialog):
    """Step-by-step guided processing with a live preview.

    Emits ``preview_ready`` with pixels to show on the canvas (never committed)
    and ``result_ready`` once with the finished image when the user accepts.
    """

    preview_ready = pyqtSignal(object, str)
    result_ready = pyqtSignal(object)

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guided Processing")
        self.setMinimumWidth(560)

        self._original = np.asarray(image, dtype=np.float32)
        self._current = self._original.copy()
        self._steps = workflow_for(self._current)
        self._index = 0
        self._worker: _StepWorker | None = None
        self._pending: np.ndarray | None = None
        # (image before the step, step_id) so Back can rewind exactly.
        self._history: list[tuple[np.ndarray, str]] = []
        self._controls: dict[str, QDoubleSpinBox] = {}

        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        self._progress_lbl = make_label("", TEXT_SECONDARY, 10)
        lay.addWidget(self._progress_lbl)

        self._title = make_label("", TEXT_PRIMARY, 15)
        self._title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 700;"
        )
        lay.addWidget(self._title)

        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet(f"color: {ACCENT}; font-size: 12px;")
        lay.addWidget(self._summary)

        self._detail = QLabel()
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        lay.addWidget(self._detail)

        self._controls_box = QWidget()
        self._controls_box.setStyleSheet(
            f"background: {BG_TERTIARY}; border: 1px solid {BORDER};"
            " border-radius: 5px;"
        )
        self._controls_layout = QVBoxLayout(self._controls_box)
        self._controls_layout.setContentsMargins(10, 10, 10, 10)
        self._controls_layout.setSpacing(6)
        lay.addWidget(self._controls_box)

        self._busy = QProgressBar()
        self._busy.setRange(0, 0)
        self._busy.setVisible(False)
        lay.addWidget(self._busy)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        lay.addWidget(self._status)

        lay.addStretch()

        nav = QHBoxLayout()
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._go_back)
        nav.addWidget(self._back_btn)

        self._skip_btn = QPushButton("Skip this step")
        self._skip_btn.clicked.connect(self._skip)
        nav.addWidget(self._skip_btn)

        nav.addStretch()

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._preview)
        nav.addWidget(self._preview_btn)

        self._apply_btn = RunBtn("Apply and continue")
        self._apply_btn.clicked.connect(self._apply)
        nav.addWidget(self._apply_btn)
        lay.addLayout(nav)

        finish = QHBoxLayout()
        finish.addStretch()
        self._auto_btn = QPushButton("Do the rest for me")
        self._auto_btn.setToolTip(
            "Apply every remaining step with its suggested settings."
        )
        self._auto_btn.clicked.connect(self._run_rest)
        finish.addWidget(self._auto_btn)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        finish.addWidget(cancel)
        lay.addLayout(finish)

        if not looks_linear(self._original):
            self._status.setText(
                "This image looks like it has already been stretched. The "
                "guided steps assume a freshly stacked (linear) image, so "
                "some of them may do little or overcook the result."
            )
        self._show_step()

    # ── step rendering ────────────────────────────────────────────────

    @property
    def _step(self) -> GuidedStep | None:
        if 0 <= self._index < len(self._steps):
            return self._steps[self._index]
        return None

    def _show_step(self):
        step = self._step
        if step is None:
            self._finish()
            return

        self._progress_lbl.setText(
            f"Step {self._index + 1} of {len(self._steps)}"
        )
        self._title.setText(step.title)
        self._summary.setText(step.summary)
        self._detail.setText(step.detail)

        while self._controls_layout.count():
            item = self._controls_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._controls.clear()

        suggested = step.suggest(self._current)
        for control in step.controls:
            row = QHBoxLayout()
            row.addWidget(make_label(control.label, TEXT_PRIMARY, 11))
            row.addStretch()
            spin = QDoubleSpinBox()
            spin.setRange(control.minimum, control.maximum)
            spin.setSingleStep(control.step)
            spin.setDecimals(control.decimals)
            spin.setValue(float(suggested.get(control.key, control.minimum)))
            spin.setFixedWidth(110)
            row.addWidget(spin)
            row.addWidget(
                help_dot(
                    param_help(
                        control.summary,
                        higher=control.higher,
                        lower=control.lower,
                    )
                )
            )
            self._controls[control.key] = spin
            holder = QWidget()
            holder.setLayout(row)
            self._controls_layout.addWidget(holder)

        if not step.controls:
            self._controls_layout.addWidget(
                make_label("Nothing to adjust for this step.", TEXT_SECONDARY, 11)
            )

        self._back_btn.setEnabled(bool(self._history))
        self._skip_btn.setText(
            "Skip this step" if not step.default_skipped else "Skip (recommended)"
        )
        self._pending = None
        if self._status.text().startswith("Preview"):
            self._status.setText("")

    def _params(self) -> dict:
        return {key: spin.value() for key, spin in self._controls.items()}

    # ── actions ───────────────────────────────────────────────────────

    def _set_busy(self, busy: bool):
        self._busy.setVisible(busy)
        for btn in (
            self._apply_btn, self._skip_btn, self._back_btn,
            self._preview_btn, self._auto_btn,
        ):
            btn.setEnabled(not busy)
        if not busy:
            self._back_btn.setEnabled(bool(self._history))

    def _run_step(self, on_done):
        step = self._step
        if step is None:
            return
        self._set_busy(True)
        self._status.setText(f"Working on {step.title.lower()}…")
        self._worker = _StepWorker(step, self._current, self._params())
        self._worker.done.connect(on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _preview(self):
        def _done(result):
            self._set_busy(False)
            self._pending = result
            self._status.setText(
                "Preview shown on the canvas. Apply to keep it, or adjust and "
                "preview again."
            )
            self.preview_ready.emit(result, f"Preview: {self._step.title}")

        self._run_step(_done)

    def _apply(self):
        def _done(result):
            self._set_busy(False)
            step = self._step
            self._history.append((self._current, step.step_id))
            self._current = result
            self._index += 1
            self._status.setText("")
            self.preview_ready.emit(self._current, f"After: {step.title}")
            self._show_step()

        self._run_step(_done)

    def _on_failed(self, message: str):
        self._set_busy(False)
        self._status.setText(
            f"That step could not run ({message}). You can skip it and carry on."
        )

    def _skip(self):
        step = self._step
        if step is None:
            return
        self._history.append((self._current, step.step_id))
        self._index += 1
        self._status.setText("")
        self._show_step()

    def _go_back(self):
        if not self._history:
            return
        image, _step_id = self._history.pop()
        self._current = image
        self._index = max(0, self._index - 1)
        self.preview_ready.emit(self._current, "Stepped back")
        self._show_step()

    def _run_rest(self):
        """Apply every remaining step with its suggested settings."""
        from astraios.core.guided import run_workflow

        remaining = {s.step_id for s in self._steps[self._index:]}
        skip = {
            s.step_id for s in self._steps
            if s.step_id in remaining and s.default_skipped
        }
        self._set_busy(True)
        self._status.setText("Applying the remaining steps…")
        try:
            # Only the steps from here on: run_workflow re-derives the list,
            # so skip the ones already done.
            done = {s.step_id for s in self._steps[: self._index]}
            run = run_workflow(self._current, skip=skip | done)
            self._current = run.image
        except Exception as exc:
            self._set_busy(False)
            self._status.setText(f"Could not finish automatically: {exc}")
            return
        self._set_busy(False)
        self._index = len(self._steps)
        self._finish()

    def _finish(self):
        self.result_ready.emit(self._current)
        self.accept()
