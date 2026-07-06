"""Transient Hunter dialog.

Drives astraios.core.transient_hunter (find sources that are new, moved, or
vanished between a reference image and a later frame of the same field —
supernova/nova and asteroid/comet candidates), ported from Seti Astro Suite
Pro (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


class _NumericItem(QTableWidgetItem):
    """A table item that sorts by its underlying numeric value, not its text."""

    def __init__(self, value: float, text: str):
        super().__init__(text)
        self._value = float(value)

    def __lt__(self, other):
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)


class _CandidateOverlay(QWidget):
    """Small painted overlay of candidate (x, y) positions over the field."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._w = 1
        self._h = 1
        self._points: list[tuple[float, float, Qt.GlobalColor]] = []

    def set_field(self, width: int, height: int) -> None:
        self._w = max(1, int(width))
        self._h = max(1, int(height))
        self.update()

    def set_candidates(self, candidates) -> None:
        from astraios.core.transient_hunter import TransientKind

        colors = {
            TransientKind.NEW: Qt.GlobalColor.red,
            TransientKind.MOVED: Qt.GlobalColor.yellow,
            TransientKind.VANISHED: Qt.GlobalColor.cyan,
        }
        self._points = [
            (c.x, c.y, colors.get(c.kind, Qt.GlobalColor.white)) for c in candidates
        ]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, Qt.GlobalColor.black)
        painter.setPen(QPen(Qt.GlobalColor.darkGray, 1))
        painter.drawRect(0, 0, w - 1, h - 1)

        if not self._points:
            painter.setPen(QPen(Qt.GlobalColor.gray))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No candidates")
            return

        sx = w / self._w
        sy = h / self._h
        for x, y, color in self._points:
            painter.setPen(QPen(color, 1))
            painter.setBrush(color)
            px, py = x * sx, y * sy
            painter.drawEllipse(int(px) - 4, int(py) - 4, 8, 8)


class _Worker(QThread):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, reference, new_image, params):
        super().__init__()
        self._reference = reference
        self._new_image = new_image
        self._params = params

    def run(self):
        try:
            from astraios.core.transient_hunter import hunt_transients
            result = hunt_transients(
                self._reference, self._new_image, self._params,
                already_aligned=False,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.done.emit(result)
        except Exception as exc:
            log.exception("Transient hunt failed")
            self.failed.emit(str(exc))


class TransientHunterDialog(QDialog):
    """Find new/moved/vanished sources between a reference and a later frame."""

    preview_requested = pyqtSignal(object)

    def __init__(self, reference_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transient Hunter (SN / Asteroid)")
        self.setMinimumSize(640, 640)
        self._reference = reference_image
        self._new_image: np.ndarray | None = None
        self._worker: _Worker | None = None
        self._result = None  # exposed for scripted verification

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Compares the current image (used as the reference) with a "
            "later frame of the same field to find sources that are new "
            "(supernova/nova candidate), moved (asteroid/comet candidate), "
            "or vanished."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        new_row = QHBoxLayout()
        self._new_edit = QLineEdit()
        self._new_edit.setPlaceholderText("Load the new / later frame...")
        self._new_edit.setReadOnly(True)
        browse = QPushButton("Load...")
        browse.clicked.connect(self._browse)
        new_row.addWidget(self._new_edit, 1)
        new_row.addWidget(browse)
        lay.addLayout(new_row)

        form = QFormLayout()
        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(1.0, 30.0)
        self._sigma.setValue(5.0)
        form.addRow(*self._r("Detection sigma", self._sigma,
                    "Residual-detection threshold, in noise sigma, applied "
                    "to the new-minus-reference difference image."))
        self._match_radius = QDoubleSpinBox()
        self._match_radius.setRange(1.0, 500.0)
        self._match_radius.setValue(50.0)
        form.addRow(*self._r("Match radius", self._match_radius,
                    "Max pixel distance used to pair a newly-appeared "
                    "residual with a vanished one, classifying it as a "
                    "moving (asteroid/comet) candidate rather than two "
                    "separate new/vanished candidates."))
        self._edge_margin = QDoubleSpinBox()
        self._edge_margin.setRange(0.0, 0.4)
        self._edge_margin.setSingleStep(0.01)
        self._edge_margin.setValue(0.05)
        form.addRow(*self._r("Edge margin", self._edge_margin,
                    "Fraction of the frame border excluded from candidate "
                    "detection, to avoid registration-edge artifacts."))
        self._normalize = QCheckBox("Normalize background/contrast")
        self._normalize.setChecked(True)
        self._normalize.setToolTip(
            "<qt>Robustly rescale the new frame's background level/"
            "contrast to match the reference before differencing, so a "
            "plain brightness offset doesn't create false residuals.</qt>")
        form.addRow(self._normalize)
        self._register = QCheckBox("Register (align) new frame")
        self._register.setChecked(True)
        self._register.setToolTip(
            "<qt>Align the new frame onto the reference frame's pixel grid "
            "before differencing. Leave on unless the frame is already "
            "pixel-aligned.</qt>")
        form.addRow(self._register)
        lay.addLayout(form)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        self._overlay = _CandidateOverlay()
        if reference_image is not None:
            h, w = reference_image.shape[-2], reference_image.shape[-1]
            self._overlay.set_field(w, h)
        lay.addWidget(self._overlay)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["Kind", "X", "Y", "Flux", "dx", "dy"])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        lay.addWidget(self._table, 1)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Hunt Transients")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        btns.addWidget(self._run_btn)
        self._diff_btn = QPushButton("Show Difference")
        self._diff_btn.setEnabled(False)
        self._diff_btn.clicked.connect(self._show_difference)
        btns.addWidget(self._diff_btn)
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    @staticmethod
    def _r(label, widget, tip):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return label, row

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load New Frame", "",
            "Images (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png)"
        )
        if not path:
            return
        from astraios.core.image_io import load_image
        try:
            img = load_image(path)
        except Exception as exc:
            self._status.setText(f"Could not load: {exc}")
            return
        new = img.data
        if self._reference is not None and new.shape[-2:] != self._reference.shape[-2:]:
            self._status.setText(
                f"New frame size {new.shape[-2:]} does not match the "
                f"reference {self._reference.shape[-2:]}."
            )
            return
        self._new_image = new
        from pathlib import Path
        self._new_edit.setText(Path(path).name)
        self._run_btn.setEnabled(True)
        self._status.setText("")

    def _run(self):
        if self._reference is None or self._new_image is None:
            return
        from astraios.core.transient_hunter import TransientHunterParams

        params = TransientHunterParams(
            detection_sigma=float(self._sigma.value()),
            match_radius=float(self._match_radius.value()),
            edge_margin_fraction=float(self._edge_margin.value()),
            normalize=self._normalize.isChecked(),
            register=self._register.isChecked(),
        )
        self._run_btn.setEnabled(False)
        self._diff_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._status.setText("Hunting...")
        self._worker = _Worker(self._reference, self._new_image, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_progress(self, fraction: float, message: str):
        self._progress.setValue(int(fraction * 100))
        self._status.setText(message)

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._result = result
        self._overlay.set_candidates(result.candidates)

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(result.candidates))
        for row, c in enumerate(result.candidates):
            self._table.setItem(row, 0, QTableWidgetItem(c.kind.name))
            self._table.setItem(row, 1, _NumericItem(c.x, f"{c.x:.1f}"))
            self._table.setItem(row, 2, _NumericItem(c.y, f"{c.y:.1f}"))
            self._table.setItem(row, 3, _NumericItem(c.flux, f"{c.flux:.4g}"))
            dx_text = f"{c.dx:.1f}" if c.dx is not None else "-"
            dy_text = f"{c.dy:.1f}" if c.dy is not None else "-"
            self._table.setItem(row, 4, _NumericItem(c.dx if c.dx is not None else 0.0, dx_text))
            self._table.setItem(row, 5, _NumericItem(c.dy if c.dy is not None else 0.0, dy_text))
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()

        self._diff_btn.setEnabled(bool(result.diff_images))
        self._status.setText(f"{len(result.candidates)} candidate(s) found.")

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")

    def _show_difference(self):
        if self._result is None or not self._result.diff_images:
            return
        self.preview_requested.emit(self._result.diff_images[0])
