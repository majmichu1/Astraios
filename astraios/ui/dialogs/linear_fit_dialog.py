"""Linear Fit dialog.

Drives astraios.core.linear_fit (map the current image's levels onto a
reference image via a robust slope+intercept fit), ported from Seti Astro
Suite Pro (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
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
    QSpinBox,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


class _Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, reference, params):
        super().__init__()
        self._image, self._reference, self._params = image, reference, params

    def run(self):
        try:
            from astraios.core.linear_fit import linear_fit
            self.done.emit(linear_fit(self._image, self._reference, self._params))
        except Exception as exc:
            log.exception("Linear fit failed")
            self.failed.emit(str(exc))


class LinearFitDialog(QDialog):
    """Match the current image's levels to a reference frame."""

    result_ready = pyqtSignal(object)

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Linear Fit")
        self.setMinimumWidth(440)
        self._image = image
        self._reference: np.ndarray | None = None
        self._worker: _Worker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Rescales the current image so its brightness levels match a "
            "reference frame (a straight-line fit of slope and offset). Use "
            "it to match channel backgrounds before combining, or to bring "
            "one frame onto another's levels."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        ref_row = QHBoxLayout()
        self._ref_edit = QLineEdit()
        self._ref_edit.setPlaceholderText("Load the reference frame...")
        self._ref_edit.setReadOnly(True)
        browse = QPushButton("Load...")
        browse.clicked.connect(self._browse)
        ref_row.addWidget(self._ref_edit, 1)
        ref_row.addWidget(browse)
        lay.addLayout(ref_row)

        form = QFormLayout()
        self._per_channel = QCheckBox("Fit each channel separately")
        self._per_channel.setChecked(True)
        self._per_channel.setToolTip(
            "<qt>Fit R, G, B independently (corrects color casts). Untick "
            "to apply one shared fit to all channels (preserves color, "
            "only rescales brightness).</qt>")
        form.addRow(self._per_channel)
        self._sigma_clip = QCheckBox("Reject outliers (sigma clip)")
        self._sigma_clip.setChecked(True)
        self._sigma_clip.setToolTip(
            "<qt>Ignore stars and hot pixels when fitting so the line "
            "follows the true background relationship.</qt>")
        form.addRow(self._sigma_clip)
        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(1.0, 6.0)
        self._sigma.setSingleStep(0.5)
        self._sigma.setValue(3.0)
        form.addRow(*self._r("Sigma", self._sigma,
                    "How aggressively outliers are rejected. Lower rejects "
                    "more; 3 is standard."))
        self._iters = QSpinBox()
        self._iters.setRange(1, 10)
        self._iters.setValue(5)
        form.addRow(*self._r("Iterations", self._iters,
                    "Rejection passes. More refines the fit against stubborn "
                    "outliers."))
        lay.addLayout(form)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._apply_btn = QPushButton("Apply Linear Fit")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        btns.addWidget(self._apply_btn)
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
            self, "Load Reference Frame", "",
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
        ref = img.data
        if ref.shape[-2:] != self._image.shape[-2:]:
            self._status.setText(
                f"Reference size {ref.shape[-2:]} does not match the image "
                f"{self._image.shape[-2:]}."
            )
            return
        self._reference = ref
        from pathlib import Path
        self._ref_edit.setText(Path(path).name)
        self._apply_btn.setEnabled(True)
        self._status.setText("")

    def _apply(self):
        from astraios.core.linear_fit import LinearFitParams

        if self._reference is None:
            return
        params = LinearFitParams(
            per_channel=self._per_channel.isChecked(),
            sigma_clip=self._sigma_clip.isChecked(),
            sigma=float(self._sigma.value()),
            max_iters=int(self._iters.value()),
        )
        self._apply_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._status.setText("Fitting...")
        self._worker = _Worker(self._image, self._reference, params)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        if not isinstance(result, np.ndarray):
            self._status.setText("No result produced.")
            self._apply_btn.setEnabled(True)
            return
        self.result_ready.emit(result)
        self.accept()

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._apply_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
