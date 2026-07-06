"""Add Stars dialog.

Drives astraios.core.add_stars (recombine a stars-only layer back onto a
starless image -- the inverse of a star-removal step like StarNet), ported
from Seti Astro Suite Pro (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

# (label, AddStarsBlendMode value) -- exactly the two modes SASpro's
# "Blend Type" combo offers; see astraios.core.add_stars module docstring.
_MODE_LABELS = [
    ("Screen", "screen"),
    ("Add", "add"),
]


class _Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, starless, stars, params):
        super().__init__()
        self._starless, self._stars, self._params = starless, stars, params

    def run(self):
        try:
            from astraios.core.add_stars import add_stars
            self.done.emit(add_stars(self._starless, self._stars, self._params))
        except Exception as exc:
            log.exception("Add Stars failed")
            self.failed.emit(str(exc))


class AddStarsDialog(QDialog):
    """Recombine a stars-only layer onto the current (starless) image."""

    result_ready = pyqtSignal(object)

    def __init__(self, starless: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Stars")
        self.setMinimumWidth(440)
        self._starless = starless
        self._stars: np.ndarray | None = None
        self._worker: _Worker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Recombines a stars-only layer back onto the current starless "
            "image -- the inverse of a star-removal step. The current "
            "image is treated as the starless base."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        stars_row = QHBoxLayout()
        self._stars_edit = QLineEdit()
        self._stars_edit.setPlaceholderText("Load the stars-only image...")
        self._stars_edit.setReadOnly(True)
        browse = QPushButton("Load...")
        browse.clicked.connect(self._browse)
        stars_row.addWidget(self._stars_edit, 1)
        stars_row.addWidget(browse)
        stars_row.addWidget(help_dot(param_help(
            "The stars-only layer to add back onto the starless image "
            "(e.g. the star mask/layer saved off by a star-removal tool "
            "such as StarNet).",
            how="Must have the same pixel dimensions as the current image. "
                "A mono stars layer broadcasts onto a color starless image "
                "and vice versa.",
            tip="Loading a new file replaces the previous stars layer.",
        )))
        lay.addLayout(stars_row)

        form = QFormLayout()
        self._mode = QComboBox()
        self._mode.addItems([label for label, _ in _MODE_LABELS])
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._mode)
        mode_row.addWidget(help_dot(param_help(
            "How the stars layer combines with the starless image.",
            how="<b>Screen</b> = 1-(1-starless)(1-stars): stars brighten the "
                "image but the result can never clip or exceed white, so "
                "bright stars stay soft-edged. <b>Add</b> = starless + "
                "stars: a straight linear sum that can blow out to pure "
                "white where both layers are already bright.",
            higher=None, lower=None,
            default="Screen matches most starless/stars-only workflows; "
                    "use Add only if Screen looks too dim.",
        )))
        mode_row.addStretch()
        form.addRow("Blend Type", mode_row)

        self._amount = QSlider(Qt.Orientation.Horizontal)
        self._amount.setRange(0, 100)
        self._amount.setValue(100)
        self._amount.setTickInterval(10)
        self._amount.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._amount_val = QLabel("100%")
        self._amount_val.setFixedWidth(40)
        self._amount.valueChanged.connect(
            lambda v: self._amount_val.setText(f"{v}%")
        )
        amount_row = QHBoxLayout()
        amount_row.addWidget(self._amount, 1)
        amount_row.addWidget(self._amount_val)
        amount_row.addWidget(help_dot(param_help(
            "Blend Ratio -- how much of the stars layer to bring back in.",
            how="Cross-fades between the untouched starless image and the "
                "fully-blended result: "
                "<i>(1-amount)&times;starless + amount&times;blend</i>.",
            higher="Closer to 100% = the full Screen/Add blend -- stars "
                   "come back at their original intensity.",
            lower="Closer to 0% = fades back toward the plain starless "
                  "image -- 0% leaves it completely unchanged.",
            default="100% (SASpro's default) applies the full blend.",
        )))
        form.addRow("Blend Ratio", amount_row)
        lay.addLayout(form)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._apply_btn = QPushButton("Add Stars")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        btns.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Stars-Only Image", "",
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
        stars = img.data
        if stars.shape[-2:] != self._starless.shape[-2:]:
            self._status.setText(
                f"Stars-only image size {stars.shape[-2:]} does not match "
                f"the current image {self._starless.shape[-2:]}."
            )
            return
        self._stars = stars
        from pathlib import Path
        self._stars_edit.setText(Path(path).name)
        self._apply_btn.setEnabled(True)
        self._status.setText("")

    def _selected_mode(self):
        from astraios.core.add_stars import AddStarsBlendMode
        return AddStarsBlendMode(_MODE_LABELS[self._mode.currentIndex()][1])

    def _apply(self):
        from astraios.core.add_stars import AddStarsParams

        if self._stars is None:
            return
        params = AddStarsParams(
            blend_mode=self._selected_mode(),
            amount=self._amount.value() / 100.0,
        )
        self._apply_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        self._status.setText("Adding stars...")
        self._worker = _Worker(self._starless, self._stars, params)
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
