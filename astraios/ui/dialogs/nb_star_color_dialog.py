"""NB Star Color dialog — recombine a narrowband composite with natural
broadband star color.

The recombination core is ported from Seti Astro Suite Pro (GPL-3.0,
Franklin Marek); this dialog drives astraios.core.nb_star_color.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


class _RecombineWorker(QThread):
    """Runs recombine_star_color off the GUI thread (large images)."""

    finished_ok = pyqtSignal(object)  # ndarray
    failed = pyqtSignal(str)

    def __init__(self, nb_image, rgb_stars, params):
        super().__init__()
        self._nb_image = nb_image
        self._rgb_stars = rgb_stars
        self._params = params

    def run(self):
        try:
            from astraios.core.nb_star_color import recombine_star_color

            result = recombine_star_color(self._nb_image, self._rgb_stars, self._params)
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("NB star color recombination failed")
            self.failed.emit(str(exc))


class NBStarColorDialog(QDialog):
    """Borrow natural star color from a broadband frame for a narrowband image."""

    # Emitted with the recombined (3, H, W) RGB ndarray when Apply succeeds.
    result_ready = pyqtSignal(object)

    def __init__(self, nb_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NB Star Color")
        self.setMinimumWidth(460)
        self._nb_image = nb_image
        self._rgb_stars: np.ndarray | None = None
        self._worker: _RecombineWorker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Mapping Ha/OIII/SII straight onto R/G/B gives stars odd, "
            "bloated colors. This borrows natural star color from a "
            "broadband (RGB/OSC) frame -- typically a stars-only extraction "
            "-- and blends it into the narrowband composite."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        src = QGroupBox("Broadband star frame")
        srow = QHBoxLayout(src)
        self._path_label = QLabel("No file loaded")
        self._path_label.setWordWrap(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        srow.addWidget(self._path_label, 1)
        srow.addWidget(browse_btn)
        lay.addWidget(src)

        form_group = QGroupBox("Recombination")
        form = QFormLayout(form_group)

        self._ratio_spin = QDoubleSpinBox()
        self._ratio_spin.setRange(0.0, 1.0)
        self._ratio_spin.setSingleStep(0.05)
        self._ratio_spin.setValue(0.30)
        form.addRow("Ha/green ratio", self._row(
            self._ratio_spin,
            "Ha:broadband-green blend weight for the output green channel. "
            "1.0 uses pure Ha; 0.0 uses pure broadband green.",
        ))

        self._stretch_check = QCheckBox("Enable star stretch")
        self._stretch_check.setChecked(True)
        self._stretch_check.toggled.connect(self._update_enabled_state)
        form.addRow(self._row(
            self._stretch_check,
            "Applies a non-linear \"star stretch\" boost after combining, "
            "to bring up faint star color.",
        ))

        self._stretch_factor_spin = QDoubleSpinBox()
        self._stretch_factor_spin.setRange(0.1, 20.0)
        self._stretch_factor_spin.setSingleStep(0.5)
        self._stretch_factor_spin.setValue(5.0)
        form.addRow("Stretch factor", self._row(
            self._stretch_factor_spin,
            "Exponent in the star-stretch formula. Higher values push "
            "faint signal (and star wings) up harder.",
        ))

        self._saturation_spin = QDoubleSpinBox()
        self._saturation_spin.setRange(0.0, 5.0)
        self._saturation_spin.setSingleStep(0.05)
        self._saturation_spin.setValue(1.0)
        form.addRow("Saturation", self._row(
            self._saturation_spin,
            "HSV saturation multiplier applied after combining. 1.0 "
            "leaves color unchanged.",
        ))

        self._scnr_check = QCheckBox("Apply green SCNR")
        self._scnr_check.setChecked(True)
        self._scnr_check.toggled.connect(self._update_enabled_state)
        form.addRow(self._row(
            self._scnr_check,
            "Removes the residual narrowband green cast (average-neutral "
            "SCNR) after combining.",
        ))

        self._scnr_amount_spin = QDoubleSpinBox()
        self._scnr_amount_spin.setRange(0.0, 1.0)
        self._scnr_amount_spin.setSingleStep(0.05)
        self._scnr_amount_spin.setValue(1.0)
        form.addRow("SCNR amount", self._row(
            self._scnr_amount_spin,
            "SCNR strength. 1.0 reproduces g = min(g, (r+b)/2) exactly.",
        ))

        self._preserve_lum_check = QCheckBox("Preserve luminance in SCNR")
        form.addRow(self._row(
            self._preserve_lum_check,
            "Rescales R/G/B during SCNR to preserve perceived luminance.",
        ))

        lay.addWidget(form_group)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        btns.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self._update_enabled_state()

    @staticmethod
    def _row(widget, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return row

    def _update_enabled_state(self):
        self._stretch_factor_spin.setEnabled(self._stretch_check.isChecked())
        scnr_on = self._scnr_check.isChecked()
        self._scnr_amount_spin.setEnabled(scnr_on)
        self._preserve_lum_check.setEnabled(scnr_on)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Broadband/RGB Star Frame", "",
            "Images (*.fits *.fit *.fts *.tif *.tiff *.png *.jpg *.xisf);;All files (*)",
        )
        if not path:
            return
        from astraios.core.image_io import load_image

        try:
            img = load_image(path)
        except Exception as exc:
            self._status.setText(f"Could not load image: {exc}")
            self._rgb_stars = None
            self._apply_btn.setEnabled(False)
            return

        nb_hw = self._nb_image.shape[-2:]
        rgb_hw = img.data.shape[-2:]
        if nb_hw != rgb_hw:
            self._status.setText(
                f"Size mismatch: the narrowband image is {nb_hw[1]}x{nb_hw[0]}, "
                f"but the selected frame is {rgb_hw[1]}x{rgb_hw[0]}. They must "
                "have the same width and height."
            )
            self._rgb_stars = None
            self._apply_btn.setEnabled(False)
            return

        self._rgb_stars = img.data.astype(np.float32)
        self._path_label.setText(Path(path).name)
        self._status.setText("")
        self._apply_btn.setEnabled(True)

    def get_params(self):
        from astraios.core.nb_star_color import NBStarColorParams

        return NBStarColorParams(
            ratio=float(self._ratio_spin.value()),
            enable_star_stretch=self._stretch_check.isChecked(),
            stretch_factor=float(self._stretch_factor_spin.value()),
            saturation=float(self._saturation_spin.value()),
            apply_scnr=self._scnr_check.isChecked(),
            scnr_amount=float(self._scnr_amount_spin.value()),
            preserve_luminance=self._preserve_lum_check.isChecked(),
        )

    def _apply(self):
        if self._rgb_stars is None:
            self._status.setText("Load a broadband/RGB star frame first.")
            return
        params = self.get_params()
        self._apply_btn.setEnabled(False)
        self._status.setText("Recombining...")

        self._worker = _RecombineWorker(self._nb_image, self._rgb_stars, params)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self.result_ready.emit(result)
        self._apply_btn.setEnabled(True)
        self._status.setText("Done.")

    def _on_fail(self, msg: str):
        self._apply_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
