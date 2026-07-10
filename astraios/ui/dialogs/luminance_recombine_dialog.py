"""Luminance Recombine dialog — replace a color image's luminance with a
separately-processed L frame (LRGB / narrowband finishing step).

The compositing core is ported from Seti Astro Suite Pro (GPL-3.0, Franklin
Marek); this dialog drives astraios.core.luminance_recombine.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
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

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)


class _RecombineWorker(QThread):
    """Runs recombine_luminance off the GUI thread (large images)."""

    finished_ok = pyqtSignal(object)  # ndarray
    failed = pyqtSignal(str)

    def __init__(self, color, luma, params):
        super().__init__()
        self._color = color
        self._luma = luma
        self._params = params

    def run(self):
        try:
            from astraios.core.luminance_recombine import recombine_luminance

            result = recombine_luminance(self._color, self._luma, self._params)
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("Luminance recombine failed")
            self.failed.emit(str(exc))


class LuminanceRecombineDialog(QDialog):
    """Replace the current color image's luminance with a separate L frame."""

    # Emitted with the recombined (3, H, W) RGB ndarray when Apply succeeds.
    result_ready = pyqtSignal(object)

    def __init__(self, color_image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Luminance Recombine (LRGB)")
        self.setMinimumWidth(460)
        self._color_image = color_image
        self._luma: np.ndarray | None = None
        self._worker: _RecombineWorker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Replaces this color image's own luminance with a separately "
            "stacked/processed L frame -- the classic LRGB (or narrowband "
            "luminance) finishing step. Hue and chroma from the color image "
            "are preserved exactly."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        src = QGroupBox("Luminance frame")
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

        from astraios.core.luminance_recombine import LUMA_PROFILES

        self._luma_method_combo = QComboBox()
        self._luma_method_combo.addItems(list(LUMA_PROFILES.keys()))
        idx = self._luma_method_combo.findText("rec709")
        if idx >= 0:
            self._luma_method_combo.setCurrentIndex(idx)
        form.addRow("Luma method", self._row(
            self._luma_method_combo,
            "How luminance is derived from a color luma source, and the "
            "weights used to measure this image's own luminance during "
            "recombine. Ignored when the luma frame is already mono. "
            "Includes standard Rec.709/601/2020 weightings plus a table of "
            "camera-sensor-specific weights.",
        ))

        self._blend_spin = QDoubleSpinBox()
        self._blend_spin.setRange(0.0, 1.0)
        self._blend_spin.setSingleStep(0.05)
        self._blend_spin.setDecimals(2)
        self._blend_spin.setValue(1.0)
        form.addRow("Blend", self._row(
            self._blend_spin, param_help(
                "Mix ratio between the original color image and the "
                "fully recombined result.",
                higher="Leans more toward the fully luminance-recombined "
                       "result.",
                lower="Leans more toward the original, unrecombined "
                      "color image.",
                default="0.0 = original color, 1.0 = fully recombined.",
            ),
        ))

        self._pedestal_spin = QDoubleSpinBox()
        self._pedestal_spin.setRange(0.0, 0.5)
        self._pedestal_spin.setSingleStep(0.01)
        self._pedestal_spin.setDecimals(2)
        self._pedestal_spin.setValue(0.05)
        form.addRow("Pedestal", self._row(
            self._pedestal_spin, param_help(
                "Noise-floor compression (lift-then-compress) amount "
                "applied before computing the scale factor.",
                how="Protects near-zero pixels from hue skew when the "
                    "luminance scale factor is computed.",
                higher="Protects more of the noise floor, at the cost of "
                       "slightly flattening the faintest signal.",
                lower="Less protection — faint pixels are more exposed "
                      "to hue skew.",
                default="0.0 disables it.",
            ),
        ))

        self._knee_spin = QDoubleSpinBox()
        self._knee_spin.setRange(0.0, 1.0)
        self._knee_spin.setSingleStep(0.05)
        self._knee_spin.setDecimals(2)
        self._knee_spin.setValue(0.0)
        form.addRow("Highlight soft knee", self._row(
            self._knee_spin, param_help(
                "Softens (rolls off) the per-pixel scale factor in "
                "highlights to reduce clipping/halos when the new "
                "luminance is much brighter than the color image's own.",
                higher="Rolls off more of the highlight range — safer "
                       "against clipping/halos, but flattens bright "
                       "cores slightly.",
                lower="Less roll-off, closer to a pure linear scale.",
                default="0.0 disables it (pure linear scale).",
            ),
        ))

        self._sat_spin = QDoubleSpinBox()
        self._sat_spin.setRange(-1.0, 1.0)
        self._sat_spin.setSingleStep(0.05)
        self._sat_spin.setDecimals(2)
        self._sat_spin.setValue(0.0)
        form.addRow("Saturation boost", self._row(
            self._sat_spin, param_help(
                "HSV saturation adjustment applied to the color image "
                "before its luminance is measured and replaced.",
                higher="More saturated color.",
                lower="Less saturated color; the negative end desaturates "
                      "toward grayscale.",
                default="0.0 = no change, 1.0 = double saturation, "
                        "-1.0 = grayscale.",
            ),
        ))

        self._chroma_nr_spin = QDoubleSpinBox()
        self._chroma_nr_spin.setRange(0.0, 20.0)
        self._chroma_nr_spin.setSingleStep(0.5)
        self._chroma_nr_spin.setDecimals(1)
        self._chroma_nr_spin.setValue(0.0)
        form.addRow("Chrominance NR sigma", self._row(
            self._chroma_nr_spin, param_help(
                "Gaussian sigma (pixels) for a chrominance-only noise "
                "reduction pass (blurs Cb/Cr, leaving luminance "
                "untouched) applied to the color image before luminance "
                "is replaced.",
                higher="Smooths color noise more aggressively, but can "
                       "bleed color across fine detail edges.",
                lower="Lighter smoothing — less color-noise cleanup.",
                default="0.0 disables it.",
            ),
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

    @staticmethod
    def _row(widget, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return row

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Luminance Frame", "",
            "Images (*.fits *.fit *.fts *.tif *.tiff *.png *.jpg *.xisf);;All files (*)",
        )
        if not path:
            return
        from astraios.core.image_io import load_image

        try:
            img = load_image(path)
        except Exception as exc:
            self._status.setText(f"Could not load image: {exc}")
            self._luma = None
            self._apply_btn.setEnabled(False)
            return

        data = img.data.astype(np.float32)
        color_hw = self._color_image.shape[-2:]
        luma_hw = data.shape[-2:]
        if color_hw != luma_hw:
            self._status.setText(
                f"Size mismatch: the color image is {color_hw[1]}x{color_hw[0]}, "
                f"but the selected luma frame is {luma_hw[1]}x{luma_hw[0]}. They "
                "must have the same width and height."
            )
            self._luma = None
            self._apply_btn.setEnabled(False)
            return

        self._luma = data
        mono = data.ndim == 2 or (data.ndim == 3 and data.shape[0] == 1)
        self._path_label.setText(
            f"{Path(path).name}" + (" (mono)" if mono else " (color -- luma will be derived)")
        )
        self._status.setText("")
        self._apply_btn.setEnabled(True)

    def get_params(self):
        from astraios.core.luminance_recombine import LuminanceRecombineParams

        return LuminanceRecombineParams(
            luma_method=self._luma_method_combo.currentText(),
            blend=float(self._blend_spin.value()),
            pedestal=float(self._pedestal_spin.value()),
            highlight_soft_knee=float(self._knee_spin.value()),
            saturation_boost=float(self._sat_spin.value()),
            chrominance_nr_sigma=float(self._chroma_nr_spin.value()),
        )

    def _apply(self):
        if self._luma is None:
            self._status.setText("Load a luminance frame first.")
            return
        params = self.get_params()
        self._apply_btn.setEnabled(False)
        self._status.setText("Recombining...")

        self._worker = _RecombineWorker(self._color_image, self._luma, params)
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
