"""Magnitude Tool dialog.

Drives astraios.core.magnitude_tool (aperture photometry + optional
photometric zero-point calibration + limiting-magnitude estimate), ported
from Seti Astro Suite Pro (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import csv
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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

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


class _Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, params):
        super().__init__()
        self._image, self._params = image, params

    def run(self):
        try:
            from astraios.core.magnitude_tool import measure_magnitudes
            self.done.emit(measure_magnitudes(self._image, self._params))
        except Exception as exc:
            log.exception("Magnitude measurement failed")
            self.failed.emit(str(exc))


class MagnitudeToolDialog(QDialog):
    """Detect stars and measure instrumental/calibrated magnitudes.

    Read-only with respect to the working image — this only reports a
    magnitude table, it never modifies the current image.
    """

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measure Magnitudes")
        self.setMinimumSize(640, 540)
        self._image = image
        self._worker: _Worker | None = None
        self._result = None  # set on completion; exposed for scripted verification

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Detects stars and measures aperture-photometry magnitudes for "
            "each. Does not modify the working image."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        form = QFormLayout()
        self._aperture = QDoubleSpinBox()
        self._aperture.setRange(1.0, 100.0)
        self._aperture.setValue(10.0)
        form.addRow(*self._r("Aperture radius", self._aperture, param_help(
            "Radius, in pixels, of the photometry aperture centered on "
            "each detected star.",
            higher="Captures more of a star's flux (and more sky "
                   "background noise) — needed for larger/softer stars.",
            lower="Tighter aperture — less background contamination, but "
                  "can clip flux from bloated or poorly-focused stars.",
        )))
        self._annulus_in = QDoubleSpinBox()
        self._annulus_in.setRange(1.0, 200.0)
        self._annulus_in.setValue(15.0)
        form.addRow(*self._r("Annulus inner", self._annulus_in, param_help(
            "Inner radius of the sky-background annulus, in pixels.",
            higher="Leaves a wider gap between the star aperture and the "
                   "background ring, reducing star-light contamination.",
            lower="Keeps the background ring close to the star — risks "
                  "including some of the star's own light.",
        )))
        self._annulus_out = QDoubleSpinBox()
        self._annulus_out.setRange(2.0, 300.0)
        self._annulus_out.setValue(20.0)
        form.addRow(*self._r("Annulus outer", self._annulus_out, param_help(
            "Outer radius of the sky-background annulus, in pixels.",
            higher="Averages the background over a wider ring — steadier "
                   "estimate, but more likely to catch neighboring stars.",
            lower="Narrower ring — more local, but noisier background "
                  "estimate.",
        )))
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(1.0, 50.0)
        self._threshold.setValue(5.0)
        form.addRow(*self._r("Detection threshold", self._threshold, param_help(
            "Source-detection threshold, in sigma above the background "
            "noise.",
            higher="Stricter — only the brightest, most obvious stars are "
                   "detected.",
            lower="More permissive — detects fainter stars too, but risks "
                  "picking up noise spikes.",
            default="5 is a safe default.",
        )))
        self._max_sources = QSpinBox()
        self._max_sources.setRange(1, 5000)
        self._max_sources.setValue(500)
        form.addRow(*self._r("Max sources", self._max_sources,
                    "Cap on the number of detected stars measured."))

        self._use_zp = QCheckBox("Fix zero point")
        self._use_zp.setToolTip(
            "<qt>Supply a known photometric zero point to calibrate "
            "magnitudes directly, instead of leaving them instrumental "
            "only.</qt>")
        form.addRow(self._use_zp)
        self._zero_point = QDoubleSpinBox()
        self._zero_point.setRange(-50.0, 50.0)
        self._zero_point.setValue(25.0)
        self._zero_point.setEnabled(False)
        self._use_zp.toggled.connect(self._zero_point.setEnabled)
        form.addRow(*self._r("Zero point", self._zero_point,
                    "Magnitude added to the instrumental magnitude when "
                    "fixed above: calibrated = instrumental + zero point."))

        self._limiting_sigma = QDoubleSpinBox()
        self._limiting_sigma.setRange(1.0, 20.0)
        self._limiting_sigma.setValue(5.0)
        form.addRow(*self._r("Limiting mag sigma", self._limiting_sigma, param_help(
            "N-sigma flux threshold used to estimate the limiting "
            "(faintest reliably detectable) magnitude from the background "
            "noise.",
            higher="A more conservative (brighter) limiting-magnitude "
                   "estimate — only counts sources well above the noise.",
            lower="A more optimistic (fainter) limiting-magnitude "
                  "estimate, closer to the noise floor.",
            default="5 is a common choice.",
        )))
        lay.addLayout(form)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["X", "Y", "Flux", "Instrumental Mag", "Calibrated Mag"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        lay.addWidget(self._table, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Measure")
        self._run_btn.clicked.connect(self._run)
        btns.addWidget(self._run_btn)
        self._export_btn = QPushButton("Export CSV...")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        btns.addWidget(self._export_btn)
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

    def _run(self):
        from astraios.core.magnitude_tool import MagnitudeParams

        params = MagnitudeParams(
            aperture_radius=float(self._aperture.value()),
            annulus_inner=float(self._annulus_in.value()),
            annulus_outer=float(self._annulus_out.value()),
            detection_threshold=float(self._threshold.value()),
            max_sources=int(self._max_sources.value()),
            zero_point=float(self._zero_point.value()) if self._use_zp.isChecked() else None,
            limiting_mag_sigma=float(self._limiting_sigma.value()),
        )
        self._run_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        self._status.setText("Measuring...")
        self._worker = _Worker(self._image, params)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._result = result

        n = int(result.n_stars)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(n)
        for row in range(n):
            x, y, flux = float(result.x[row]), float(result.y[row]), float(result.flux[row])
            inst = float(result.instrumental_mag[row])
            cal = float(result.calibrated_mag[row]) if result.calibrated_mag is not None else None
            self._table.setItem(row, 0, _NumericItem(x, f"{x:.2f}"))
            self._table.setItem(row, 1, _NumericItem(y, f"{y:.2f}"))
            self._table.setItem(row, 2, _NumericItem(flux, f"{flux:.4g}"))
            self._table.setItem(row, 3, _NumericItem(inst, f"{inst:.3f}"))
            cal_text = f"{cal:.3f}" if cal is not None else "-"
            self._table.setItem(
                row, 4, _NumericItem(cal if cal is not None else float("inf"), cal_text)
            )
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()

        if n == 0:
            self._status.setText("No stars detected.")
            self._summary.setText("")
            self._export_btn.setEnabled(False)
            return

        if result.zero_point is not None:
            std = result.zero_point_std if result.zero_point_std is not None else 0.0
            zp_text = f"{result.zero_point:.3f} +/- {std:.3f} (n={result.zero_point_n})"
        else:
            zp_text = "not calibrated"
        lim_text = f"{result.limiting_mag:.2f}" if result.limiting_mag is not None else "n/a"
        self._summary.setText(
            f"{n} stars measured. Zero point: {zp_text}. "
            f"Limiting magnitude: {lim_text}."
        )
        self._status.setText("Done.")
        self._export_btn.setEnabled(True)

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")

    def _export_csv(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Magnitudes CSV", "magnitudes.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        try:
            r = self._result
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x", "y", "flux", "instrumental_mag", "calibrated_mag"])
                for i in range(int(r.n_stars)):
                    cal = float(r.calibrated_mag[i]) if r.calibrated_mag is not None else ""
                    writer.writerow([
                        float(r.x[i]), float(r.y[i]), float(r.flux[i]),
                        float(r.instrumental_mag[i]), cal,
                    ])
            self._status.setText(f"Exported {int(r.n_stars)} stars to {path}")
        except Exception as exc:
            self._status.setText(f"Export failed: {exc}")
