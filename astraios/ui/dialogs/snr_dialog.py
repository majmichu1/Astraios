"""SNR (signal-to-noise measurement) tool dialog.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Read-only report: measures signal-to-noise ratio and never modifies the
working image (no ``result_ready`` signal).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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


class _Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, params):
        super().__init__()
        self._image, self._params = image, params

    def run(self):
        try:
            from astraios.core.snr import measure_snr
            self.done.emit(measure_snr(self._image, self._params))
        except Exception as exc:
            log.exception("SNR measurement failed")
            self.failed.emit(str(exc))


class SNRDialog(QDialog):
    """Detect and report signal-to-noise ratio for a region of the image.

    Read-only with respect to the working image -- this only reports SNR
    measurements, it never modifies the current image.
    """

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measure SNR")
        self.setMinimumSize(660, 600)
        self._image = image
        self._worker: _Worker | None = None
        self._result = None  # set on completion; exposed for scripted verification

        if image.ndim == 2:
            self._height, self._width = image.shape
        else:
            _, self._height, self._width = image.shape

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Measures signal-to-noise ratio for a target region against a "
            "background region (or robust whole-frame estimates if no "
            "region is set). Does not modify the working image."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        # ── Background region ───────────────────────────────────────────
        bg_group = QGroupBox("Background region")
        bg_form = QFormLayout(bg_group)
        self._bg_enable = QCheckBox("Use explicit background rectangle")
        bg_help = param_help(
            "Restrict background sampling to a rectangle you specify.",
            how="Median and standard deviation are computed directly "
                "from the pixels inside this rectangle, matching the "
                "original SASpro tool exactly.",
            default="Off: the whole frame is used instead, with "
                    "sigma-clipped robust statistics to reject stars "
                    "and the target itself.",
        )
        self._bg_enable.setToolTip(f"<qt>{bg_help}</qt>")
        bg_form.addRow(self._bg_enable)

        self._bg_x = QSpinBox()
        self._bg_x.setRange(0, max(0, self._width - 1))
        self._bg_y = QSpinBox()
        self._bg_y.setRange(0, max(0, self._height - 1))
        self._bg_w = QSpinBox()
        self._bg_w.setRange(1, max(1, self._width))
        self._bg_w.setValue(min(50, self._width))
        self._bg_h = QSpinBox()
        self._bg_h.setRange(1, max(1, self._height))
        self._bg_h.setValue(min(50, self._height))
        self._bg_spins = (self._bg_x, self._bg_y, self._bg_w, self._bg_h)
        for spin in self._bg_spins:
            spin.setEnabled(False)
        self._bg_enable.toggled.connect(
            lambda on: [s.setEnabled(on) for s in self._bg_spins]
        )

        bg_form.addRow(*self._r("X", self._bg_x, "Left edge of the background rectangle, in px."))
        bg_form.addRow(*self._r("Y", self._bg_y, "Top edge of the background rectangle, in px."))
        bg_form.addRow(*self._r("Width", self._bg_w, "Width of the background rectangle, in px."))
        bg_form.addRow(*self._r("Height", self._bg_h, "Height of the background rectangle, in px."))
        lay.addWidget(bg_group)

        # ── Signal region ────────────────────────────────────────────────
        sig_group = QGroupBox("Signal region")
        sig_form = QFormLayout(sig_group)
        self._sig_enable = QCheckBox("Use explicit signal rectangle")
        sig_help = param_help(
            "Restrict the signal measurement to a rectangle around "
            "your target (star, galaxy, nebula core, etc).",
            how="The median pixel value inside this rectangle is used "
                "as the signal level.",
            default="Off: the whole frame's median stands in for the "
                    "signal level (a quick global check, not a "
                    "per-object measurement).",
        )
        self._sig_enable.setToolTip(f"<qt>{sig_help}</qt>")
        sig_form.addRow(self._sig_enable)

        self._sig_x = QSpinBox()
        self._sig_x.setRange(0, max(0, self._width - 1))
        self._sig_y = QSpinBox()
        self._sig_y.setRange(0, max(0, self._height - 1))
        self._sig_w = QSpinBox()
        self._sig_w.setRange(1, max(1, self._width))
        self._sig_w.setValue(min(50, self._width))
        self._sig_h = QSpinBox()
        self._sig_h.setRange(1, max(1, self._height))
        self._sig_h.setValue(min(50, self._height))
        self._sig_spins = (self._sig_x, self._sig_y, self._sig_w, self._sig_h)
        for spin in self._sig_spins:
            spin.setEnabled(False)
        self._sig_enable.toggled.connect(
            lambda on: [s.setEnabled(on) for s in self._sig_spins]
        )

        sig_form.addRow(*self._r("X", self._sig_x, "Left edge of the signal rectangle, in px."))
        sig_form.addRow(*self._r("Y", self._sig_y, "Top edge of the signal rectangle, in px."))
        sig_form.addRow(*self._r("Width", self._sig_w, "Width of the signal rectangle, in px."))
        sig_form.addRow(*self._r("Height", self._sig_h, "Height of the signal rectangle, in px."))
        lay.addWidget(sig_group)

        # ── Auto-background robustness ──────────────────────────────────
        clip_group = QGroupBox(
            "Auto-background robustness (used when no background rectangle is set)"
        )
        clip_form = QFormLayout(clip_group)
        self._sigma = QDoubleSpinBox()
        self._sigma.setRange(0.5, 10.0)
        self._sigma.setSingleStep(0.5)
        self._sigma.setValue(3.0)
        clip_form.addRow(*self._r("Clip sigma", self._sigma, param_help(
            "Sigma threshold for rejecting outliers (stars, the target "
            "itself) when estimating background from the whole frame.",
            higher="More permissive - keeps more bright pixels in the "
                   "background sample, which can overstate the noise.",
            lower="Stricter - rejects more pixels as outliers, focusing "
                  "the estimate on the quietest sky.",
            default="3.0 is standard.",
        )))
        self._maxiters = QSpinBox()
        self._maxiters.setRange(1, 20)
        self._maxiters.setValue(5)
        clip_form.addRow(*self._r(
            "Clip iterations", self._maxiters,
            "Maximum sigma-clip rejection passes.",
        ))
        lay.addWidget(clip_group)

        self._per_channel = QCheckBox("Per-channel SNR (colour images)")
        self._per_channel.setChecked(True)
        lay.addWidget(self._per_channel)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        self._table = QTableWidget(0, 10)
        self._table.setHorizontalHeaderLabels([
            "Channel", "Signal mean", "Signal median", "Bg mean", "Bg median",
            "Bg std (noise)", "Bg MAD", "Net signal", "SNR", "SNR (dB)",
        ])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self._table, 1)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        lay.addWidget(self._summary)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Measure")
        self._run_btn.clicked.connect(self._run)
        btns.addWidget(self._run_btn)
        self._copy_btn = QPushButton("Copy Results")
        self._copy_btn.setEnabled(False)
        self._copy_btn.clicked.connect(self._copy)
        btns.addWidget(self._copy_btn)
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
        from astraios.core.snr import SNRParams

        bg_bbox = (
            (int(self._bg_x.value()), int(self._bg_y.value()),
             int(self._bg_w.value()), int(self._bg_h.value()))
            if self._bg_enable.isChecked() else None
        )
        sig_bbox = (
            (int(self._sig_x.value()), int(self._sig_y.value()),
             int(self._sig_w.value()), int(self._sig_h.value()))
            if self._sig_enable.isChecked() else None
        )
        params = SNRParams(
            background_bbox=bg_bbox,
            signal_bbox=sig_bbox,
            sigma=float(self._sigma.value()),
            maxiters=int(self._maxiters.value()),
            per_channel=self._per_channel.isChecked(),
        )
        self._run_btn.setEnabled(False)
        self._copy_btn.setEnabled(False)
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

        rows = list(result.channels) + [result.overall]
        self._table.setRowCount(len(rows))
        for row, ch in enumerate(rows):
            values = [
                ch.name,
                f"{ch.signal_mean:.6f}",
                f"{ch.signal_median:.6f}",
                f"{ch.background_mean:.6f}",
                f"{ch.background_median:.6f}",
                f"{ch.background_std:.6f}",
                f"{ch.background_mad:.6f}",
                f"{ch.net_signal:.6f}",
                f"{ch.snr:.3e}",
                f"{ch.snr_db:.2f} dB",
            ]
            for col, val in enumerate(values):
                item = QTableWidgetItem(val)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignLeft if col == 0
                    else Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()

        bg_desc = (
            "explicit rectangle" if not result.background_auto else "sigma-clipped whole frame"
        )
        sig_desc = "explicit rectangle" if not result.signal_auto else "whole frame"
        self._summary.setText(
            f"Overall SNR: {result.overall.snr:.3e} ({result.overall.snr_db:.2f} dB). "
            f"Background from {bg_desc}; signal from {sig_desc}."
        )
        self._status.setText("Done.")
        self._copy_btn.setEnabled(True)

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")

    def _copy(self):
        if self._result is None:
            return
        lines = [
            "Channel\tSignal mean\tSignal median\tBg mean\tBg median\t"
            "Bg std\tBg MAD\tNet signal\tSNR\tSNR (dB)"
        ]
        for ch in list(self._result.channels) + [self._result.overall]:
            lines.append(
                f"{ch.name}\t{ch.signal_mean:.6f}\t{ch.signal_median:.6f}\t"
                f"{ch.background_mean:.6f}\t{ch.background_median:.6f}\t"
                f"{ch.background_std:.6f}\t{ch.background_mad:.6f}\t"
                f"{ch.net_signal:.6f}\t{ch.snr:.6e}\t{ch.snr_db:.3f}"
            )
        try:
            QApplication.clipboard().setText("\n".join(lines))
        except Exception:
            log.warning("Could not access clipboard")
