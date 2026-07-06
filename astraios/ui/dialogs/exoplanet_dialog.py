"""Exoplanet Transit dialog.

Drives astraios.core.exoplanet_transit (differential aperture photometry
across an aligned frame sequence, producing a light curve and a transit-dip
verdict), ported from Seti Astro Suite Pro (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


class _LightCurveWidget(QWidget):
    """Small custom-painted connected-line plot of relative flux vs. time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._times: list[float] = []
        self._flux: list[float] = []
        self._mid_transit: float | None = None

    def set_data(self, times, flux, mid_transit=None) -> None:
        self._times = list(times)
        self._flux = list(flux)
        self._mid_transit = mid_transit
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, Qt.GlobalColor.black)

        finite = [
            (t, f) for t, f in zip(self._times, self._flux, strict=False)
            if np.isfinite(t) and np.isfinite(f)
        ]
        if len(finite) < 2:
            painter.setPen(QPen(Qt.GlobalColor.gray))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data")
            return

        margin = 24.0
        ts = [t for t, _ in finite]
        fs = [f for _, f in finite]
        t_min, t_max = min(ts), max(ts)
        f_min, f_max = min(fs), max(fs)
        if t_max <= t_min:
            t_max = t_min + 1.0
        if f_max <= f_min:
            f_max = f_min + 1e-6
        # A little headroom so the curve doesn't hug the plot edges.
        pad = 0.05 * (f_max - f_min)
        f_min -= pad
        f_max += pad

        def to_px(t: float, f: float) -> tuple[float, float]:
            x = margin + (t - t_min) / (t_max - t_min) * (w - 2 * margin)
            y = h - margin - (f - f_min) / (f_max - f_min) * (h - 2 * margin)
            return x, y

        if f_min <= 1.0 <= f_max:
            _, y1 = to_px(t_min, 1.0)
            painter.setPen(QPen(Qt.GlobalColor.darkGray, 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(margin), int(y1), int(w - margin), int(y1))

        if self._mid_transit is not None and t_min <= self._mid_transit <= t_max:
            xm, _ = to_px(self._mid_transit, f_min)
            painter.setPen(QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(xm), int(margin), int(xm), int(h - margin))

        painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
        prev = None
        for t, f in finite:
            x, y = to_px(t, f)
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)


class _Worker(QThread):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, frame_paths, target_xy, params):
        super().__init__()
        self._paths = frame_paths
        self._target_xy = target_xy
        self._params = params

    def run(self):
        try:
            from astraios.core.exoplanet_transit import analyze_transit
            result = analyze_transit(
                self._paths, self._target_xy, comparison_xys=None,
                params=self._params,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.done.emit(result)
        except Exception as exc:
            log.exception("Exoplanet transit analysis failed")
            self.failed.emit(str(exc))


class ExoplanetDialog(QDialog):
    """Measure a differential light curve and flag a transit dip."""

    def __init__(self, parent=None, frame_paths=None):
        super().__init__(parent)
        self.setWindowTitle("Exoplanet Transit")
        self.setMinimumSize(560, 600)
        self._frame_paths = [str(p) for p in frame_paths] if frame_paths else []
        self._worker: _Worker | None = None
        self._result = None  # exposed for scripted verification

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Measures a differential (target vs. ensemble-comparison) light "
            "curve across an aligned frame sequence and flags a transit-"
            "shaped dip. Read the target star's pixel X/Y off the first "
            "frame before running."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        form = QFormLayout()
        self._target_x = QDoubleSpinBox()
        self._target_x.setRange(0.0, 100000.0)
        form.addRow(*self._r("Target X", self._target_x,
                    "Pixel X coordinate of the target star, read off the "
                    "first frame."))
        self._target_y = QDoubleSpinBox()
        self._target_y.setRange(0.0, 100000.0)
        form.addRow(*self._r("Target Y", self._target_y,
                    "Pixel Y coordinate of the target star."))

        self._auto_select = QCheckBox("Auto-select comparison stars")
        self._auto_select.setChecked(True)
        self._auto_select.setToolTip(
            "<qt>Detect and rank nearby comparison stars automatically "
            "(brightest and most frame-to-frame stable). Unticking falls "
            "back to a single comparison star instead of an ensemble.</qt>")
        form.addRow(self._auto_select)
        self._n_comparisons = QSpinBox()
        self._n_comparisons.setRange(1, 30)
        self._n_comparisons.setValue(5)
        form.addRow(*self._r("Comparison stars", self._n_comparisons,
                    "How many auto-selected comparison stars to combine "
                    "into the reference flux (ignored when auto-select is "
                    "off — a single comparison star is used instead)."))

        self._aperture = QDoubleSpinBox()
        self._aperture.setRange(1.0, 100.0)
        self._aperture.setValue(10.0)
        form.addRow(*self._r("Aperture radius", self._aperture,
                    "Photometry aperture radius, in pixels."))
        self._annulus_in = QDoubleSpinBox()
        self._annulus_in.setRange(1.0, 200.0)
        self._annulus_in.setValue(15.0)
        form.addRow(*self._r("Annulus inner", self._annulus_in,
                    "Inner radius of the sky-background annulus, in "
                    "pixels."))
        self._annulus_out = QDoubleSpinBox()
        self._annulus_out.setRange(2.0, 300.0)
        self._annulus_out.setValue(20.0)
        form.addRow(*self._r("Annulus outer", self._annulus_out,
                    "Outer radius of the sky-background annulus, in "
                    "pixels."))

        self._detrend = QComboBox()
        self._detrend.addItems(["None", "Linear", "Quadratic"])
        self._detrend.setCurrentIndex(2)
        form.addRow(*self._r("Detrend", self._detrend,
                    "Removes a slow polynomial baseline (airmass, focus, "
                    "cloud drift) from the light curve before dip "
                    "detection."))

        self._threshold_ppt = QDoubleSpinBox()
        self._threshold_ppt.setRange(0.1, 500.0)
        self._threshold_ppt.setValue(20.0)
        self._threshold_ppt.setSuffix(" ppt")
        form.addRow(*self._r("Detection threshold", self._threshold_ppt,
                    "Minimum dip depth, in parts-per-thousand, required to "
                    "flag a transit detection."))
        lay.addLayout(form)

        self._frames_label = QLabel(self._frames_status_text())
        self._frames_label.setWordWrap(True)
        lay.addWidget(self._frames_label)

        self._curve = _LightCurveWidget()
        lay.addWidget(self._curve)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)
        self._verdict = QLabel("")
        self._verdict.setWordWrap(True)
        lay.addWidget(self._verdict)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Analyze Transit")
        self._run_btn.setEnabled(len(self._frame_paths) >= 3)
        self._run_btn.clicked.connect(self._run)
        btns.addWidget(self._run_btn)
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    def _frames_status_text(self) -> str:
        n = len(self._frame_paths)
        if n == 0:
            return (
                "No frames provided — open this from Tools > Exoplanet "
                "Transit with an aligned frame set loaded."
            )
        if n < 10:
            return (
                f"{n} frame(s) provided (fewer than the ~10+ recommended "
                "for a reliable light curve)."
            )
        return f"{n} frames ready."

    @staticmethod
    def _r(label, widget, tip):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return label, row

    def _run(self):
        if len(self._frame_paths) < 3:
            self._status.setText("Need at least 3 frames to analyze a transit.")
            return
        from astraios.core.exoplanet_transit import DetrendMethod, ExoplanetTransitParams

        detrend_map = [DetrendMethod.NONE, DetrendMethod.LINEAR, DetrendMethod.QUADRATIC]
        params = ExoplanetTransitParams(
            aperture_radius=float(self._aperture.value()),
            annulus_inner=float(self._annulus_in.value()),
            annulus_outer=float(self._annulus_out.value()),
            n_comparison_stars=(
                int(self._n_comparisons.value()) if self._auto_select.isChecked() else 1
            ),
            detrend_method=detrend_map[self._detrend.currentIndex()],
            detection_threshold_ppt=float(self._threshold_ppt.value()),
        )
        target_xy = (float(self._target_x.value()), float(self._target_y.value()))

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._status.setText("Analyzing...")
        self._verdict.setText("")
        self._worker = _Worker(self._frame_paths, target_xy, params)
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
        self._curve.set_data(result.times, result.relative_flux, result.mid_transit_time)

        depth_ppt = result.transit_depth * 1000.0
        if result.transit_detected:
            mid = (
                f"{result.mid_transit_time:.5f}"
                if result.mid_transit_time is not None else "n/a"
            )
            unit = "JD" if result.time_is_jd else "frame index"
            self._verdict.setText(
                f"Transit DETECTED — depth {depth_ppt:.2f} ppt, "
                f"mid-transit {mid} ({unit}), {result.n_good_frames} good frames."
            )
            self._verdict.setStyleSheet("color: #6bcb6b; font-weight: bold;")
        else:
            self._verdict.setText(
                f"No transit detected (deepest dip {depth_ppt:.2f} ppt, "
                f"{result.n_good_frames} good frames)."
            )
            self._verdict.setStyleSheet("color: #ccc;")
        self._status.setText("Done.")

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
