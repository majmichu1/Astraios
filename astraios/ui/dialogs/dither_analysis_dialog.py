"""Dither Analysis dialog.

Drives astraios.core.dither_analysis (quantify per-frame dither offsets and
report scatter/coverage/nearest-neighbor/clustering/walking-noise
diagnostics across a set of registered frames), ported from Seti Astro
Suite Pro (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


class _ScatterWidget(QWidget):
    """Small custom-painted scatter of per-frame (dx, dy) dither offsets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._dx = []
        self._dy = []

    def set_data(self, dx, dy) -> None:
        self._dx = list(dx)
        self._dy = list(dy)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, Qt.GlobalColor.black)

        painter.setPen(QPen(Qt.GlobalColor.darkGray, 1))
        cx, cy = w / 2.0, h / 2.0
        painter.drawLine(0, int(cy), w, int(cy))
        painter.drawLine(int(cx), 0, int(cx), h)

        if not self._dx:
            painter.setPen(QPen(Qt.GlobalColor.gray))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data")
            return

        margin = 20.0
        max_r = max(1e-6, max(abs(v) for v in (*self._dx, *self._dy)))
        scale = (min(w, h) / 2.0 - margin) / max_r

        painter.setPen(QPen(Qt.GlobalColor.cyan, 1))
        painter.setBrush(Qt.GlobalColor.cyan)
        for x, y in zip(self._dx, self._dy, strict=False):
            px = cx + x * scale
            py = cy + y * scale
            painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)

        painter.setPen(QPen(Qt.GlobalColor.yellow, 1))
        painter.setBrush(Qt.GlobalColor.yellow)
        painter.drawEllipse(int(cx) - 4, int(cy) - 4, 8, 8)


class _Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, frame_paths, params):
        super().__init__()
        self._paths, self._params = frame_paths, params

    def run(self):
        try:
            from astraios.core.dither_analysis import analyze_dither
            self.done.emit(analyze_dither(self._paths, self._params))
        except Exception as exc:
            log.exception("Dither analysis failed")
            self.failed.emit(str(exc))


def _format_report(r) -> str:
    walking = "WALKING NOISE DETECTED" if r.is_walking else "No walking noise detected"
    clustered = f"{r.n_clusters} cluster(s), flagged clustered" if r.is_clustered else \
        f"{r.n_clusters} cluster(s), not flagged"
    return (
        f"Frames analyzed:        {r.n_frames}\n"
        "\n"
        f"Mean dither radius:      {r.mean_radius:.3f} px\n"
        f"Median dither radius:    {r.median_radius:.3f} px\n"
        f"Max dither radius:       {r.max_radius:.3f} px\n"
        f"RMS offset:              {r.rms_offset:.3f} px\n"
        "\n"
        f"Mean step:               {r.mean_step:.3f} px\n"
        f"Max step:                {r.max_step:.3f} px\n"
        f"Std dev (dx, dy):        {r.std_dx:.3f}, {r.std_dy:.3f} px\n"
        f"Span (x, y):             {r.span_x:.3f}, {r.span_y:.3f} px\n"
        "\n"
        f"Coverage (convex hull):  {r.coverage_px:.1f} px^2\n"
        f"Preferred direction:     {r.preferred_direction_deg:.1f} deg\n"
        "\n"
        f"Nearest-neighbor min:    {r.nearest_neighbor_min_px:.3f} px\n"
        f"Nearest-neighbor mean:   {r.nearest_neighbor_mean_px:.3f} px\n"
        "\n"
        f"Clustering:              {clustered}\n"
        "\n"
        f"Walking-noise verdict:   {walking}\n"
        f"  Linearity ratio:       {r.linearity_ratio:.3f} "
        "(PCA 1st/2nd singular value)\n"
        f"  Temporal drift corr:   {r.temporal_drift_corr:.3f}\n"
        f"  Direction consistency: {r.dir_consistency:.3f}\n"
        "\n"
        f"Summary: {r.quality_summary}"
    )


class DitherAnalysisDialog(QDialog):
    """Analyze dither quality across a set of aligned/registered frames."""

    def __init__(self, parent=None, frame_paths: list | None = None):
        super().__init__(parent)
        self.setWindowTitle("Dither Analysis")
        self.setMinimumWidth(560)
        self.setMinimumHeight(520)
        self._frame_paths = list(frame_paths) if frame_paths else []
        self._worker: _Worker | None = None
        self._result = None  # set on completion; exposed for scripted verification

        lay = QVBoxLayout(self)
        intro_row = QHBoxLayout()
        intro = QLabel(
            "Measures per-frame sub-pixel offsets across the aligned frame "
            "set and reports dither spread, coverage, nearest-neighbor "
            "spacing, and whether the pattern looks like healthy random "
            "jitter or systematic drift (\"walking noise\")."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        intro_row.addWidget(intro, 1)
        intro_row.addWidget(help_dot(
            "Offsets are measured by FFT phase correlation against the "
            "first frame. Walking noise (systematic drift rather than "
            "random dithering) hurts hot-pixel/satellite-trail rejection "
            "during stacking — if flagged, consider re-dithering with "
            "larger, more random steps."
        ))
        lay.addLayout(intro_row)

        self._scatter = _ScatterWidget()
        lay.addWidget(self._scatter)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        self._report = QTextEdit()
        self._report.setReadOnly(True)
        self._report.setFontFamily("monospace")
        lay.addWidget(self._report, 1)

        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        if len(self._frame_paths) >= 2:
            self._run()
        else:
            self._status.setText(
                "Provide at least two aligned frames to analyze dither."
            )

    def _run(self):
        from astraios.core.dither_analysis import DitherAnalysisParams

        params = DitherAnalysisParams()
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        self._status.setText(
            f"Analyzing dither across {len(self._frame_paths)} frames…"
        )
        self._worker = _Worker(self._frame_paths, params)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._result = result
        self._status.setText(result.quality_summary)
        self._scatter.set_data(result.dx, result.dy)
        self._report.setPlainText(_format_report(result))

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._status.setText(f"Failed: {msg}")
