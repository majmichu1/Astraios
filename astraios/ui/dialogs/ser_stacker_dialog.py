"""SER planetary/lunar/solar stacking dialog.

Drives astraios.core.ser_stacker (lucky-imaging stacking of a SER video),
which is ported from Seti Astro Suite Pro (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
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


class _StackWorker(QThread):
    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(object)  # ndarray
    failed = pyqtSignal(str)

    def __init__(self, ser_path, params):
        super().__init__()
        self._ser_path = ser_path
        self._params = params

    def run(self):
        try:
            from astraios.core.ser_stacker import stack_ser

            # ser_stacker's callback is (current, total, message); the UI
            # bar wants a 0..1 fraction.
            def _prog(current, total, message):
                frac = current / total if total else 0.0
                self.progress.emit(min(max(frac, 0.0), 1.0), message)

            result = stack_ser(self._ser_path, self._params, progress=_prog)
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("SER stacking failed")
            self.failed.emit(str(exc))


class SERStackerDialog(QDialog):
    """Stack a SER planetary/lunar/solar video via lucky imaging."""

    result_ready = pyqtSignal(object)  # stacked ndarray

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SER Planetary Stacker")
        self.setMinimumWidth(460)
        self._worker: _StackWorker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Stacks a SER video of a planet, the Moon, or the Sun using "
            "lucky imaging: it scores every frame for sharpness, keeps the "
            "best, aligns them, and combines them into one clean image."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        # File row
        file_row = QHBoxLayout()
        self._file_edit = QLineEdit()
        self._file_edit.setPlaceholderText("Choose a .ser video file...")
        self._file_edit.textChanged.connect(self._on_file_changed)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        file_row.addWidget(self._file_edit, 1)
        file_row.addWidget(browse)
        lay.addLayout(file_row)
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #80c0ff; font-size: 11px;")
        lay.addWidget(self._info_label)

        # Lucky imaging
        lucky = QGroupBox("Lucky Imaging")
        lform = QFormLayout(lucky)
        self._keep = QDoubleSpinBox()
        self._keep.setRange(1.0, 100.0)
        self._keep.setValue(20.0)
        self._keep.setSuffix(" %")
        lform.addRow("Keep best", self._row(self._keep,
                     "Percent of the sharpest frames to keep. 10-30% is "
                     "typical: fewer frames are sharper but noisier."))
        self._align_combo = QComboBox()
        self._align_combo.addItems(["Phase correlation", "Centroid", "None"])
        lform.addRow("Alignment", self._row(self._align_combo,
                     "How kept frames are registered to the reference. "
                     "Phase correlation is the accurate default; Centroid is "
                     "faster for a bright disk on black; None skips alignment."))
        self._ref_combo = QComboBox()
        self._ref_combo.addItems(["Best single frame", "Mean of best N"])
        lform.addRow("Reference", self._row(self._ref_combo,
                     "The alignment reference. Mean of best N is a steadier "
                     "target when the single best frame is still noisy."))
        self._upsample = QSpinBox()
        self._upsample.setRange(1, 20)
        self._upsample.setValue(4)
        lform.addRow("Sub-pixel", self._row(self._upsample,
                     "Sub-pixel alignment precision. Higher aligns more "
                     "finely (slower); 4 is a good default."))
        lay.addWidget(lucky)

        # Integration
        integ = QGroupBox("Integration")
        iform = QFormLayout(integ)
        self._integ_combo = QComboBox()
        self._integ_combo.addItems(["Average", "Median", "Sigma clip"])
        iform.addRow("Method", self._row(self._integ_combo,
                     "How aligned frames are combined. Average is smoothest, "
                     "Median rejects moving artifacts, Sigma clip rejects "
                     "outliers statistically."))
        self._normalize = QCheckBox("Normalize frame brightness")
        self._normalize.setChecked(True)
        self._normalize.setToolTip(
            "<qt>Match each frame's brightness to the reference before "
            "combining, so transparency changes during capture don't "
            "cause banding.</qt>")
        iform.addRow(self._normalize)
        self._drizzle = QDoubleSpinBox()
        self._drizzle.setRange(1.0, 3.0)
        self._drizzle.setSingleStep(0.5)
        self._drizzle.setValue(1.0)
        self._drizzle.setSuffix("x")
        iform.addRow("Drizzle scale", self._row(self._drizzle,
                     "Upscale the output canvas. 1.5x-2x can pull out extra "
                     "detail when you have many well-aligned frames."))
        self._max_frames = QSpinBox()
        self._max_frames.setRange(0, 1000000)
        self._max_frames.setValue(0)
        self._max_frames.setSpecialValueText("All")
        iform.addRow("Max frames", self._row(self._max_frames,
                     "Cap the number of frames scanned, for a quick preview "
                     "on a huge capture. 0 = use all frames."))
        lay.addWidget(integ)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._stack_btn = QPushButton("Stack SER")
        self._stack_btn.setEnabled(False)
        self._stack_btn.clicked.connect(self._stack)
        btns.addWidget(self._stack_btn)
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
            self, "Choose SER Video", "", "SER Video (*.ser *.SER);;All Files (*)"
        )
        if path:
            self._file_edit.setText(path)

    def _on_file_changed(self, text: str):
        text = text.strip()
        self._stack_btn.setEnabled(bool(text))
        if not text:
            self._info_label.setText("")
            return
        try:
            from astraios.core.ser_stacker import read_ser_header
            h = read_ser_header(text)
            self._info_label.setText(
                f"{h.frame_count} frames, {h.width}x{h.height}, "
                f"{h.pixel_depth}-bit, {h.color_name}"
            )
        except Exception as exc:
            self._info_label.setText(f"Could not read header: {exc}")

    def _stack(self):
        from astraios.core.ser_stacker import SERStackParams

        align_map = {"Phase correlation": "phase_correlation",
                     "Centroid": "centroid", "None": "none"}
        integ_map = {"Average": "average", "Median": "median",
                     "Sigma clip": "sigma_clip"}
        ref_map = {"Best single frame": "best_frame", "Mean of best N": "best_stack"}
        params = SERStackParams(
            keep_percent=float(self._keep.value()),
            alignment_method=align_map[self._align_combo.currentText()],
            reference_mode=ref_map[self._ref_combo.currentText()],
            upsample_factor=int(self._upsample.value()),
            integration_method=integ_map[self._integ_combo.currentText()],
            normalize_frames=self._normalize.isChecked(),
            drizzle_scale=float(self._drizzle.value()),
            max_frames=(int(self._max_frames.value()) or None),
        )
        self._stack_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Stacking...")
        self._worker = _StackWorker(self._file_edit.text().strip(), params)
        self._worker.progress.connect(
            lambda f, m: (self._progress.setValue(int(f * 100)),
                          self._status.setText(m)))
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setValue(100)
        if not isinstance(result, np.ndarray):
            self._status.setText("Stacking produced no image.")
            self._stack_btn.setEnabled(True)
            return
        self._status.setText("Done.")
        self.result_ready.emit(result)
        self.accept()

    def _on_fail(self, msg: str):
        self._progress.setVisible(False)
        self._stack_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
