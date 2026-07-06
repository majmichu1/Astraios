"""Planetary/solar de-rotation dialog.

Drives astraios.core.derotate (de-rotate a SER video's frames to a common
reference-time view before stacking), ported from Seti Astro Suite Pro
(GPL-3.0, Franklin Marek). See astraios/core/derotate.py for the ported
geometry and the GPU/CPU dispatch notes.

This dialog loads frames from a SER file, de-rotates them, and produces a
simple mean-combine of the de-rotated stack as its result. It is a
companion to SER Planetary Stacker (ser_stacker_dialog.py), not a
replacement: that dialog does quality-ranked lucky-imaging selection and
sigma-clip/median combine but has no field-rotation correction (see its
module docstring's "Deferred relative to SASpro" note). For a fast-spinning
target (Jupiter, the Sun), run this first to de-rotate, or use it standalone
for a quick rotation-corrected average.
"""

from __future__ import annotations

import logging

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
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

#: .NET ticks (SER trailer timestamp unit) -> seconds.
_TICKS_PER_SECOND = 1.0e7


class _DerotateWorker(QThread):
    progress = pyqtSignal(float, str)
    finished_ok = pyqtSignal(object)  # ndarray (mean-combined de-rotated stack)
    failed = pyqtSignal(str)

    def __init__(self, ser_path: str, params, max_frames: int | None):
        super().__init__()
        self._ser_path = ser_path
        self._params = params
        self._max_frames = max_frames

    def run(self):
        try:
            from astraios.core.derotate import derotate_frames
            from astraios.core.ser_reader import SERFrameReader

            frames: list[np.ndarray] = []
            times_s: list[float] = []
            have_all_timestamps = True

            with SERFrameReader(self._ser_path) as reader:
                n = len(reader)
                if self._max_frames:
                    n = min(n, self._max_frames)
                for i in range(n):
                    frames.append(reader.read_frame(i))
                    ts = reader.read_timestamp(i)
                    if ts is None:
                        have_all_timestamps = False
                        times_s.append(float(i))
                    else:
                        times_s.append(ts / _TICKS_PER_SECOND)
                    self.progress.emit(0.5 * (i + 1) / max(n, 1), f"Reading frame {i + 1}/{n}")

            if have_all_timestamps and times_s:
                t0 = times_s[0]
                times_s = [t - t0 for t in times_s]
                self._params.frame_times_s = times_s
            else:
                self._params.frame_times_s = None
                log.warning(
                    "SER file has no per-frame trailer timestamps; rotation "
                    "scheduling falls back to uniform frame-index spacing."
                )

            def _prog(frac: float, message: str):
                self.progress.emit(0.5 + 0.5 * frac, message)

            out = derotate_frames(frames, self._params, progress=_prog)
            if not out:
                self.finished_ok.emit(None)
                return
            stacked = np.mean(np.stack(out, axis=0), axis=0).astype(np.float32)
            self.finished_ok.emit(stacked)
        except Exception as exc:
            log.exception("De-rotation failed")
            self.failed.emit(str(exc))


class DerotateDialog(QDialog):
    """De-rotate a SER video's frames and mean-combine the result."""

    result_ready = pyqtSignal(object)

    def __init__(self, parent=None, ser_path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Planetary De-rotation")
        self.setMinimumWidth(480)
        self._worker: _DerotateWorker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Corrects a rotating body's apparent spin across a SER capture "
            "before stacking: each frame is unprojected onto a sphere, "
            "shifted back to a reference-time orientation, and re-sampled. "
            "The de-rotated frames are then mean-combined."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

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

        disc = QGroupBox("Disc Geometry")
        dform = QFormLayout(disc)
        self._cx = QDoubleSpinBox()
        self._cx.setRange(0.0, 100000.0)
        self._cy = QDoubleSpinBox()
        self._cy.setRange(0.0, 100000.0)
        self._r = QDoubleSpinBox()
        self._r.setRange(1.0, 100000.0)
        self._r.setValue(100.0)
        dform.addRow("Center X (px)", self._row(self._cx, param_help(
            "Disc center, X pixel coordinate.",
            how="The sphere used for de-rotation is centered here; get this "
                "from the SER Viewer's first frame.",
        )))
        dform.addRow("Center Y (px)", self._row(self._cy, param_help(
            "Disc center, Y pixel coordinate.",
        )))
        dform.addRow("Radius (px)", self._row(self._r, param_help(
            "Disc (limb) radius in pixels.",
            how="Everything outside this radius is treated as off-disc "
                "(background/sky) and left untouched.",
        )))
        self._pole_angle = QDoubleSpinBox()
        self._pole_angle.setRange(-180.0, 180.0)
        self._pole_angle.setSuffix(" deg")
        dform.addRow("Pole angle", self._row(self._pole_angle, param_help(
            "Rotates image coordinates so the planet's spin axis points "
            "\"up\" before the rotation math runs.",
            how="0 if the pole already points up in the frame (typical for "
                "an alt-az mount without field derotation).",
        )))
        self._subobs_lat = QDoubleSpinBox()
        self._subobs_lat.setRange(-90.0, 90.0)
        self._subobs_lat.setSuffix(" deg")
        dform.addRow("Sub-observer lat", self._row(self._subobs_lat, param_help(
            "Sub-observer latitude — how much the pole is tilted toward or "
            "away from the observer.",
            higher="More pole visible (tilted toward you).",
            lower="More of the opposite pole visible (tilted away).",
            default="0 = equator-on.",
        )))
        lay.addWidget(disc)

        rot = QGroupBox("Rotation")
        rform = QFormLayout(rot)
        self._rate = QDoubleSpinBox()
        self._rate.setRange(-100000.0, 100000.0)
        self._rate.setDecimals(3)
        self._rate.setSuffix(" deg/hr")
        rform.addRow("Rotation rate", self._row(self._rate, param_help(
            "The body's apparent rotation rate at the time of capture.",
            how="Each frame is shifted back by rate x (its timestamp minus "
                "the reference frame's timestamp), using the SER trailer's "
                "per-frame timestamps when present. Jupiter's System II "
                "rate is about 870 deg/hr; the Sun's synodic rate is much "
                "slower (roughly 0.02-0.05 deg/hr depending on latitude).",
            tip="Sign matters: if the correction moves features the wrong "
                "way, negate this value.",
        )))
        self._ref_index = QSpinBox()
        self._ref_index.setRange(0, 1000000)
        rform.addRow("Reference frame", self._row(self._ref_index, param_help(
            "Index of the frame all others are de-rotated to match "
            "(0 = first frame read).",
        )))
        lay.addWidget(rot)

        opts = QGroupBox("Resample")
        oform = QFormLayout(opts)
        self._interp = QComboBox()
        self._interp.addItems(["Nearest", "Linear", "Cubic"])
        self._interp.setCurrentText("Cubic")
        oform.addRow("Interpolation", self._row(self._interp, param_help(
            "Resample quality for the per-pixel sphere warp.",
            higher="Cubic is sharpest but can ring slightly at the limb.",
            lower="Nearest is fastest and ring-free but blocky.",
        )))
        self._max_frames = QSpinBox()
        self._max_frames.setRange(0, 1000000)
        self._max_frames.setSpecialValueText("All")
        oform.addRow("Max frames", self._row(self._max_frames, param_help(
            "Cap the number of frames read from the SER file. 0 = all.",
        )))
        lay.addWidget(opts)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("De-rotate && Combine")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        btns.addWidget(self._run_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        if ser_path:
            self._file_edit.setText(ser_path)

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
        self._run_btn.setEnabled(bool(text))
        if not text:
            self._info_label.setText("")
            return
        try:
            from astraios.core.ser_reader import read_ser_header
            h = read_ser_header(text)
            self._info_label.setText(
                f"{h.frame_count} frames, {h.width}x{h.height}, "
                f"{h.pixel_depth}-bit, {h.color_name}"
            )
            if self._cx.value() == 0.0 and self._cy.value() == 0.0:
                self._cx.setValue(h.width / 2.0)
                self._cy.setValue(h.height / 2.0)
        except Exception as exc:
            self._info_label.setText(f"Could not read header: {exc}")

    def _run(self):
        from astraios.core.derotate import DerotateParams

        interp_map = {"Nearest": "nearest", "Linear": "linear", "Cubic": "cubic"}
        params = DerotateParams(
            cx=float(self._cx.value()),
            cy=float(self._cy.value()),
            r=float(self._r.value()),
            pole_angle_rad=float(np.deg2rad(self._pole_angle.value())),
            subobs_lat_rad=float(np.deg2rad(self._subobs_lat.value())),
            rotation_rate_deg_per_hour=float(self._rate.value()),
            reference_index=int(self._ref_index.value()),
            interpolation=interp_map[self._interp.currentText()],
        )
        max_frames = int(self._max_frames.value()) or None

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("De-rotating...")
        self._worker = _DerotateWorker(self._file_edit.text().strip(), params, max_frames)
        self._worker.progress.connect(
            lambda f, m: (self._progress.setValue(int(f * 100)), self._status.setText(m))
        )
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setValue(100)
        if not isinstance(result, np.ndarray):
            self._status.setText("De-rotation produced no image.")
            self._run_btn.setEnabled(True)
            return
        self._status.setText("Done.")
        self.result_ready.emit(result)
        self.accept()

    def _on_fail(self, msg: str):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
