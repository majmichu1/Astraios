"""SER Viewer dialog — scrub, play, and inspect a planetary/lunar/solar
SER video, and send any single frame to the canvas.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro)
``serviewer.py``, Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

``serviewer.py`` is a 1800+ line viewer built around a much larger feature
set (ROI crop, Bayer-pattern override/auto-detect, ROI-based trim+export,
zoom/pan, live surface-anchor tracking, batch-stacker launch). Only what the
task calls out is ported here: frame scrub (slider + prev/next), playback
(``_toggle_play``/``_tick_playback``, looping at the last frame, the same
~30 fps ``QTimer`` interval of 33 ms), and per-frame stats (the source's
``lbl_frame`` "``{cur+1} / {frames}``" counter; the source has no numeric
per-frame statistics display beyond that counter, so the min/mean/max/std
and lucky-imaging sharpness score shown here are an Astraios addition,
reusing ``ser_stacker.frame_quality_score`` rather than inventing a new
metric). Decoding reuses ``astraios.core.ser_reader.SERFrameReader``
directly instead of porting SASpro's own reader.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)

from astraios.core.image_io import ImageData
from astraios.core.ser_reader import SERFrameReader

log = logging.getLogger(__name__)

#: Matches serviewer.py's ``self._timer.setInterval(33)`` (~30 fps scrub/play).
_PLAYBACK_INTERVAL_MS = 33


class SERViewerDialog(QDialog):
    """Scrub/play a SER video and send a chosen frame to the canvas."""

    frame_selected = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, path: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("SER Viewer")
        self.resize(820, 660)
        self.setMinimumSize(520, 420)

        self._reader: SERFrameReader | None = None
        self._cur = 0
        self._playing = False

        self._timer = QTimer(self)
        self._timer.setInterval(_PLAYBACK_INTERVAL_MS)
        self._timer.timeout.connect(self._tick_playback)

        self._build_ui()
        if path:
            self._open(path)

    # ── UI setup ──────────────────────────────────────────────────────────

    def _build_ui(self):
        lay = QVBoxLayout(self)

        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Choose a .ser video file...")
        self._path_edit.setReadOnly(True)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        file_row.addWidget(self._path_edit, 1)
        file_row.addWidget(browse)
        lay.addLayout(file_row)

        self._info_label = QLabel("No SER file loaded.")
        self._info_label.setStyleSheet("color: #888;")
        self._info_label.setWordWrap(True)
        lay.addWidget(self._info_label)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(400, 300)
        self._image_label.setStyleSheet(
            "background-color: #000; border: 1px solid #30363d; border-radius: 4px;"
        )
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        lay.addWidget(self._image_label, 1)

        # Scrub row: prev / slider / next / frame counter (source's
        # sld + lbl_frame, plus prev/next convenience buttons).
        scrub = QHBoxLayout()
        self._btn_prev = QPushButton("|<")
        self._btn_prev.setFixedWidth(32)
        self._btn_prev.clicked.connect(self._prev)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._btn_next = QPushButton(">|")
        self._btn_next.setFixedWidth(32)
        self._btn_next.clicked.connect(self._next)
        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(90)
        scrub.addWidget(self._btn_prev)
        scrub.addWidget(self._slider, 1)
        scrub.addWidget(self._btn_next)
        scrub.addWidget(self._frame_label)
        lay.addLayout(scrub)

        # Playback row.
        play_row = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_play.clicked.connect(self._toggle_play)
        play_row.addWidget(self._btn_play)
        play_row.addStretch()
        lay.addLayout(play_row)

        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #aaa;"
        )
        lay.addWidget(self._stats_label)

        btns = QHBoxLayout()
        self._send_btn = QPushButton("Send Current Frame to Canvas")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_current_frame)
        btns.addWidget(self._send_btn)
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── File loading ──────────────────────────────────────────────────────

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SER Video", "", "SER Video (*.ser *.SER);;All Files (*)"
        )
        if path:
            self._open(path)

    def _open(self, path: str):
        self._stop_playback()
        if self._reader is not None:
            self._reader.close()
            self._reader = None

        try:
            self._reader = SERFrameReader(path)
        except Exception as exc:
            log.warning("Failed to open SER file %s: %s", path, exc)
            self._info_label.setText(f"Could not open file: {exc}")
            self._send_btn.setEnabled(False)
            return

        self._path_edit.setText(path)
        h = self._reader.header
        self._info_label.setText(
            f"{h.frame_count} frames, {h.width}x{h.height}, "
            f"{h.pixel_depth}-bit, {h.color_name}"
        )
        self._cur = 0
        self._slider.blockSignals(True)
        self._slider.setRange(0, max(0, h.frame_count - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._send_btn.setEnabled(h.frame_count > 0)
        self._refresh()

    # ── Scrub / playback (ported from serviewer.py) ──────────────────────

    def _prev(self):
        if self._reader is None:
            return
        if self._cur > 0:
            self._cur -= 1
            self._slider.blockSignals(True)
            self._slider.setValue(self._cur)
            self._slider.blockSignals(False)
            self._refresh()

    def _next(self):
        if self._reader is None:
            return
        if self._cur < len(self._reader) - 1:
            self._cur += 1
            self._slider.blockSignals(True)
            self._slider.setValue(self._cur)
            self._slider.blockSignals(False)
            self._refresh()

    def _on_slider_changed(self, v: int):
        self._cur = int(v)
        self._refresh()

    def _toggle_play(self):
        if self._reader is None:
            return
        self._playing = not self._playing
        self._btn_play.setText("Pause" if self._playing else "Play")
        if self._playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _stop_playback(self):
        self._playing = False
        self._timer.stop()
        self._btn_play.setText("Play")

    def _tick_playback(self):
        if self._reader is None:
            return
        if self._cur >= len(self._reader) - 1:
            self._cur = 0
        else:
            self._cur += 1
        self._slider.blockSignals(True)
        self._slider.setValue(self._cur)
        self._slider.blockSignals(False)
        self._refresh()

    # ── Rendering / stats ─────────────────────────────────────────────────

    def _refresh(self):
        if self._reader is None:
            return
        total = len(self._reader)
        self._frame_label.setText(f"{self._cur + 1} / {total}")
        if total == 0:
            return

        try:
            frame = self._reader.read_frame(self._cur)
        except Exception as exc:
            log.warning("Failed to read SER frame %d: %s", self._cur, exc)
            self._stats_label.setText(f"Frame read failed: {exc}")
            return

        self._current_frame = frame
        self._update_stats(frame)

        image = ImageData(data=frame, header={})
        display = image.to_display(stretch=True)
        pixmap = self._ndarray_to_pixmap(display)
        if not self._image_label.size().isNull():
            scaled = pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._image_label.setPixmap(scaled)

    def _update_stats(self, frame: np.ndarray):
        from astraios.core.ser_stacker import frame_quality_score

        gray = frame.mean(axis=0) if frame.ndim == 3 else frame
        quality = frame_quality_score(gray)
        self._stats_label.setText(
            f"min={frame.min():.4f}  mean={frame.mean():.4f}  "
            f"max={frame.max():.4f}  std={frame.std():.4f}  "
            f"sharpness={quality:.3f}"
        )

    @staticmethod
    def _ndarray_to_pixmap(arr: np.ndarray) -> QPixmap:
        h, w, ch = arr.shape
        fmt = QImage.Format.Format_RGB888 if ch >= 3 else QImage.Format.Format_Grayscale8
        if ch > 3:
            arr = arr[:, :, :3]
            ch = 3
        arr = np.ascontiguousarray(arr)
        img = QImage(arr.tobytes(), w, h, w * ch, fmt)
        return QPixmap.fromImage(img.copy())

    # ── Send to canvas ───────────────────────────────────────────────────

    def _send_current_frame(self):
        frame = getattr(self, "_current_frame", None)
        if frame is None:
            return
        self.frame_selected.emit(frame)

    # ── Events ────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent | None):  # noqa: N802
        if event is None:
            return
        if event.key() == Qt.Key.Key_Left:
            self._prev()
        elif event.key() == Qt.Key.Key_Right:
            self._next()
        elif event.key() == Qt.Key.Key_Space:
            self._toggle_play()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._reader is not None:
            self._refresh()

    def closeEvent(self, event):  # noqa: N802
        self._stop_playback()
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        super().closeEvent(event)
