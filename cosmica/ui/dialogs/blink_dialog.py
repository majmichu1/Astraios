"""Blink Dialog — quick full-frame viewer for scrolling through subs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap, QWheelEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
)

from cosmica.core.image_io import load_image

log = logging.getLogger(__name__)


class BlinkDialog(QDialog):
    """Scroll through image subs to reject bad ones."""

    rejected = pyqtSignal(list)

    FPS_MIN = 0.5
    FPS_MAX = 5.0
    FPS_DEFAULT = 2.0

    def __init__(self, frame_paths: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._frame_paths = list(frame_paths) if frame_paths else []
        self._current_idx = 0
        self._rejected: set[str] = set()
        self._auto_active = False

        self._auto_timer = QTimer(self)
        self._auto_timer.timeout.connect(self._advance)

        self._setup_ui()
        self._show_frame()

    # ── UI Setup ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("Blink — Frame Selector")
        self.resize(1000, 750)
        self.setMinimumSize(600, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(400, 300)
        self._image_label.setStyleSheet(
            "background-color: #000; border: 1px solid #30363d; border-radius: 4px;"
        )
        self._image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._image_label, 1)

        # Controls row ──────────────────────────────────────────────────────
        controls = QHBoxLayout()
        controls.setSpacing(6)

        self._btn_prev = QPushButton("◀  Prev")
        self._btn_prev.clicked.connect(self._prev)

        self._btn_next = QPushButton("Next  ▶")
        self._btn_next.clicked.connect(self._next)

        self._btn_auto = QPushButton("▶▶  Auto")
        self._btn_auto.setCheckable(True)
        self._btn_auto.clicked.connect(self._toggle_auto)

        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(int(self.FPS_MIN * 10), int(self.FPS_MAX * 10))
        self._speed_slider.setValue(int(self.FPS_DEFAULT * 10))
        self._speed_slider.setFixedWidth(120)
        self._speed_slider.valueChanged.connect(self._on_speed_changed)

        self._speed_label = QLabel(f"{self.FPS_DEFAULT:.1f} fps")
        self._speed_label.setFixedWidth(48)

        controls.addWidget(self._btn_prev)
        controls.addWidget(self._btn_next)
        controls.addSpacing(12)
        controls.addWidget(self._btn_auto)
        controls.addWidget(self._speed_slider)
        controls.addWidget(self._speed_label)
        controls.addStretch()

        layout.addLayout(controls)

        # Frame info row ────────────────────────────────────────────────────
        info = QHBoxLayout()
        info.setSpacing(8)

        self._frame_info = QLabel()
        self._frame_info.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 12px;"
        )
        self._filename_label = QLabel()
        self._filename_label.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 12px;"
        )

        info.addWidget(self._frame_info)
        info.addWidget(self._filename_label)
        info.addStretch()
        layout.addLayout(info)

        # Reject checkbox ───────────────────────────────────────────────────
        self._reject_cb = QCheckBox("Reject this frame")
        self._reject_cb.stateChanged.connect(self._on_reject_toggled)
        layout.addWidget(self._reject_cb)

        # Stats bar ─────────────────────────────────────────────────────────
        self._stats_label = QLabel()
        self._stats_label.setStyleSheet(
            "font-family: 'JetBrains Mono', monospace; font-size: 11px;"
        )
        layout.addWidget(self._stats_label)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Frame loading ─────────────────────────────────────────────────────

    def _load_pixmap(self, path: str) -> QPixmap | None:
        try:
            img_data = load_image(path)
            display = img_data.to_display(stretch=True)
            return self._ndarray_to_pixmap(display)
        except Exception as exc:
            log.warning("Failed to load %s: %s", path, exc)
            return None

    @staticmethod
    def _ndarray_to_pixmap(arr: np.ndarray) -> QPixmap:
        h, w, ch = arr.shape
        fmt = QImage.Format.Format_RGB888 if ch >= 3 else QImage.Format.Format_Grayscale8
        if ch > 3:
            arr = arr[:, :, :3]
            ch = 3
        img = QImage(arr.data, w, h, w * ch, fmt)
        return QPixmap.fromImage(img.copy())

    def _show_frame(self):
        total = len(self._frame_paths)
        if total == 0:
            self._image_label.clear()
            return

        idx = self._current_idx
        path = self._frame_paths[idx]
        fname = Path(path).name

        self._frame_info.setText(f"Frame: {idx + 1}/{total}")
        self._filename_label.setText(fname)

        self._reject_cb.blockSignals(True)
        self._reject_cb.setChecked(path in self._rejected)
        self._reject_cb.blockSignals(False)

        pixmap = self._load_pixmap(path)
        if pixmap is not None and not self._image_label.size().isNull():
            scaled = pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._image_label.setPixmap(scaled)
        else:
            self._image_label.clear()

        self._update_stats()
        self._update_buttons()

    # ── Navigation ────────────────────────────────────────────────────────

    def _prev(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._show_frame()

    def _next(self):
        if self._current_idx < len(self._frame_paths) - 1:
            self._current_idx += 1
            self._show_frame()
        elif self._auto_active:
            self._stop_auto()

    def _advance(self):
        if self._current_idx < len(self._frame_paths) - 1:
            self._current_idx += 1
            self._show_frame()
        else:
            self._stop_auto()

    # ── Auto-blink ────────────────────────────────────────────────────────

    def _toggle_auto(self, checked: bool):
        if checked:
            self._start_auto()
        else:
            self._stop_auto()

    def _start_auto(self):
        self._auto_active = True
        self._btn_auto.setText("⏸  Pause")
        interval = int(1000.0 / self._current_fps())
        self._auto_timer.start(interval)

    def _stop_auto(self):
        self._auto_active = False
        self._auto_timer.stop()
        self._btn_auto.setText("▶▶  Auto")
        self._btn_auto.setChecked(False)

    def _current_fps(self) -> float:
        val: int = self._speed_slider.value()
        return val / 10.0

    def _on_speed_changed(self):
        fps = self._current_fps()
        self._speed_label.setText(f"{fps:.1f} fps")
        if self._auto_active:
            self._auto_timer.setInterval(int(1000.0 / fps))

    # ── Rejection ─────────────────────────────────────────────────────────

    def _on_reject_toggled(self, state: int):
        path = self._frame_paths[self._current_idx]
        if state == Qt.CheckState.Checked.value:
            self._rejected.add(path)
        else:
            self._rejected.discard(path)
        self._update_stats()

    # ── Stats ─────────────────────────────────────────────────────────────

    def _update_stats(self):
        total = len(self._frame_paths)
        rejected = len(self._rejected)
        accepted = total - rejected
        self._stats_label.setText(f"Accepted: {accepted}  |  Rejected: {rejected}")

    def _update_buttons(self):
        total = len(self._frame_paths)
        self._btn_prev.setEnabled(self._current_idx > 0 and total > 0)
        self._btn_next.setEnabled(self._current_idx < total - 1 and total > 0)
        self._btn_auto.setEnabled(total > 1)

    # ── Events ────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent | None):  # noqa: N802
        if event is None:
            return
        if event.key() == Qt.Key.Key_Left:
            self._prev()
        elif event.key() == Qt.Key.Key_Right:
            self._next()
        elif event.key() == Qt.Key.Key_Space:
            self._reject_cb.setChecked(not self._reject_cb.isChecked())
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._reject_cb.setChecked(True)
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent | None):  # noqa: N802
        if event is None:
            return
        delta = event.angleDelta().y()
        if delta > 0:
            self._prev()
        elif delta < 0:
            self._next()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._show_frame()

    def closeEvent(self, event):  # noqa: N802
        self._auto_timer.stop()
        self.rejected.emit(sorted(self._rejected))
        super().closeEvent(event)

    # ── Convenience ───────────────────────────────────────────────────────

    @classmethod
    def blink_frames(cls, frame_paths: list[str], parent=None) -> list[str]:
        dialog = cls(frame_paths, parent)
        dialog.exec()
        return sorted(dialog._rejected)
