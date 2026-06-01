"""Live Stack Dialog — real-time frame accumulation with live preview."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QDir, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from cosmica.core.image_io import load_image
from cosmica.core.live_stack import LiveStacker

log = logging.getLogger(__name__)


class _FrameLoader(QThread):
    """Background loader that pushes frames one at a time."""

    frame_ready = pyqtSignal(object)
    finished_count = pyqtSignal(int, int)
    error_loading = pyqtSignal(str)

    def __init__(self, paths: list[str], parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._cancelled = False

    def run(self):
        loaded = 0
        skipped = 0
        for p in self._paths:
            if self._cancelled:
                break
            try:
                img = load_image(p)
                self.frame_ready.emit(img.data)
                loaded += 1
            except Exception as exc:
                log.debug("Skipped %s: %s", Path(p).name, exc)
                skipped += 1
        self.finished_count.emit(loaded, skipped)

    def cancel(self):
        self._cancelled = True


class LiveStackDialog(QDialog):
    """Dialog for live stacking frames from a folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stacker = LiveStacker()
        self._running = False
        self._loader: _FrameLoader | None = None
        self._loaded_count = 0
        self._total_count = 0
        self._frame_paths: list[str] = []
        self._t0 = 0.0

        self._setup_ui()

        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._refresh_preview)

    # ── UI Setup ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        self.setWindowTitle("Live Stack")
        self.resize(960, 700)
        self.setMinimumSize(640, 480)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Left: Controls ──────────────────────────────────────────────
        ctrl = QVBoxLayout()
        ctrl.setSpacing(8)

        self._btn_folder = QPushButton("📂 Load Folder…")
        self._btn_folder.clicked.connect(self._load_folder)
        ctrl.addWidget(self._btn_folder)

        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #30363d;")
        ctrl.addWidget(sep)

        mode_label = QLabel("Alignment")
        mode_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        ctrl.addWidget(mode_label)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["FFT Translation", "No Alignment"])
        self._mode_combo.setStyleSheet(self._combo_ss())
        ctrl.addWidget(self._mode_combo)

        ctrl.addSpacing(8)

        self._btn_start = QPushButton("▶  Start Stacking")
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._toggle_stacking)
        self._btn_start.setStyleSheet(self._accent_btn_ss())
        ctrl.addWidget(self._btn_start)

        self._btn_save = QPushButton("💾 Save Result…")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._save_result)
        ctrl.addWidget(self._btn_save)

        self._btn_reset = QPushButton("🗑 Reset")
        self._btn_reset.setEnabled(False)
        self._btn_reset.clicked.connect(self._reset_stacker)
        ctrl.addWidget(self._btn_reset)

        ctrl.addSpacing(16)

        self._status_label = QLabel("No frames loaded")
        self._status_label.setStyleSheet(
            "color: #8b949e; font-size: 12px; font-family: 'JetBrains Mono', monospace;"
        )
        ctrl.addWidget(self._status_label)

        self._elapsed_label = QLabel("Elapsed: 0s")
        self._elapsed_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-family: monospace;"
        )
        ctrl.addWidget(self._elapsed_label)

        ctrl.addStretch()

        ctrl_widget = QWidget()
        ctrl_widget.setFixedWidth(200)
        ctrl_widget.setLayout(ctrl)
        layout.addWidget(ctrl_widget)

        # ── Right: Preview ──────────────────────────────────────────────
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(400, 300)
        self._preview_label.setStyleSheet(
            "background-color: #000; border: 1px solid #30363d; border-radius: 4px;"
        )
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._preview_label, 1)

    # ── Styles ───────────────────────────────────────────────────────────

    @staticmethod
    def _combo_ss() -> str:
        return (
            "QComboBox { background: #21262d; color: #e6edf3; border: 1px solid #30363d;"
            " border-radius: 4px; padding: 4px 8px; font-size: 11px; }"
            " QComboBox::drop-down { border: none; }"
            " QComboBox QAbstractItemView { background: #21262d; color: #e6edf3;"
            " selection-background-color: #2ea043; }"
        )

    @staticmethod
    def _accent_btn_ss() -> str:
        return (
            "QPushButton { background: #2ea043; color: #fff; border: none;"
            " border-radius: 4px; padding: 8px 16px; font-size: 12px; font-weight: bold; }"
            " QPushButton:hover { background: #3fb950; }"
            " QPushButton:disabled { background: #21262d; color: #484f58; }"
        )

    # ── Actions ──────────────────────────────────────────────────────────

    def _load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with Light Frames", QDir.homePath(),
        )
        if not folder:
            return
        supported = (".fit", ".fits", ".fts", ".xisf", ".tif", ".tiff", ".png")
        paths = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in supported and not p.name.startswith(".")
        )
        if not paths:
            QMessageBox.information(self, "No Frames", "No supported image files found.")
            return

        self._frame_paths = paths
        self._total_count = len(paths)
        self._loaded_count = 0
        self._btn_start.setEnabled(True)
        self._status_label.setText(f"{len(paths)} frames loaded")
        log.info("Live stack: loaded %d frames from %s", len(paths), folder)

    def _toggle_stacking(self):
        if self._running:
            self._stop_stacking()
        else:
            self._start_stacking()

    def _start_stacking(self):
        if not self._frame_paths:
            return

        self._stacker.reset()
        mode = self._mode_combo.currentText()
        self._stacker.alignment_mode = "fft" if mode == "FFT Translation" else "none"

        self._running = True
        self._loaded_count = 0
        self._t0 = time.monotonic()

        self._btn_start.setText("⏹  Stop")
        self._btn_start.setStyleSheet(
            "QPushButton { background: #da3633; color: #fff; border: none;"
            " border-radius: 4px; padding: 8px 16px; font-size: 12px; font-weight: bold; }"
            " QPushButton:hover { background: #f85149; }"
        )
        self._btn_folder.setEnabled(False)
        self._mode_combo.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._btn_reset.setEnabled(False)

        self._preview_timer.start()

        self._loader = _FrameLoader(self._frame_paths, self)
        self._loader.frame_ready.connect(self._on_frame_ready)
        self._loader.finished_count.connect(self._on_loader_done)
        self._loader.error_loading.connect(lambda m: log.warning("Load error: %s", m))
        self._loader.start()

    def _stop_stacking(self):
        self._running = False
        self._preview_timer.stop()

        if self._loader and self._loader.isRunning():
            self._loader.cancel()
            self._loader.wait()

        self._btn_start.setText("▶  Start Stacking")
        self._btn_start.setStyleSheet(self._accent_btn_ss())
        self._btn_folder.setEnabled(True)
        self._mode_combo.setEnabled(True)
        self._btn_save.setEnabled(self._stacker.n_frames > 0)
        self._btn_reset.setEnabled(True)

        elapsed = time.monotonic() - self._t0
        log.info(
            "Live stack finished: %d frames in %.1fs",
            self._stacker.n_frames, elapsed,
        )

    @pyqtSlot(object)
    def _on_frame_ready(self, data: np.ndarray):
        self._stacker.add_frame(data)
        self._loaded_count += 1

    @pyqtSlot(int, int)
    def _on_loader_done(self, loaded: int, skipped: int):
        if self._running:
            self._stop_stacking()

    def _reset_stacker(self):
        self._stacker.reset()
        self._loaded_count = 0
        self._preview_label.clear()
        self._status_label.setText(
            f"{self._total_count} frames loaded"
            if self._frame_paths
            else "No frames loaded"
        )
        self._elapsed_label.setText("Elapsed: 0s")
        self._btn_save.setEnabled(False)
        self._btn_reset.setEnabled(False)
        self._btn_start.setEnabled(bool(self._frame_paths))

    def _refresh_preview(self):
        preview = self._stacker.get_live_preview()
        self._display_preview(preview)

        n = self._stacker.n_frames
        total = self._total_count
        self._status_label.setText(f"Stacked: {n} / {total} frames")

        elapsed = time.monotonic() - self._t0
        if elapsed < 60:
            self._elapsed_label.setText(f"Elapsed: {elapsed:.0f}s")
        else:
            self._elapsed_label.setText(f"Elapsed: {elapsed / 60:.1f} min")

    def _display_preview(self, arr: np.ndarray):
        if arr.size == 0 or arr.shape[-2:] == (100, 100) and arr.max() == 0:
            return

        h, w = arr.shape[-2], arr.shape[-1]
        if arr.ndim == 2:
            fmt = QImage.Format.Format_Grayscale8
            display = np.clip(arr * 255, 0, 255).astype(np.uint8)
            bytes_per_line = w
            img = QImage(display.data, w, h, bytes_per_line, fmt)
        else:
            fmt = QImage.Format.Format_RGB888
            ch = arr.shape[0]
            if ch >= 3:
                display = np.clip(arr[:3].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            else:
                display = np.clip(arr[0] * 255, 0, 255).astype(np.uint8)
                fmt = QImage.Format.Format_Grayscale8
            bytes_per_line = display.shape[1] * (3 if ch >= 3 else 1)
            img = QImage(display.data, display.shape[1], display.shape[0], bytes_per_line, fmt)

        pixmap = QPixmap.fromImage(img.copy())

        if pixmap and not self._preview_label.size().isNull():
            scaled = pixmap.scaled(
                self._preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._preview_label.setPixmap(scaled)

    def _save_result(self):
        result = self._stacker.get_result()
        if result is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Live Stack Result", "", "FITS (*.fits);;TIFF (*.tiff);;PNG (*.png)"
        )
        if not path:
            return

        from cosmica.core.image_io import ImageData, save_image
        img = ImageData(data=result, header={})
        try:
            save_image(img, path=path)
            log.info("Live stack result saved to %s", path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{exc}")

    # ── Events ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._loader and self._loader.isRunning():
            self._loader.cancel()
            self._loader.wait()
        self._preview_timer.stop()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._stacker.n_frames > 0:
            self._refresh_preview()
