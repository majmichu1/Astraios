"""Processing Log Panel — operation log, with progress and Cancel while a job runs."""

from __future__ import annotations

import logging
from datetime import datetime

from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from astraios.ui.theme import (
    ACCENT,
    BG_SECONDARY,
    BORDER,
    ORANGE,
    RED,
    TEXT_DIM,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

_MONO = "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace"


class LogPanel(QWidget):
    """Bottom panel: header, an on-demand progress row, and the log."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("LogPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"#LogPanel {{ background: {BG_SECONDARY}; border-top: 1px solid {BORDER}; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header: title · GPU status · Clear ───────────────────────────
        header = QWidget()
        header.setObjectName("LogHeader")
        header.setStyleSheet(
            f"#LogHeader {{ background: transparent; border-bottom: 1px solid {BORDER}; }}"
        )
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(10, 3, 10, 3)
        header_row.setSpacing(6)
        _title = QLabel("PROCESSING LOG")
        _title.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; font-weight: 600;"
            " letter-spacing: 0.6px; background: transparent;"
        )
        self._log_header_gpu = QLabel("")
        self._log_header_gpu.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; font-family: {_MONO}; background: transparent;"
        )
        _clear_btn = QPushButton("Clear")
        _clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        _clear_btn.setStyleSheet(
            f"QPushButton {{ color: {TEXT_PRIMARY}; font-size: 9px; padding: 1px 6px;"
            f" background: transparent; border: 1px solid {BORDER}; border-radius: 6px; }}"
            f" QPushButton:hover {{ background: {BORDER}; }}"
        )
        _clear_btn.clicked.connect(self.clear_log)
        header_row.addWidget(_title)
        header_row.addStretch()
        header_row.addWidget(self._log_header_gpu)
        header_row.addWidget(_clear_btn)
        layout.addWidget(header)

        # ── Progress row: only while something runs ──────────────────────
        # At rest this was a full-width empty bar under a "Ready" label,
        # duplicating the toolbar's status and costing a row of log lines.
        # It now appears with the job and carries the Cancel button.
        self._progress_row = QWidget()
        self._progress_row.setStyleSheet("background: transparent;")
        progress_row = QHBoxLayout(self._progress_row)
        progress_row.setContentsMargins(10, 4, 10, 2)
        progress_row.setSpacing(10)
        self._progress_label = QLabel("Ready")
        self._progress_label.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; background: transparent;"
        )
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(4)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_btn.setStyleSheet(
            f"QPushButton {{ color: {RED}; font-size: 10px; font-weight: 600; padding: 2px 10px;"
            f" background: transparent; border: 1px solid {RED}; border-radius: 6px; }}"
            " QPushButton:hover { background: #3a1a1a; }"
        )
        self._cancel_btn.clicked.connect(self.cancel_requested)
        self._cancel_btn.setVisible(False)
        progress_row.addWidget(self._progress_label, 1)
        progress_row.addWidget(self._progress_bar, 3)
        progress_row.addWidget(self._cancel_btn)
        self._progress_row.setVisible(False)
        layout.addWidget(self._progress_row)

        # ── Log text ─────────────────────────────────────────────────────
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFrameShape(QTextEdit.Shape.NoFrame)
        self._log_text.setStyleSheet(
            f"QTextEdit {{ background: {BG_SECONDARY}; border: none; border-radius: 0;"
            f" padding: 2px 4px; font-size: 10px; font-family: {_MONO}; }}"
        )
        layout.addWidget(self._log_text, 1)

    @pyqtSlot(float, str)
    def update_progress(self, fraction: float, message: str):
        if self._progress_bar.maximum() == 0:
            self._progress_bar.setRange(0, 1000)  # leave busy/indeterminate mode
        self._progress_bar.setValue(int(fraction * 1000))
        self._progress_label.setText(message)
        self._progress_row.setVisible(True)

    def set_busy(self, busy: bool, message: str = ""):
        """Indeterminate (marquee) progress for operations that report no fraction
        (e.g. loading a large file) so the app doesn't look frozen."""
        if busy:
            self._progress_bar.setRange(0, 0)
            if message:
                self._progress_label.setText(message)
            self._progress_row.setVisible(True)
        else:
            self._progress_bar.setRange(0, 1000)
            self._progress_bar.setValue(0)
            if not self._cancel_btn.isVisible():
                self._progress_row.setVisible(False)

    @pyqtSlot(str, str)
    def log(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "info": TEXT_SECONDARY,
            "warning": ORANGE,
            "error": RED,
            "success": ACCENT,
        }
        icons = {
            "info": "",
            "warning": "⚠ ",
            "error": "✕ ",
            "success": "✓ ",
        }
        color = colors.get(level, TEXT_SECONDARY)
        icon = icons.get(level, "")
        safe = (
            message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        self._log_text.append(
            f'<span style="color:{TEXT_DIM}">{timestamp}</span>&nbsp;&nbsp;'
            f'<span style="color:{color}">{icon}{safe}</span>'
        )
        self._log_text.verticalScrollBar().setValue(
            self._log_text.verticalScrollBar().maximum()
        )

    def clear_log(self):
        self._log_text.clear()

    def update_gpu_status(self, text: str):
        """Update the live GPU/VRAM label in the log panel header."""
        self._log_header_gpu.setText(text)

    def reset_progress(self):
        self._progress_bar.setRange(0, 1000)  # leave busy mode if it was set
        self._progress_bar.setValue(0)
        self._progress_label.setText("Ready")
        self._cancel_btn.setVisible(False)
        self._progress_row.setVisible(False)

    def set_cancel_visible(self, visible: bool):
        self._cancel_btn.setVisible(visible)
        if visible:
            self._progress_row.setVisible(True)


class _LogSignalBridge(QObject):
    """Thread-safe bridge: emits a Qt signal so log calls from worker threads
    are always delivered to the GUI thread via the event queue."""

    message = pyqtSignal(str, str)  # (text, level)


class QtLogHandler(logging.Handler):
    """Route Python logging to the LogPanel — thread-safe via Qt queued signals."""

    def __init__(self, log_panel: LogPanel):
        super().__init__()
        self._panel = log_panel
        self._bridge = _LogSignalBridge()
        # QueuedConnection: signal emitted from any thread, slot always runs in GUI thread
        self._bridge.message.connect(self._panel.log, Qt.ConnectionType.QueuedConnection)

    def emit(self, record: logging.LogRecord):
        level_map = {
            logging.DEBUG: "info",
            logging.INFO: "info",
            logging.WARNING: "warning",
            logging.ERROR: "error",
            logging.CRITICAL: "error",
        }
        level = level_map.get(record.levelno, "info")
        # Emit the signal — safe from any thread; Qt queues it for the GUI thread
        self._bridge.message.emit(record.getMessage(), level)
