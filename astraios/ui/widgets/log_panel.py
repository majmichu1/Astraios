"""Processing Log Panel — displays operation logs and progress."""

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


class LogPanel(QWidget):
    """Bottom panel showing processing log and progress bar."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header row: title + live GPU status + clear button
        header_row = QHBoxLayout()
        header_row.setContentsMargins(8, 4, 8, 2)
        _title = QLabel("Processing Log")
        _title.setStyleSheet("color: #e6edf3; font-size: 11px; font-weight: 700;")
        self._log_header_gpu = QLabel("")
        self._log_header_gpu.setStyleSheet("color: #8b949e; font-size: 11px;")
        _clear_btn = QPushButton("Clear")
        _clear_btn.setMaximumWidth(52)
        _clear_btn.setMaximumHeight(18)
        _clear_btn.setStyleSheet(
            "QPushButton { color: #8b949e; font-size: 10px; border: 1px solid #30363d;"
            " border-radius: 3px; padding: 0 4px; background: transparent; }"
            " QPushButton:hover { color: #e6edf3; border-color: #8b949e; }"
        )
        _clear_btn.clicked.connect(self.clear_log)
        header_row.addWidget(_title)
        header_row.addStretch()
        header_row.addWidget(self._log_header_gpu)
        header_row.addWidget(_clear_btn)
        layout.addLayout(header_row)

        # Progress bar row
        progress_row = QHBoxLayout()
        self._progress_label = QLabel("Ready")
        self._progress_label.setStyleSheet("color: #969696; font-size: 12px;")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setMaximumHeight(16)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setMaximumWidth(70)
        self._cancel_btn.setMaximumHeight(18)
        self._cancel_btn.setStyleSheet(
            "QPushButton { color: #e06c75; font-size: 11px; border: 1px solid #e06c75;"
            " border-radius: 3px; padding: 0 4px; }"
            " QPushButton:hover { background: #3a1a1a; }"
        )
        self._cancel_btn.clicked.connect(self.cancel_requested)
        self._cancel_btn.setVisible(False)
        progress_row.addWidget(self._progress_label, 1)
        progress_row.addWidget(self._progress_bar, 2)
        progress_row.addWidget(self._cancel_btn)
        layout.addLayout(progress_row)

        # Log text
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(120)
        self._log_text.setStyleSheet(
            "QTextEdit { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 11px; }"
        )
        layout.addWidget(self._log_text)

    @pyqtSlot(float, str)
    def update_progress(self, fraction: float, message: str):
        self._progress_bar.setValue(int(fraction * 1000))
        self._progress_label.setText(message)

    @pyqtSlot(str, str)
    def log(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "info": "#d4d4d4",
            "warning": "#e5c07b",
            "error": "#e06c75",
            "success": "#98c379",
        }
        icons = {
            "info": "·",
            "warning": "⚠",
            "error": "✕",
            "success": "✓",
        }
        color = colors.get(level, "#d4d4d4")
        icon = icons.get(level, "·")
        self._log_text.append(
            f'<span style="color:#6e6e6e">[{timestamp}]</span> '
            f'<span style="color:{color}">{icon} {message}</span>'
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
        self._progress_bar.setValue(0)
        self._progress_label.setText("Ready")
        self._cancel_btn.setVisible(False)

    def set_cancel_visible(self, visible: bool):
        self._cancel_btn.setVisible(visible)


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
