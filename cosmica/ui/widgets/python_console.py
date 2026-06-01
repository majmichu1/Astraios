"""Embedded Python REPL console for interactive scripting."""

from __future__ import annotations

import io
import sys
import traceback

from PyQt6.QtCore import QRegularExpression, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QKeyEvent,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

_BANNER = (
    "Cosmica Python Console  —  {v}\n"
    "  image      — current ImageData  (set when you open the console)\n"
    "  data        — image.data ndarray (H,W) or (C,H,W) float32 [0,1]\n"
    "  apply(arr)  — push ndarray back as current image\n"
    "  show(arr)   — display array on canvas without replacing current image\n"
    "  np, cv2, torch, fits, plt — pre-imported for convenience\n"
    "  cosmica     — full cosmica package\n"
    "  Shift+Enter — multi-line input  |  Up/Down — history  |  Ctrl+L — clear\n"
).format(v=sys.version.split()[0])

_KEYWORDS = (
    "False None True and as assert async await break class continue def del "
    "elif else except finally for from global if import in is lambda nonlocal "
    "not or pass raise return try while with yield"
).split()

_BUILTINS = "abs all any bin bool bytes callable chr dict dir divmod enumerate eval "  \
            "exec filter float format frozenset getattr hasattr hash help hex id "  \
            "input int isinstance issubclass iter len list map max min next object "  \
            "open ord pow print range repr reversed round set setattr slice sorted "  \
            "staticmethod str sum super tuple type vars zip".split()


class _PySyntaxHighlighter(QSyntaxHighlighter):
    """Minimal Python syntax highlighter for the input box."""

    def __init__(self, doc):
        super().__init__(doc)
        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#c678dd"))
        kw_fmt.setFontWeight(700)

        bi_fmt = QTextCharFormat()
        bi_fmt.setForeground(QColor("#56b6c2"))

        str_fmt = QTextCharFormat()
        str_fmt.setForeground(QColor("#98c379"))

        cmt_fmt = QTextCharFormat()
        cmt_fmt.setForeground(QColor("#5c6370"))
        cmt_fmt.setFontItalic(True)

        num_fmt = QTextCharFormat()
        num_fmt.setForeground(QColor("#d19a66"))

        self._rules = []
        for kw in _KEYWORDS:
            self._rules.append((QRegularExpression(rf"\b{kw}\b"), kw_fmt))
        for bi in _BUILTINS:
            self._rules.append((QRegularExpression(rf"\b{bi}\b"), bi_fmt))
        # Strings
        self._rules.append((QRegularExpression(r"\"[^\"\\]*(\\.[^\"\\]*)*\""), str_fmt))
        self._rules.append((QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), str_fmt))
        # Numbers
        self._rules.append((QRegularExpression(r"\b\d+(\.\d+)?([eE][+-]?\d+)?\b"), num_fmt))
        # Comments
        self._rules.append((QRegularExpression(r"#[^\n]*"), cmt_fmt))

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)


class _HistoryLineEdit(QPlainTextEdit):
    """Multi-line input with up/down history navigation and syntax highlighting."""

    execute_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(100)
        self.setPlaceholderText("Enter Python code — Shift+Enter for multi-line, Enter to run, Ctrl+L to clear")
        self._history: list[str] = []
        self._hist_idx = -1
        font = QFont("Monospace", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)
        self._highlighter = _PySyntaxHighlighter(self.document())

    def keyPressEvent(self, event: QKeyEvent):
        mods = event.modifiers()
        key = event.key()

        # Ctrl+L — clear output (send to console widget)
        if key == Qt.Key.Key_L and (mods & Qt.KeyboardModifier.ControlModifier):
            self.execute_requested.emit("__clear_output__")
            return

        # Enter without Shift — execute
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (mods & Qt.KeyboardModifier.ShiftModifier):
            text = self.toPlainText().strip()
            if text:
                self._history.append(text)
                self._hist_idx = -1
                self.execute_requested.emit(text)
                self.clear()
            return

        # Up — history back
        if key == Qt.Key.Key_Up and not self.toPlainText():
            if self._history:
                self._hist_idx = min(self._hist_idx + 1, len(self._history) - 1)
                self.setPlainText(self._history[-(self._hist_idx + 1)])
                self.moveCursor(QTextCursor.MoveOperation.End)
            return

        # Down — history forward
        if key == Qt.Key.Key_Down:
            if self._hist_idx > 0:
                self._hist_idx -= 1
                self.setPlainText(self._history[-(self._hist_idx + 1)])
                self.moveCursor(QTextCursor.MoveOperation.End)
            elif self._hist_idx == 0:
                self._hist_idx = -1
                self.clear()
            return

        super().keyPressEvent(event)


class _VarInspector(QTableWidget):
    """Simple variable inspector showing current namespace."""

    def __init__(self, parent=None):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Name", "Type", "Value"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        font = QFont("Monospace", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(font)

    def refresh(self, namespace: dict):
        skip = {"__name__", "__builtins__", "__doc__", "__package__", "__loader__", "__spec__"}
        items = sorted(
            [(k, v) for k, v in namespace.items() if not k.startswith("_") and k not in skip],
            key=lambda x: x[0],
        )
        self.setRowCount(len(items))
        for row, (name, val) in enumerate(items):
            type_str = type(val).__name__
            try:
                import numpy as np
                if isinstance(val, np.ndarray):
                    val_str = f"ndarray shape={val.shape} dtype={val.dtype}"
                elif hasattr(val, "__len__") and not isinstance(val, str) and len(val) > 8:
                    val_str = f"{type_str}[{len(val)}]"
                else:
                    val_str = repr(val)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "…"
            except Exception:
                val_str = "<error>"
            self.setItem(row, 0, QTableWidgetItem(name))
            self.setItem(row, 1, QTableWidgetItem(type_str))
            self.setItem(row, 2, QTableWidgetItem(val_str))


class PythonConsoleWidget(QWidget):
    """Embedded Python console with syntax highlighting and variable inspector."""

    #: Emitted when apply(arr) is called — main window connects this
    image_updated = pyqtSignal(object)  # np.ndarray
    #: Emitted when show(arr) is called — shows without replacing
    image_preview = pyqtSignal(object)  # np.ndarray

    def __init__(self, parent=None):
        super().__init__(parent)
        self._namespace: dict = {}
        self._setup_namespace()
        self._build_ui()
        self._write(_BANNER, color="#888888")
        self._write(">>> ", color="#61afef")

    def _setup_namespace(self):
        console_ref = self

        def apply(arr):
            """Push an ndarray back as the current image."""
            import numpy as _np
            arr = _np.asarray(arr, dtype=_np.float32)
            console_ref.image_updated.emit(arr)
            console_ref._write("Image updated.\n", color="#98c379")
            console_ref._inspector.refresh(console_ref._namespace)

        def show(arr):
            """Display array on canvas without replacing current image."""
            import numpy as _np
            arr = _np.asarray(arr, dtype=_np.float32)
            console_ref.image_preview.emit(arr)
            console_ref._write("Showing preview.\n", color="#56b6c2")

        import numpy as np
        try:
            import cv2
        except ImportError:
            cv2 = None
        try:
            import torch as _torch
        except ImportError:
            _torch = None
        try:
            from astropy.io import fits
        except ImportError:
            fits = None
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            plt = None

        import cosmica

        self._namespace = {
            "__name__": "__console__",
            "np": np,
            "cv2": cv2,
            "torch": _torch,
            "fits": fits,
            "plt": plt,
            "cosmica": cosmica,
            "image": None,
            "data": None,
            "_writable_data": None,
            "apply": apply,
            "show": show,
        }

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Tab widget: Output | Variables
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.South)

        # Output area
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        font = QFont("Monospace", 10)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._output.setFont(font)
        self._output.setStyleSheet(
            "background: #1a1a2e; color: #e0e0e0; border: 1px solid #444;"
        )
        tabs.addTab(self._output, "Output")

        # Variable inspector
        self._inspector = _VarInspector()
        tabs.addTab(self._inspector, "Variables")

        splitter.addWidget(tabs)

        # Input area
        input_widget = QWidget()
        input_layout = QVBoxLayout(input_widget)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(2)

        self._input = _HistoryLineEdit()
        self._input.setStyleSheet(
            "background: #16213e; color: #e0e0e0; border: 1px solid #555;"
        )
        self._input.execute_requested.connect(self._execute)
        input_layout.addWidget(self._input)

        btn_row = QHBoxLayout()
        btn_run = QPushButton("Run (Enter)")
        btn_run.clicked.connect(lambda: self._execute(self._input.toPlainText().strip()))
        btn_row.addWidget(btn_run)

        btn_clear = QPushButton("Clear Output")
        btn_clear.clicked.connect(self._clear_output)
        btn_row.addWidget(btn_clear)

        btn_load = QPushButton("Load Script…")
        btn_load.clicked.connect(self._load_script)
        btn_row.addWidget(btn_load)

        btn_save = QPushButton("Save Script…")
        btn_save.clicked.connect(self._save_script)
        btn_row.addWidget(btn_save)

        btn_row.addStretch()
        input_layout.addLayout(btn_row)

        splitter.addWidget(input_widget)
        splitter.setSizes([300, 120])
        layout.addWidget(splitter)

    def set_image(self, image_data):
        """Update the 'image' and 'data' variables in the console namespace."""
        self._namespace["image"] = image_data
        if image_data is not None:
            data_copy = image_data.data.copy()
            data_copy.flags.writeable = False
            self._namespace["data"] = data_copy
            self._namespace["_writable_data"] = image_data.data
        else:
            self._namespace["data"] = None
            self._namespace["_writable_data"] = None
        self._inspector.refresh(self._namespace)

    def _execute(self, source: str):
        if not source.strip():
            return

        # Special internal command
        if source == "__clear_output__":
            self._clear_output()
            return

        self._write(f">>> {source}\n", color="#61afef")

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = buf = io.StringIO()

        try:
            try:
                result = eval(compile(source, "<console>", "eval"), self._namespace)
                output = buf.getvalue()
                if output:
                    self._write(output)
                if result is not None:
                    self._write(repr(result) + "\n", color="#e5c07b")
            except SyntaxError:
                exec(compile(source, "<console>", "exec"), self._namespace)
                output = buf.getvalue()
                if output:
                    self._write(output)
        except Exception:
            output = buf.getvalue()
            if output:
                self._write(output)
            self._write(traceback.format_exc(), color="#e06c75")
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        self._inspector.refresh(self._namespace)
        self._output.moveCursor(QTextCursor.MoveOperation.End)

    def _write(self, text: str, color: str | None = None):
        self._output.moveCursor(QTextCursor.MoveOperation.End)
        if color:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            cursor = self._output.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(text, fmt)
        else:
            self._output.insertPlainText(text)
        self._output.moveCursor(QTextCursor.MoveOperation.End)

    def _clear_output(self):
        self._output.clear()
        self._write(_BANNER, color="#888888")

    def _load_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Python Script", "", "Python (*.py);;All Files (*)"
        )
        if path:
            try:
                with open(path, "r") as f:
                    code = f.read()
                self._input.setPlainText(code)
            except Exception as exc:
                self._write(f"Could not load {path}: {exc}\n", color="#e06c75")

    def _save_script(self):
        text = self._input.toPlainText()
        if not text.strip():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Python Script", "", "Python (*.py);;All Files (*)"
        )
        if path:
            try:
                with open(path, "w") as f:
                    f.write(text)
                self._write(f"Script saved to {path}\n", color="#98c379")
            except Exception as exc:
                self._write(f"Could not save: {exc}\n", color="#e06c75")
