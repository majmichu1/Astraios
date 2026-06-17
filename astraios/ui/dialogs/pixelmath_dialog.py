"""Pixel Math Dialog — evaluate mathematical expressions on image pixels.

Inspired by Siril's Pixel Math with syntax highlighting, expression history,
function reference, and per-channel application.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from astraios.core.pixel_math import (
    PixelMathError,
    evaluate,
    prepare_variables,
    validate_expression,
)

_ACCENT = "#58a6ff"
_ACCENT_DARK = "#1f3a5f"
_BG_PRIMARY = "#0d1117"
_BG_SECONDARY = "#161b22"
_BG_TERTIARY = "#21262d"
_TEXT_PRIMARY = "#e6edf3"
_TEXT_SECONDARY = "#8b949e"
_BORDER = "#30363d"
_SUCCESS = "#3fb950"
_ERROR = "#f85149"


class _ExpressionHighlighter(QSyntaxHighlighter):
    """Simple syntax highlighter for pixel math expressions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules = []

        fmt_fn = QTextCharFormat()
        fmt_fn.setForeground(QColor("#d2a8ff"))
        self._rules.append((
            r"\b(min|max|abs|sqrt|log|log10|exp|clip|normalize|"
            r"mean|median|sin|cos|pow|iif|mtf|round|floor|ceil)\b",
            fmt_fn,
        ))

        fmt_var = QTextCharFormat()
        fmt_var.setForeground(QColor("#7ee787"))
        self._rules.append((r"\b([A-Z][A-Za-z0-9_]*|[a-z][a-z0-9_]*)\b", fmt_var))

        fmt_num = QTextCharFormat()
        fmt_num.setForeground(QColor("#79c0ff"))
        self._rules.append((r"\b\d+\.?\d*\b", fmt_num))

        fmt_op = QTextCharFormat()
        fmt_op.setForeground(QColor("#ff7b72"))
        self._rules.append((r"[\+\-\*/\<\>\=\!\(\)\[\],]", fmt_op))

    def highlightBlock(self, text: str):
        import re
        for pattern, fmt in self._rules:
            for m in re.finditer(pattern, text):
                start = m.start()
                length = m.end() - m.start()
                self.setFormat(start, length, fmt)


class PixelMathDialog(QDialog):
    """Dialog for evaluating pixel math expressions on the current image."""

    result_ready = pyqtSignal(np.ndarray)

    def __init__(
        self,
        image: np.ndarray,
        parent=None,
        available_images: dict[str, np.ndarray] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Pixel Math")
        self.setMinimumSize(680, 560)
        self.resize(780, 640)

        self._image = image
        self._available_images = available_images or {}
        self._settings = QSettings("Astraios", "Astraios")

        self._build_ui()
        self._load_history()

    def _build_ui(self):
        self.setStyleSheet(f"""
            QDialog {{ background: {_BG_PRIMARY}; }}
            QGroupBox {{
                font-size: 12px; font-weight: 600; color: {_TEXT_SECONDARY};
                border: 1px solid {_BORDER}; border-radius: 6px;
                margin-top: 12px; padding: 12px 8px 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; subcontrol-position: top left;
                padding: 0 6px;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {_BORDER}; }}")

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        # ── Expression input ───────────────────────────────────────────
        expr_row = QHBoxLayout()
        expr_row.setSpacing(6)

        self._expr_input = QLineEdit()
        self._expr_input.setPlaceholderText(
            "e.g.  T * 2  or  clip(R - ref1, 0, 1)  or  sqrt(L)"
        )
        self._expr_input.setStyleSheet(f"""
            QLineEdit {{
                font-family: 'JetBrains Mono', 'Fira Code', monospace;
                font-size: 14px; padding: 8px 10px;
                background: {_BG_SECONDARY}; color: {_TEXT_PRIMARY};
                border: 1px solid {_BORDER}; border-radius: 6px;
            }}
            QLineEdit:focus {{ border-color: {_ACCENT}; }}
        """)
        self._expr_input.textChanged.connect(self._on_expression_changed)
        self._highlighter = _ExpressionHighlighter(self._expr_input)
        expr_row.addWidget(self._expr_input, 1)

        self._history_combo = QComboBox()
        self._history_combo.setMinimumWidth(200)
        self._history_combo.setStyleSheet(f"""
            QComboBox {{ {self._style_input()} }}
            QComboBox:focus {{ border-color: {_ACCENT}; }}
            QComboBox::drop-down {{ border: none; padding-right: 8px; }}
            QComboBox QAbstractItemView {{
                background: {_BG_TERTIARY}; color: {_TEXT_PRIMARY};
                selection-background-color: {_ACCENT_DARK};
                border: 1px solid {_BORDER};
                font-family: monospace; font-size: 11px;
            }}
        """)
        self._history_combo.setPlaceholderText("History…")
        self._history_combo.activated.connect(self._on_history_selected)
        expr_row.addWidget(self._history_combo)

        top_layout.addLayout(expr_row)

        # ── Validation + status ────────────────────────────────────────
        status_row = QHBoxLayout()
        self._validation_label = QLabel("Enter an expression above")
        self._validation_label.setStyleSheet(
            f"color: {_TEXT_SECONDARY}; font-size: 11px; padding-left: 4px;"
        )
        status_row.addWidget(self._validation_label, 1)
        top_layout.addLayout(status_row)

        # ── Options row ────────────────────────────────────────────────
        opts_row = QHBoxLayout()
        opts_row.setSpacing(12)

        opts_row.addWidget(QLabel("Apply to:"))
        self._channel_combo = QComboBox()
        channel_opts = ["All channels"]
        if self._image.ndim == 3:
            channel_opts += ["Red (R)", "Green (G)", "Blue (B)"]
        channel_opts += ["Luminance (L)", "Mono average"]
        self._channel_combo.addItems(channel_opts)
        self._channel_combo.setStyleSheet(f"QComboBox {{ {self._style_input()} }}")
        opts_row.addWidget(self._channel_combo)

        self._create_new_check = QCheckBox("Create new image")
        self._create_new_check.setStyleSheet(
            f"color: {_TEXT_PRIMARY}; font-size: 12px;"
        )
        opts_row.addWidget(self._create_new_check)
        opts_row.addStretch()

        top_layout.addLayout(opts_row)

        # ── Reference images ───────────────────────────────────────────
        if self._available_images:
            self._ref_group = QGroupBox("Reference Images")
            ref_layout = QVBoxLayout(self._ref_group)

            self._ref_table = QTableWidget(len(self._available_images), 3)
            self._ref_table.setHorizontalHeaderLabels(["Use", "Image", "Variable"])
            hh = self._ref_table.horizontalHeader()
            if hh is not None:
                hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self._ref_table.setStyleSheet(f"""
                QTableWidget {{ background: {_BG_TERTIARY}; color: {_TEXT_PRIMARY};
                    font-size: 11px; border: 1px solid {_BORDER};
                    gridline-color: {_BORDER}; }}
                QTableWidget::item {{ padding: 2px 4px; }}
                QHeaderView::section {{ background: {_BG_SECONDARY};
                    color: {_TEXT_SECONDARY}; border: 1px solid {_BORDER};
                    padding: 3px; }}
            """)
            self._ref_table.setMaximumHeight(140)
            vh = self._ref_table.verticalHeader()
            if vh is not None:
                vh.setDefaultSectionSize(22)
            vh.setVisible(False)

            for row, name in enumerate(self._available_images):
                cb = QCheckBox()
                cb.setStyleSheet("margin-left: 8px;")
                self._ref_table.setCellWidget(row, 0, cb)
                ni = QTableWidgetItem(name)
                ni.setFlags(ni.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._ref_table.setItem(row, 1, ni)
                ve = QLineEdit(f"ref{row + 1}")
                ve.setStyleSheet(
                    f"font-family: monospace; font-size: 11px; "
                    f"background: transparent; border: 1px solid {_BORDER}; "
                    f"color: {_TEXT_PRIMARY}; padding: 1px 4px;"
                )
                self._ref_table.setCellWidget(row, 2, ve)

            ref_layout.addWidget(self._ref_table)
            top_layout.addWidget(self._ref_group)

        splitter.addWidget(top)

        # ── Bottom: function reference + log ───────────────────────────
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        func_ref = QGroupBox("Functions Reference")
        func_lay = QVBoxLayout(func_ref)
        func_lay.setContentsMargins(8, 4, 8, 4)
        func_text = QLabel(
            "<b style='color:#d2a8ff;'>min</b>(a,b) "
            "<b style='color:#d2a8ff;'>max</b>(a,b) "
            "<b style='color:#d2a8ff;'>abs</b>(x) "
            "<b style='color:#d2a8ff;'>sqrt</b>(x) "
            "<b style='color:#d2a8ff;'>log</b>(x) "
            "<b style='color:#d2a8ff;'>exp</b>(x) "
            "<b style='color:#d2a8ff;'>clip</b>(x,lo,hi) "
            "<b style='color:#d2a8ff;'>normalize</b>(x) "
            "<b style='color:#d2a8ff;'>mean</b>(x) "
            "<b style='color:#d2a8ff;'>median</b>(x) "
            "<b style='color:#d2a8ff;'>pow</b>(x,n) "
            "<b style='color:#d2a8ff;'>iif</b>(cond,a,b) "
            "<b style='color:#d2a8ff;'>mtf</b>(x,f) "
            "<b style='color:#d2a8ff;'>sin</b>(x) <b style='color:#d2a8ff;'>cos</b>(x) "
            "<span style='color:{}'>Variables:</span> "
            "<span style='color:#7ee787;'>T R G B L</span>".format(_TEXT_SECONDARY)
        )
        func_text.setWordWrap(True)
        func_text.setTextFormat(Qt.TextFormat.RichText)
        func_text.setStyleSheet(f"color: {_TEXT_SECONDARY}; font-size: 11px;")
        func_lay.addWidget(func_text)
        bottom_layout.addWidget(func_ref)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        self._log.setStyleSheet(f"""
            QTextEdit {{
                font-family: 'JetBrains Mono', 'Fira Code', monospace;
                font-size: 11px; background: {_BG_SECONDARY};
                color: {_TEXT_PRIMARY}; border: 1px solid {_BORDER};
                border-radius: 4px; padding: 4px;
            }}
        """)
        bottom_layout.addWidget(self._log)

        splitter.addWidget(bottom)
        root.addWidget(splitter, 1)

        # ── Buttons ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._btn_apply = QPushButton("Apply")
        self._btn_apply.setEnabled(False)
        self._btn_apply.setStyleSheet(self._style_btn(_ACCENT))
        self._btn_apply.clicked.connect(self._evaluate)
        btn_row.addWidget(self._btn_apply)

        self._btn_validate = QPushButton("Validate")
        self._btn_validate.setStyleSheet(self._style_btn(_BG_TERTIARY))
        self._btn_validate.clicked.connect(self._validate_current)
        btn_row.addWidget(self._btn_validate)

        btn_row.addStretch()

        self._btn_close = QPushButton("Close")
        self._btn_close.setStyleSheet(self._style_btn("#454545"))
        self._btn_close.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_close)

        root.addLayout(btn_row)

    @staticmethod
    def _style_input() -> str:
        return (
            f"background: {_BG_SECONDARY}; color: {_TEXT_PRIMARY}; "
            f"border: 1px solid {_BORDER}; border-radius: 4px; "
            f"padding: 4px 8px; font-size: 12px;"
        )

    @staticmethod
    def _style_btn(color: str) -> str:
        return f"""
            QPushButton {{
                background: {color}; color: white; font-size: 12px;
                font-weight: 600; padding: 6px 20px; border: none;
                border-radius: 4px;
            }}
            QPushButton:hover {{ opacity: 0.8; }}
            QPushButton:disabled {{ background: {_BG_TERTIARY}; color: {_TEXT_SECONDARY}; }}
        """

    # ── Expression handling ────────────────────────────────────────────

    def _on_expression_changed(self, text: str):
        if not text.strip():
            self._validation_label.setText("Enter an expression above")
            self._validation_label.setStyleSheet(
                f"color: {_TEXT_SECONDARY}; font-size: 11px; padding-left: 4px;"
            )
            self._btn_apply.setEnabled(False)
            return
        error = validate_expression(text)
        if error is None:
            self._validation_label.setText("✓ Valid expression")
            self._validation_label.setStyleSheet(
                f"color: {_SUCCESS}; font-size: 11px; padding-left: 4px;"
            )
            self._btn_apply.setEnabled(True)
        else:
            self._validation_label.setText(f"✗ {error}")
            self._validation_label.setStyleSheet(
                f"color: {_ERROR}; font-size: 11px; padding-left: 4px;"
            )
            self._btn_apply.setEnabled(False)

    def _validate_current(self):
        expr = self._expr_input.text().strip()
        if not expr:
            return
        error = validate_expression(expr)
        if error is None:
            self._log.append(f"✓ <span style='color:{_SUCCESS};'>Valid:</span> {expr}")
        else:
            self._log.append(f"✗ <span style='color:{_ERROR};'>Error:</span> {error} — {expr}")

    def _on_history_selected(self, index: int):
        if index < 0:
            return
        text = self._history_combo.currentText()
        if text and text != self._expr_input.text():
            self._expr_input.setText(text)

    def _evaluate(self):
        expr = self._expr_input.text().strip()
        if not expr:
            return

        self._btn_apply.setEnabled(False)
        self._btn_apply.setText("Processing…")
        self._log.append(
            f"<span style='color:{_ACCENT};'>▸</span> {expr}"
        )

        self._save_to_history(expr)
        QTimer.singleShot(0, lambda: self._do_evaluate(expr))

    def _build_variables(self) -> dict[str, np.ndarray]:
        variables = prepare_variables(self._image)
        if self._available_images and hasattr(self, "_ref_table"):
            for row in range(self._ref_table.rowCount()):
                cb = self._ref_table.cellWidget(row, 0)
                if not isinstance(cb, QCheckBox) or not cb.isChecked():
                    continue
                ve = self._ref_table.cellWidget(row, 2)
                if not isinstance(ve, QLineEdit):
                    continue
                var_name = ve.text().strip()
                if not var_name:
                    continue
                ni = self._ref_table.item(row, 1)
                if ni is None:
                    continue
                img_name = ni.text()
                if img_name in self._available_images:
                    variables[var_name] = self._available_images[img_name]
        return variables

    def _apply_to_channels(self, data: np.ndarray,
                           result: np.ndarray) -> np.ndarray:
        """Return the result early if it already matches the target channel shape."""
        channel = self._channel_combo.currentText()
        if channel == "All channels":
            return result
        idx = {"Red (R)": 0, "Green (G)": 1, "Blue (B)": 2}.get(channel)
        if idx is not None and data.ndim == 3 and result.shape == data[idx].shape:
            out = data.copy()
            out[idx] = result
            return out
        return result

    def _do_evaluate(self, expr: str):
        try:
            variables = self._build_variables()
            result = evaluate(expr, variables)

            channel = self._channel_combo.currentText()
            if channel == "Luminance (L)":
                if self._image.ndim == 3:
                    from astraios.core.channels import extract_luminance
                    lum = extract_luminance(self._image)
                    variables["L"] = lum
                    result = evaluate(expr, variables)
            elif channel == "Mono average":
                if self._image.ndim == 3:
                    mono = self._image.mean(axis=0)
                    variables["T"] = mono
                    result = evaluate(expr, variables)

            if not self._create_new_check.isChecked():
                result = self._apply_to_channels(self._image, result)

            shape_str = "×".join(str(d) for d in result.shape)
            self._log.append(
                f"  Result: {shape_str}, "
                f"min={result.min():.4f}, max={result.max():.4f}, "
                f"mean={result.mean():.4f}"
            )
            self.result_ready.emit(result.astype(np.float32))
            self.accept()
        except PixelMathError as e:
            self._log.append(f"  <span style='color:{_ERROR};'>Error: {e}</span>")
            self._btn_apply.setEnabled(True)
            self._btn_apply.setText("Apply")
        except Exception as e:
            self._log.append(f"  <span style='color:{_ERROR};'>Unexpected: {e}</span>")
            self._btn_apply.setEnabled(True)
            self._btn_apply.setText("Apply")

    # ── History persistence ───────────────────────────────────────────

    def _save_to_history(self, expr: str):
        history = self._settings.value("pixelmath/history", [])
        if not isinstance(history, list):
            history = []
        if expr in history:
            history.remove(expr)
        history.insert(0, expr)
        history = history[:20]
        self._settings.setValue("pixelmath/history", history)
        self._refresh_history(history)

    def _load_history(self):
        history = self._settings.value("pixelmath/history", [])
        if not isinstance(history, list):
            history = []
        self._refresh_history(history)

    def _refresh_history(self, history: list[str]):
        self._history_combo.clear()
        for h in history:
            self._history_combo.addItem(h)
