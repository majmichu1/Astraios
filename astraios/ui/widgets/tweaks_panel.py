"""Tweaks Panel — floating frameless panel for live UI customization."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from astraios.ui.theme import (
    BG_SECONDARY,
    BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

_SWATCH_COLORS = [
    ("green",  "#2ea043"),
    ("blue",   "#388bfd"),
    ("purple", "#8957e5"),
    ("gold",   "#d29922"),
    ("red",    "#f85149"),
]


class TweaksPanel(QWidget):
    """Floating frameless settings panel."""

    accent_changed = pyqtSignal(str)        # color name
    workflow_visible = pyqtSignal(bool)
    log_visible = pyqtSignal(bool)
    log_height_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        super().__init__(parent, flags)
        self.setObjectName("TweaksPanel")
        self.setFixedWidth(240)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            #TweaksPanel {{
                background-color: {BG_SECONDARY};
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QLabel {{
                background-color: transparent;
                color: {TEXT_SECONDARY};
                font-size: 11px;
            }}
            QCheckBox {{
                color: {TEXT_PRIMARY};
                font-size: 12px;
                spacing: 6px;
            }}
        """)

        self._selected_accent = "green"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Title
        title = QLabel("UI Tweaks")
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600; background: transparent;"
        )
        layout.addWidget(title)

        # Accent color swatches
        accent_label = QLabel("ACCENT COLOR")
        layout.addWidget(accent_label)

        swatch_row = QHBoxLayout()
        swatch_row.setSpacing(6)
        self._swatch_btns: dict[str, QPushButton] = {}
        for name, color in _SWATCH_COLORS:
            btn = QPushButton()
            btn.setFixedSize(26, 26)
            btn.setToolTip(name.capitalize())
            btn.setStyleSheet(self._swatch_style(color, selected=(name == self._selected_accent)))
            btn.clicked.connect(lambda checked, n=name: self._on_accent(n))
            self._swatch_btns[name] = btn
            swatch_row.addWidget(btn)
        swatch_row.addStretch()
        layout.addLayout(swatch_row)

        # Separator
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {BORDER};")
        layout.addWidget(sep)

        # Workflow bar toggle
        self._workflow_cb = QCheckBox("Show workflow bar")
        self._workflow_cb.setChecked(True)
        self._workflow_cb.toggled.connect(self.workflow_visible)
        layout.addWidget(self._workflow_cb)

        # Log toggle
        self._log_cb = QCheckBox("Show processing log")
        self._log_cb.setChecked(True)
        self._log_cb.toggled.connect(self.log_visible)
        layout.addWidget(self._log_cb)

        # Separator
        sep2 = QWidget()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background-color: {BORDER};")
        layout.addWidget(sep2)

        # Log height slider
        log_h_label = QLabel("LOG HEIGHT")
        layout.addWidget(log_h_label)

        slider_row = QHBoxLayout()
        self._log_slider = QSlider(Qt.Orientation.Horizontal)
        self._log_slider.setRange(60, 240)
        self._log_slider.setValue(120)
        self._log_slider.setTickInterval(30)
        self._log_slider.valueChanged.connect(self._on_log_height)
        self._log_h_val = QLabel("120px")
        self._log_h_val.setFixedWidth(36)
        slider_row.addWidget(self._log_slider, 1)
        slider_row.addWidget(self._log_h_val)
        layout.addLayout(slider_row)

    def _swatch_style(self, color: str, selected: bool) -> str:
        border = "2px solid #ffffff" if selected else f"2px solid {BORDER}"
        return (
            f"QPushButton {{ background-color: {color}; border: {border}; border-radius: 13px; }}"
            f"QPushButton:hover {{ border: 2px solid {TEXT_PRIMARY}; }}"
        )

    def _on_accent(self, name: str):
        self._selected_accent = name
        swatch_colors = [c for _, c in _SWATCH_COLORS]
        for color_name, color in zip(self._swatch_btns.keys(), swatch_colors):
            self._swatch_btns[color_name].setStyleSheet(
                self._swatch_style(color, selected=(color_name == name))
            )
        self.accent_changed.emit(name)

    def _on_log_height(self, value: int):
        self._log_h_val.setText(f"{value}px")
        self._log_slider.setEnabled(self._log_cb.isChecked())
        self.log_height_changed.emit(value)

    def position_near(self, anchor_widget: QWidget):
        """Position bottom-right relative to anchor_widget's window."""
        window = anchor_widget.window()
        geo = window.geometry()
        x = geo.right() - self.width() - 12
        y = geo.bottom() - self.height() - 40
        self.move(x, y)
