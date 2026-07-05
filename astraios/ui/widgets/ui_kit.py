"""ui_kit.py — Astraios reusable PyQt6 widget library.

Provides CollapsibleSection, SliderRow, RunBtn, FieldRow and helpers
that match the HTML prototype's visual design exactly.
Drop this into astraios/ui/widgets/ui_kit.py.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# ── Design tokens (mirrors theme.py) ─────────────────────
BG_PRIMARY    = "#0d1117"
BG_SECONDARY  = "#161b22"
BG_TERTIARY   = "#21262d"
BG_HOVER      = "#30363d"
BORDER        = "#30363d"
TEXT_PRIMARY  = "#e6edf3"
TEXT_SECONDARY= "#8b949e"
ACCENT        = "#2ea043"
ACCENT_HOVER  = "#3fb950"
ACCENT_DARK   = "#1a4d2e"
ACCENT_PURPLE = "#8957e5"
RED           = "#f85149"
ORANGE        = "#d29922"
BLUE          = "#388bfd"

FONT_MONO = '"JetBrains Mono", "Fira Code", "Cascadia Code", Consolas, monospace'


# ── Helpers ───────────────────────────────────────────────

def scrollable_tab(layout: QVBoxLayout) -> QScrollArea:
    """Wrap a QVBoxLayout in a styled, horizontally-locked scroll area."""
    layout.addStretch()
    container = QWidget()
    container.setStyleSheet(f"background-color: {BG_PRIMARY};")
    container.setLayout(layout)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(container)
    scroll.setStyleSheet(f"""
        QScrollArea {{ border: none; background: {BG_PRIMARY}; }}
        QScrollBar:vertical {{
            background: {BG_PRIMARY}; width: 6px; margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background: {BG_HOVER}; border-radius: 3px; min-height: 20px;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    """)
    return scroll


def make_label(text: str, color: str = TEXT_SECONDARY,
               size: int = 12, bold: bool = False,
               mono: bool = False) -> QLabel:
    lbl = QLabel(text)
    weight = "600" if bold else "normal"
    family = f"font-family: {FONT_MONO};" if mono else ""
    lbl.setStyleSheet(
        f"color: {color}; font-size: {size}px; font-weight: {weight}; background-color: transparent; {family}"
    )
    return lbl


def divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet(f"background: {BORDER}; border: none;")
    return line


# ── InfoLabel ─────────────────────────────────────────────

class InfoLabel(QLabel):
    """Small gray description text used inside sections."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 11px; background-color: transparent;"
        )


# ── RunBtn ────────────────────────────────────────────────

class RunBtn(QPushButton):
    """Full-width apply button. accent=True → green fill; flat=True → outlined."""
    def __init__(self, label: str, accent: bool = True,
                 flat: bool = False, parent=None):
        super().__init__(label, parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.setFixedHeight(30)
        if flat:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {BG_TERTIARY}; color: {TEXT_PRIMARY};
                    border: 1px solid {BORDER}; border-radius: 6px;
                    padding: 0 8px; font-size: 12px; font-weight: 500;
                }}
                QPushButton:hover {{ background: {BG_HOVER}; }}
                QPushButton:pressed {{ background: {ACCENT_DARK}; }}
                QPushButton:disabled {{ color: {TEXT_SECONDARY}; }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {ACCENT}; color: #ffffff;
                    border: none; border-radius: 6px;
                    padding: 0 8px; font-size: 12px; font-weight: 600;
                }}
                QPushButton:hover {{ background: {ACCENT_HOVER}; }}
                QPushButton:pressed {{ background: {ACCENT_DARK}; }}
                QPushButton:disabled {{
                    background: {BG_TERTIARY}; color: {TEXT_SECONDARY};
                }}
            """)


# ── SliderRow ─────────────────────────────────────────────

class SliderRow(QWidget):
    """Label + slider + value display. Double-click slider to reset to default."""
    value_changed = pyqtSignal(float)

    def __init__(
        self, label: str, value: float,
        min_val: float, max_val: float,
        step: float = 1.0, decimals: int = 0,
        default: float | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setStyleSheet("background-color: transparent;")
        self._dec   = decimals
        self._step  = step
        self._scale = max(1, round(1.0 / step)) if step < 1 else 1
        self._default = default if default is not None else value

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)

        # header row
        hdr = QHBoxLayout()
        self._lbl = make_label(label, TEXT_SECONDARY, 12)
        self._val_lbl = make_label(self._fmt(value), ACCENT, 11, mono=True)
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._val_lbl.setMinimumWidth(44)
        hdr.addWidget(self._lbl)
        hdr.addStretch()
        hdr.addWidget(self._val_lbl)
        vbox.addLayout(hdr)

        self._slider = _ResetSlider(
            Qt.Orientation.Horizontal,
            default_int=int(self._default * self._scale),
        )
        self._slider.setRange(
            int(min_val * self._scale), int(max_val * self._scale)
        )
        self._slider.setValue(int(value * self._scale))
        self._slider.setStyleSheet(f"""
            QSlider {{ background: transparent; border: none; }}
            QSlider::groove:horizontal {{
                height: 4px; background: {BG_TERTIARY}; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT}; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
                border: 2px solid {BG_PRIMARY};
            }}
            QSlider::handle:horizontal:hover {{ background: {ACCENT_HOVER}; }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT}; border-radius: 2px;
            }}
        """)
        self._slider.valueChanged.connect(self._on_slider)
        vbox.addWidget(self._slider)

    # ---- internal ----
    def _fmt(self, v: float) -> str:
        return f"{v:.{self._dec}f}"

    def _on_slider(self, raw: int):
        v = raw / self._scale
        self._val_lbl.setText(self._fmt(v))
        self.value_changed.emit(v)

    # ---- public API ----
    def value(self) -> float:
        return self._slider.value() / self._scale

    def setValue(self, v: float):
        self._slider.setValue(int(v * self._scale))


class _ResetSlider(QSlider):
    """Slider that resets to default on double-click."""
    def __init__(self, orientation, default_int: int = 0, parent=None):
        super().__init__(orientation, parent)
        self._default_int = default_int

    def mouseDoubleClickEvent(self, event):
        self.setValue(self._default_int)
        super().mouseDoubleClickEvent(event)


# ── Styled input factories ────────────────────────────────

_INPUT_SS = f"""
    background: {BG_TERTIARY}; color: {TEXT_PRIMARY};
    border: 1px solid {BORDER}; border-radius: 5px;
    padding: 4px 8px; font-size: 12px;
"""
_INPUT_FOCUS = f"border-color: {ACCENT};"


def styled_combo(options: list[str], current: str | None = None) -> QComboBox:
    combo = QComboBox()
    combo.addItems(options)
    if current and (idx := combo.findText(current)) >= 0:
        combo.setCurrentIndex(idx)
    combo.setStyleSheet(f"""
        QComboBox {{ {_INPUT_SS} }}
        QComboBox:focus {{ {_INPUT_FOCUS} }}
        QComboBox::drop-down {{ border: none; padding-right: 8px; }}
        QComboBox QAbstractItemView {{
            background: {BG_SECONDARY}; color: {TEXT_PRIMARY};
            selection-background-color: {ACCENT_DARK};
            border: 1px solid {BORDER};
        }}
    """)
    return combo


def styled_spin(min_val: float, max_val: float, value: float,
                step: float = 1.0, decimals: int = 0,
                suffix: str = "") -> QDoubleSpinBox | QSpinBox:
    if decimals > 0 or isinstance(step, float):
        w: QDoubleSpinBox | QSpinBox = QDoubleSpinBox()
        w.setDecimals(decimals)
        w.setSingleStep(float(step))
    else:
        w = QSpinBox()
        w.setSingleStep(int(step))
    w.setRange(min_val, max_val)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    w.setStyleSheet(f"""
        QDoubleSpinBox, QSpinBox {{
            {_INPUT_SS} font-family: {FONT_MONO};
        }}
        QDoubleSpinBox:focus, QSpinBox:focus {{ {_INPUT_FOCUS} }}
    """)
    return w


def styled_check(label: str, checked: bool = False) -> QCheckBox:
    cb = QCheckBox(label)
    cb.setChecked(checked)
    cb.setStyleSheet(f"""
        QCheckBox {{ color: {TEXT_PRIMARY}; font-size: 12px; spacing: 6px; background-color: transparent; }}
        QCheckBox::indicator {{
            width: 14px; height: 14px;
            border: 1.5px solid {BORDER}; border-radius: 3px;
            background: {BG_TERTIARY};
        }}
        QCheckBox::indicator:checked {{
            background: {ACCENT}; border-color: {ACCENT};
        }}
        QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
    """)
    return cb


def field_row(label_text: str, widget: QWidget,
              label_width: int = 110, help_text: str | None = None) -> QHBoxLayout:
    """Return a QHBoxLayout with a fixed-width label on the left.

    With ``help_text``, a hover "?" dot explaining the setting is appended.
    """
    row = QHBoxLayout()
    row.setSpacing(8)
    lbl = make_label(label_text, TEXT_SECONDARY, 12)
    lbl.setFixedWidth(label_width)
    row.addWidget(lbl)
    row.addWidget(widget)
    if help_text:
        row.addWidget(help_dot(help_text))
    return row


def help_dot(text: str, parent=None) -> QLabel:
    """A small circled "?" that explains a tool or setting on hover.

    Used next to section titles and control labels so every tool can carry a
    plain-language explanation without cluttering the panel. The tooltip is
    rich text and wraps at a readable width.
    """
    dot = QLabel("?", parent)
    dot.setFixedSize(14, 14)
    dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
    dot.setStyleSheet(
        f"QLabel {{ color: {TEXT_SECONDARY}; background: {BG_PRIMARY};"
        f" border: 1px solid {BORDER}; border-radius: 7px;"
        " font-size: 9px; font-weight: 700; }"
        f"QLabel:hover {{ color: #ffffff; background: {ACCENT}; border-color: {ACCENT}; }}"
    )
    # <qt> forces rich text so Qt word-wraps long explanations.
    dot.setToolTip(f"<qt>{text}</qt>")
    dot.setToolTipDuration(60000)
    dot.setCursor(Qt.CursorShape.WhatsThisCursor)
    return dot


def btn_row(specs: list[tuple[str, bool]]) -> tuple[QHBoxLayout, list[QPushButton]]:
    """Return (layout, [buttons]) for a row of equal-width buttons.
    specs: [(label, is_flat), ...]
    """
    layout = QHBoxLayout()
    layout.setSpacing(4)
    btns: list[QPushButton] = []
    for label, flat in specs:
        b = RunBtn(label, accent=not flat, flat=flat)
        layout.addWidget(b)
        btns.append(b)
    return layout, btns


# ── CollapsibleSection ────────────────────────────────────

class CollapsibleSection(QWidget):
    """Collapsible group box — matches the HTML <Section> component.

    Usage::

        sec = CollapsibleSection("Calibration", accent=True)
        sec.body.addWidget(...)          # raw access
        sec.add_info("Description...")   # helpers
        self._kappa = sec.add_slider("Kappa", 3.0, 0.5, 10, 0.1, 1)
        sec.add_run("▶ Stack", self.run_stacking.emit)
    """

    def __init__(
        self, title: str,
        accent: bool = False,
        default_open: bool | None = None,
        help_text: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        # Default: primary (accent) sections start open, the rest collapsed, so a
        # tab reads as a scannable menu instead of a wall of expanded tools. An
        # explicit default_open=True/False overrides this.
        if default_open is None:
            default_open = accent
        self._open = default_open
        self._default_open = default_open  # restored when the tool search clears
        # Text searched by the Tools Panel search box (title + any info text).
        self._search_text = title.lower()
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(0)

        # ── header button ──────────────────────────────────
        self._hdr = QPushButton()
        self._hdr.setFixedHeight(30)
        self._hdr.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hdr.clicked.connect(self._toggle)

        hdr_inner = QHBoxLayout(self._hdr)
        hdr_inner.setContentsMargins(10, 0, 10, 0)
        hdr_inner.setSpacing(0)

        if accent:
            pip = QFrame()
            pip.setFixedSize(3, 12)
            pip.setStyleSheet(
                f"background: {ACCENT}; border-radius: 1px; border: none;"
            )
            hdr_inner.addWidget(pip)
            hdr_inner.addSpacing(7)

        self._title_lbl = QLabel(title)
        self._title_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 12px; font-weight: 600; background-color: transparent; border: none;"
        )
        hdr_inner.addWidget(self._title_lbl)
        self._help_dot: QLabel | None = None
        if help_text:
            hdr_inner.addSpacing(6)
            self._help_dot = help_dot(help_text)
            hdr_inner.addWidget(self._help_dot)
            self._search_text += " " + help_text.lower()
        hdr_inner.addStretch()

        self._chevron = QLabel("▲" if default_open else "▼")
        self._chevron.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9px; background-color: transparent; border: none;"
        )
        hdr_inner.addWidget(self._chevron)
        outer.addWidget(self._hdr)
        self._apply_header_style()

        # ── content ────────────────────────────────────────
        self._content = QWidget()
        self._content.setVisible(default_open)
        self._content.setStyleSheet(f"""
            QWidget#sec_content {{
                background: {BG_PRIMARY};
                border: 1px solid {BORDER};
                border-top: none;
                border-radius: 0 0 6px 6px;
            }}
        """)
        self._content.setObjectName("sec_content")

        self.body = QVBoxLayout(self._content)
        self.body.setContentsMargins(10, 10, 10, 10)
        self.body.setSpacing(8)
        outer.addWidget(self._content)

    # ── toggle ────────────────────────────────────────────
    def _apply_header_style(self):
        br = "6px 6px 0 0" if self._open else "6px"
        self._hdr.setStyleSheet(f"""
            QPushButton {{
                background: {BG_SECONDARY}; border: 1px solid {BORDER};
                border-radius: {br}; text-align: left;
            }}
            QPushButton:hover {{ background: {BG_HOVER}; }}
        """)

    def _toggle(self):
        self.set_open(not self._open)

    def set_open(self, open_: bool):
        """Expand or collapse programmatically."""
        self._open = open_
        self._content.setVisible(open_)
        self._chevron.setText("▲" if open_ else "▼")
        self._apply_header_style()

    def title(self) -> str:
        return self._title_lbl.text()

    def matches(self, query: str) -> bool:
        """True if the (lowercased) query appears in the title or info text."""
        return query in self._search_text

    # ── convenience adders ────────────────────────────────
    def add_widget(self, w: QWidget) -> QWidget:
        self.body.addWidget(w)
        return w

    def add_layout(self, lay) -> object:
        self.body.addLayout(lay)
        return lay

    def add_info(self, text: str) -> InfoLabel:
        self._search_text += " " + text.lower()
        # Surface the description as a header tooltip so it's readable on hover
        # even while the section is collapsed (most are, by default).
        if not self._hdr.toolTip():
            self._hdr.setToolTip(text)
        return self.add_widget(InfoLabel(text))

    def add_slider(
        self, label: str, value: float,
        min_val: float, max_val: float,
        step: float = 1.0, decimals: int = 0,
        default: float | None = None,
        help_text: str | None = None,
    ) -> SliderRow:
        row = SliderRow(label, value, min_val, max_val, step, decimals, default)
        if help_text:
            row.setToolTip(f"<qt>{help_text}</qt>")
            row.setToolTipDuration(60000)
        self.body.addWidget(row)
        return row

    def add_combo(
        self, label: str, options: list[str],
        current: str | None = None, lw: int = 110,
        help_text: str | None = None,
    ) -> QComboBox:
        combo = styled_combo(options, current)
        self.body.addLayout(field_row(label, combo, lw, help_text=help_text))
        return combo

    def add_spin(
        self, label: str,
        min_val: float, max_val: float, value: float,
        step: float = 1.0, decimals: int = 0,
        suffix: str = "", lw: int = 110,
        help_text: str | None = None,
    ) -> QDoubleSpinBox | QSpinBox:
        spin = styled_spin(min_val, max_val, value, step, decimals, suffix)
        self.body.addLayout(field_row(label, spin, lw, help_text=help_text))
        return spin

    def add_check(self, label: str, checked: bool = False,
                  help_text: str | None = None) -> QCheckBox:
        check = styled_check(label, checked)
        if help_text:
            check.setToolTip(f"<qt>{help_text}</qt>")
            check.setToolTipDuration(60000)
        return self.add_widget(check)

    def add_run(
        self, label: str,
        callback=None,
        flat: bool = False,
    ) -> RunBtn:
        btn = RunBtn(label, accent=not flat, flat=flat)
        if callback:
            btn.clicked.connect(callback)
        self.body.addWidget(btn)
        return btn

    def add_btn_row(
        self, specs: list[tuple[str, bool]],
    ) -> list[QPushButton]:
        lay, btns = btn_row(specs)
        self.body.addLayout(lay)
        return btns

    def add_divider(self):
        self.body.addWidget(divider())

    def add_code_block(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        lbl.setStyleSheet(f"""
            background: {BG_TERTIARY}; color: {ACCENT};
            border: 1px solid {BORDER}; border-radius: 4px;
            padding: 6px 8px; font-family: {FONT_MONO}; font-size: 10px;
        """)
        self.body.addWidget(lbl)
        return lbl

    def add_status_label(self, text: str, color: str = TEXT_SECONDARY) -> QLabel:
        lbl = make_label(text, color, 10)
        self.body.addWidget(lbl)
        return lbl

    def add_inline_grid(self, widgets_2col: list[tuple[str, QWidget]]) -> None:
        """Add pairs (label, widget) in a 2-column grid layout."""
        from PyQt6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setSpacing(6)
        for i, (lbl_text, w) in enumerate(widgets_2col):
            col = (i % 2) * 2
            row_i = i // 2
            lbl = make_label(lbl_text, TEXT_SECONDARY, 10)
            grid.addWidget(lbl, row_i * 2,     col)
            grid.addWidget(w,   row_i * 2 + 1, col)
        self.body.addLayout(grid)
