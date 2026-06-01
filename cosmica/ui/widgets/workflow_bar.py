"""workflow_bar.py — Cosmica processing pipeline progress bar.

A horizontal bar that shows the 7-step workflow and lets the user
jump to any step. Drop into cosmica/ui/widgets/workflow_bar.py
and add to the main window layout between the quick toolbar and panels.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QWidget,
)

BG_SECONDARY  = "#161b22"
BG_HOVER      = "#30363d"
BORDER        = "#30363d"
TEXT_PRIMARY  = "#e6edf3"
TEXT_SECONDARY= "#8b949e"
ACCENT        = "#2ea043"
ACCENT_HOVER  = "#3fb950"
DIM           = "#4a5260"


_STEPS = [
    ("Pre-Process", "Calibrate · Cosmetic"),
    ("Stacking",    "Align · Integrate"),
    ("Background",  "Extract · ABE"),
    ("Stretch",     "GHS · Curves"),
    ("Color",       "SCNR · Calibrate"),
    ("Detail",      "Decon · Denoise"),
    ("Export",      "FITS · TIFF · PNG"),
]

# Map step index → Tools Panel tab index (0-based)
_STEP_TO_TAB = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: None}


class WorkflowBar(QWidget):
    """Horizontal pipeline bar.

    Signals
    -------
    step_clicked(int)
        Emitted with the Tools Panel tab index when a step is clicked.
    """

    step_clicked = pyqtSignal(int)   # Tools Panel tab index

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 0
        self._completed: set[int] = set()
        self.setFixedHeight(42)

        # Scroll area so bar stays usable on narrow windows
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(
            scroll.verticalScrollBarPolicy().ScrollBarAlwaysOff
        )
        scroll.setHorizontalScrollBarPolicy(
            scroll.horizontalScrollBarPolicy().ScrollBarAlwaysOff
        )
        scroll.setStyleSheet("QScrollArea { border: none; }")

        inner = QWidget()
        self._layout = QHBoxLayout(inner)
        self._layout.setContentsMargins(8, 0, 8, 0)
        self._layout.setSpacing(0)

        self._btns: list[QPushButton] = []
        self._arrows: list[QLabel] = []

        for i, (name, sub) in enumerate(_STEPS):
            btn = _StepButton(i, name, sub)
            btn.clicked.connect(lambda _, idx=i: self._on_click(idx))
            self._btns.append(btn)
            self._layout.addWidget(btn)

            if i < len(_STEPS) - 1:
                arrow = QLabel("›")
                arrow.setStyleSheet(
                    f"color: {BORDER}; font-size: 14px; padding: 0 2px;"
                )
                self._arrows.append(arrow)
                self._layout.addWidget(arrow)

        scroll.setWidget(inner)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.setStyleSheet(
            f"background: {BG_SECONDARY}; border-bottom: 1px solid {BORDER};"
        )
        self._refresh()

    # ── public API ────────────────────────────────────────

    def set_current(self, step: int) -> None:
        """Set the active step (0-based). Marks all prior steps as done."""
        self._current = step
        self._completed = set(range(step))
        self._refresh()

    def mark_complete(self, step: int) -> None:
        self._completed.add(step)
        self._refresh()

    # ── internals ─────────────────────────────────────────

    def _on_click(self, idx: int) -> None:
        tab = _STEP_TO_TAB.get(idx)
        if tab is not None:
            self.step_clicked.emit(tab)
        elif idx == 6:
            self.step_clicked.emit(-1)  # Export sentinel — no matching tab
        self.set_current(idx)

    def _refresh(self) -> None:
        for i, btn in enumerate(self._btns):
            done    = i in self._completed
            current = i == self._current
            future  = i > self._current and i not in self._completed
            btn.set_state(done=done, current=current, future=future)

        for i, arrow in enumerate(self._arrows):
            done = i in self._completed
            arrow.setStyleSheet(
                f"color: {ACCENT if done else BORDER}; "
                "font-size: 14px; padding: 0 2px;"
            )


class _StepButton(QPushButton):
    def __init__(self, idx: int, name: str, subtitle: str, parent=None):
        super().__init__(parent)
        self._idx = idx
        self.setCursor(
            __import__("PyQt6.QtCore", fromlist=["Qt"]).Qt.CursorShape.PointingHandCursor
        )
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(0)

        col = QWidget()
        col.setStyleSheet("background-color: transparent;")
        col_lay = __import__(
            "PyQt6.QtWidgets", fromlist=["QVBoxLayout"]
        ).QVBoxLayout(col)
        col_lay.setContentsMargins(0, 0, 0, 0)
        col_lay.setSpacing(1)

        self._check_name = QHBoxLayout()
        self._check_lbl  = QLabel("")
        self._check_lbl.setStyleSheet(f"color: {ACCENT}; font-size: 10px; background-color: transparent;")
        self._name_lbl   = QLabel(name)
        self._check_name.addWidget(self._check_lbl)
        self._check_name.addWidget(self._name_lbl)
        self._check_name.addStretch()

        self._sub_lbl = QLabel(subtitle)
        self._sub_lbl.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9px; background-color: transparent;"
        )

        col_lay.addLayout(self._check_name)
        col_lay.addWidget(self._sub_lbl)
        lay.addWidget(col)

    def set_state(self, done: bool, current: bool, future: bool) -> None:
        name_color = (
            ACCENT if current
            else TEXT_PRIMARY if done
            else TEXT_SECONDARY if not future
            else DIM
        )
        sub_color = ACCENT_HOVER if current else TEXT_SECONDARY
        self._name_lbl.setStyleSheet(
            f"color: {name_color}; font-size: 11px; font-weight: 600; background-color: transparent;"
        )
        self._sub_lbl.setStyleSheet(
            f"color: {sub_color}; font-size: 9px; background-color: transparent;"
        )
        self._check_lbl.setText("✓ " if done else "")
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none;
            }}
            QPushButton:hover {{ background: {BG_HOVER}; }}
        """)
