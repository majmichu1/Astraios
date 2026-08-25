"""workflow_bar.py — Astraios processing pipeline bar.

A horizontal strip under the quick toolbar that shows the workflow as a row
of steps and lets the user jump to any of them. Steps map to Tools Panel tabs
(Export emits -1, handled by the main window as Save/Export). The main window
also calls set_current()/mark_complete() as real operations succeed, so the
bar reflects actual progress, not just clicks.

The look is the design handoff's WorkflowBar: steps packed from the left,
a check mark on completed steps, the current step in the accent colour with
a 2px underline, the next step in secondary text and the rest dimmed.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from astraios.ui.theme import (
    ACCENT,
    ACCENT_HOVER,
    BG_HOVER,
    BG_SECONDARY,
    BORDER,
    TEXT_DIM,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

_STEPS = [
    ("Pre-Process", "Calibrate · Cosmetic"),
    ("Stacking",    "Align · Integrate"),
    ("Background",  "Extract · ABE"),
    ("Stretch",     "GHS · Curves"),
    ("Transform",   "Resize · Rotate"),
    ("Color",       "SCNR · Calibrate"),
    ("Detail",      "Decon · Denoise"),
    ("Export",      "FITS · TIFF · PNG"),
]

# Map step index → Tools Panel tab index (0-based). Export (step 7) emits -1,
# which the main window interprets as Save/Export rather than a tab switch.
_STEP_TO_TAB = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: -1}


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
        self.setObjectName("WorkflowBar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._current = 0
        self._completed: set[int] = set()
        self.setFixedHeight(42)

        # Scroll area so the bar stays usable on narrow windows; the scrollbar
        # itself is hidden and the wheel scrolls it sideways.
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._scroll.viewport().setStyleSheet("background: transparent;")

        inner = QWidget()
        inner.setStyleSheet("background: transparent; border: none;")
        self._layout = QHBoxLayout(inner)
        self._layout.setContentsMargins(8, 0, 8, 0)
        self._layout.setSpacing(0)

        self._btns: list[_StepButton] = []
        self._arrows: list[QLabel] = []

        for i, (name, sub) in enumerate(_STEPS):
            btn = _StepButton(i, name, sub)
            btn.clicked.connect(lambda idx: self._on_click(idx))
            self._btns.append(btn)
            self._layout.addWidget(btn)

            if i < len(_STEPS) - 1:
                arrow = QLabel("›")
                arrow.setStyleSheet(
                    f"color: {BORDER}; font-size: 10px; padding: 0 2px;"
                    " background: transparent;"
                )
                self._arrows.append(arrow)
                self._layout.addWidget(arrow)

        self._layout.addStretch()
        self._scroll.setWidget(inner)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

        self.setStyleSheet(
            f"#WorkflowBar {{ background: {BG_SECONDARY}; border-bottom: 1px solid {BORDER}; }}"
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

    def wheelEvent(self, event):
        bar = self._scroll.horizontalScrollBar()
        delta = event.angleDelta().y() or event.angleDelta().x()
        bar.setValue(bar.value() - delta // 2)
        event.accept()

    def _on_click(self, idx: int) -> None:
        tab = _STEP_TO_TAB.get(idx)
        if tab is not None:
            self.step_clicked.emit(tab)  # tab index, or -1 for Export
        self._current = idx
        self._refresh()

    def _refresh(self) -> None:
        # The design's "next" step is the one after the last completed step.
        next_idx = (max(self._completed) + 1) if self._completed else 0
        for i, btn in enumerate(self._btns):
            btn.set_state(
                done=i in self._completed,
                current=i == self._current,
                next_=(i == next_idx),
            )
        for i, arrow in enumerate(self._arrows):
            arrow.setStyleSheet(
                f"color: {ACCENT if i in self._completed else BORDER}; "
                "font-size: 10px; padding: 0 2px; background: transparent;"
            )
        if 0 <= self._current < len(self._btns):
            self._scroll.ensureWidgetVisible(self._btns[self._current], 24, 0)


class _StepButton(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, idx: int, name: str, subtitle: str, parent=None):
        super().__init__(parent)
        self._idx = idx
        self._hovered = False
        self._current = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: transparent; border: none;")

        col = QVBoxLayout(self)
        col.setContentsMargins(14, 6, 14, 6)
        col.setSpacing(1)
        col.setAlignment(Qt.AlignmentFlag.AlignCenter)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)
        top.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._check_lbl = QLabel("")
        self._check_lbl.setStyleSheet(
            f"color: {ACCENT}; font-size: 10px; background: transparent;"
        )
        self._check_lbl.hide()
        self._name_lbl = QLabel(name)
        top.addWidget(self._check_lbl)
        top.addWidget(self._name_lbl)
        col.addLayout(top)

        self._sub_lbl = QLabel(subtitle)
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        col.addWidget(self._sub_lbl)

        self.set_state(done=False, current=idx == 0, next_=idx == 0)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._idx)

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        if self._hovered:
            p.fillRect(self.rect(), QColor(BG_HOVER))
        if self._current:
            p.fillRect(0, self.height() - 2, self.width(), 2, QColor(ACCENT))
        p.end()

    def set_state(self, done: bool, current: bool, next_: bool) -> None:
        self._current = current
        name_color = (
            ACCENT if current
            else TEXT_PRIMARY if done
            else TEXT_SECONDARY if next_
            else TEXT_DIM
        )
        sub_color = ACCENT_HOVER if current else TEXT_SECONDARY
        self._name_lbl.setStyleSheet(
            f"color: {name_color}; font-size: 11px; font-weight: 600; background: transparent;"
        )
        self._sub_lbl.setStyleSheet(
            f"color: {sub_color}; font-size: 9px; background: transparent;"
        )
        self._check_lbl.setText("✓" if done else "")
        self._check_lbl.setVisible(done)
        self.update()
