"""Mask Controls Widget — dropdown for selecting/managing masks in tool groups."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

if TYPE_CHECKING:
    from cosmica.core.masks import Mask


class MaskSelector(QWidget):
    """Compact mask selection widget for use in tool groups.

    Shows a dropdown of available masks plus buttons to create/edit/invert.
    """

    mask_changed = pyqtSignal(object)  # Mask or None
    create_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._masks: dict[str, Mask] = {}  # name -> Mask

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        lbl = QLabel("Mask:")
        lbl.setStyleSheet("font-size: 11px;")
        layout.addWidget(lbl)

        self._combo = QComboBox()
        self._combo.setToolTip("Select a mask to restrict processing to specific areas")
        self._combo.addItem("None (full image)")
        self._combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self._combo, 1)

        self._btn_new = QPushButton("+")
        self._btn_new.setFixedWidth(28)
        self._btn_new.setToolTip("Create a new mask")
        self._btn_new.clicked.connect(self.create_requested.emit)
        layout.addWidget(self._btn_new)

    def add_mask(self, mask: Mask) -> None:
        """Register a mask in the selector."""
        self._masks[mask.name] = mask
        self._combo.addItem(mask.name)

    def remove_mask(self, name: str) -> None:
        """Remove a mask from the selector."""
        if name in self._masks:
            del self._masks[name]
            idx = self._combo.findText(name)
            if idx >= 0:
                self._combo.removeItem(idx)

    def clear_masks(self) -> None:
        """Remove all masks."""
        self._masks.clear()
        self._combo.clear()
        self._combo.addItem("None (full image)")

    def current_mask(self) -> Mask | None:
        """Return the currently selected mask, or None."""
        name = self._combo.currentText()
        return self._masks.get(name)

    def set_masks(self, masks: list[Mask]) -> None:
        """Replace all masks with the given list."""
        current = self._combo.currentText()
        self._masks.clear()
        self._combo.clear()
        self._combo.addItem("None (full image)")
        for m in masks:
            self._masks[m.name] = m
            self._combo.addItem(m.name)
        # Restore selection if still present
        idx = self._combo.findText(current)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

    def _on_selection_changed(self, index: int):
        self.mask_changed.emit(self.current_mask())
