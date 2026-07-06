"""Layers dock — a Photoshop-style layer stack panel for Astraios.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later. The blend-mode math and
add/remove/duplicate/merge-down operations mirror SASpro's
``pro/layers.py`` and ``pro/layers_dock.py``; the dock UI itself is
simplified for Astraios's single-image model (no MDI/multi-document view
picker, no per-layer transform dialog, no per-row widgets — one shared
blend-mode/opacity control pair acts on whichever layer is selected).

This widget is meant to be dropped into a ``QDockWidget`` by
``main_window`` — see ``_on_open_python_console`` for the pattern this
follows (lazy dock creation, ``setWidget``, signal wiring on first open).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from astraios.core.layers import BLEND_MODES, Layer, LayerStack
from astraios.ui.widgets.ui_kit import RunBtn, SliderRow, help_dot, styled_combo

log = logging.getLogger(__name__)


class LayersPanel(QWidget):
    """Dock content widget wrapping a :class:`~astraios.core.layers.LayerStack`.

    ``layers[0]`` (top of the on-screen list) is the topmost layer;
    the last row is the base. Any edit re-composites and emits
    ``composite_changed`` for a live, non-destructive canvas preview.
    ``flattened`` fires only when the user explicitly bakes the stack.
    """

    composite_changed = pyqtSignal(object)  # np.ndarray, float32 [0,1]
    flattened = pyqtSignal(object)  # np.ndarray, float32 [0,1]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stack = LayerStack()
        self._current_image_data: np.ndarray | None = None
        self._updating_controls = False

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        hint = QLabel(
            "Layers composite top → bottom — the last row is the base image. "
            "Check a row's box to show/hide that layer."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid); font-size: 11px;")
        v.addWidget(hint)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setAlternatingRowColors(True)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.currentRowChanged.connect(self._on_selection_changed)
        v.addWidget(self.list, 1)

        # ---- controls for the selected layer ----
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Blend Mode"))
        self.mode_combo = styled_combo(BLEND_MODES, current="Normal")
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        mode_row.addWidget(
            help_dot(
                "How the selected layer's pixels combine with everything "
                "below it in the stack. Ported from Seti Astro Suite Pro's "
                "layer blend modes (Normal, Screen, Multiply, Overlay, "
                "Soft/Hard Light, Color Dodge/Burn, Add, Subtract, Lighten, "
                "Darken, Difference, Relativistic Addition, Sigmoid, "
                "Luminosity, ...)."
            )
        )
        v.addLayout(mode_row)

        opacity_row = QHBoxLayout()
        self.opacity_slider = SliderRow("Opacity", 1.0, 0.0, 1.0, step=0.01, decimals=2)
        self.opacity_slider.value_changed.connect(self._on_opacity_changed)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(
            help_dot(
                "How strongly the selected layer's blended result shows "
                "through onto what's beneath it. 1.0 = fully applied, "
                "0.0 = invisible."
            )
        )
        v.addLayout(opacity_row)

        # ---- add layers ----
        add_row = QHBoxLayout()
        self.btn_add_current = QPushButton("Add From Current Image")
        self.btn_add_current.setToolTip(
            "Add a copy of the image currently loaded in Astraios as a new "
            "top layer."
        )
        self.btn_add_current.clicked.connect(self._on_add_from_current)
        add_row.addWidget(self.btn_add_current)
        self.btn_add_file = QPushButton("Add From File...")
        self.btn_add_file.clicked.connect(self._on_add_from_file)
        add_row.addWidget(self.btn_add_file)
        v.addLayout(add_row)

        # ---- manage layers ----
        manage_row = QHBoxLayout()
        self.btn_dup = QPushButton("Duplicate")
        self.btn_dup.clicked.connect(self._on_duplicate)
        manage_row.addWidget(self.btn_dup)
        self.btn_merge = QPushButton("Merge Down")
        self.btn_merge.setToolTip(
            "Flatten the selected layer onto the layer directly below it. "
            "The visual result of the whole stack is unchanged."
        )
        self.btn_merge.clicked.connect(self._on_merge_down)
        manage_row.addWidget(self.btn_merge)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)
        manage_row.addWidget(self.btn_delete)
        v.addLayout(manage_row)

        move_row = QHBoxLayout()
        self.btn_up = QPushButton("Move Up")
        self.btn_up.clicked.connect(lambda: self._on_move(-1))
        move_row.addWidget(self.btn_up)
        self.btn_down = QPushButton("Move Down")
        self.btn_down.clicked.connect(lambda: self._on_move(1))
        move_row.addWidget(self.btn_down)
        v.addLayout(move_row)

        self.btn_flatten = RunBtn("Flatten to Image", accent=True)
        self.btn_flatten.setToolTip(
            "Bake the full visible layer stack into a single image and "
            "replace the working image with it (undoable)."
        )
        self.btn_flatten.clicked.connect(self._on_flatten)
        v.addWidget(self.btn_flatten)

        self._update_controls_enabled()

    # ---- external API (called by main_window) -----------------------------

    def set_current_image(self, data: np.ndarray | None) -> None:
        """Cache the working image's pixel data for "Add From Current Image"."""
        self._current_image_data = None if data is None else np.asarray(data, dtype=np.float32)

    # ---- list <-> stack sync ------------------------------------------------

    def _rebuild_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for layer in self.stack.layers:
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked
            )
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._update_controls_enabled()

    def _select_row(self, row: int) -> None:
        row = max(0, min(row, self.list.count() - 1))
        if row >= 0:
            self.list.setCurrentRow(row)
        else:
            self._sync_selected_controls()

    def _selected_index(self) -> int:
        return self.list.currentRow()

    def _sync_selected_controls(self) -> None:
        idx = self._selected_index()
        self._updating_controls = True
        try:
            if 0 <= idx < len(self.stack.layers):
                layer = self.stack.layers[idx]
                self.mode_combo.setCurrentText(layer.blend_mode)
                self.opacity_slider.setValue(layer.opacity)
            else:
                self.mode_combo.setCurrentText("Normal")
                self.opacity_slider.setValue(1.0)
        finally:
            self._updating_controls = False
        self._update_controls_enabled()

    def _update_controls_enabled(self) -> None:
        n = len(self.stack.layers)
        idx = self._selected_index()
        has_selection = 0 <= idx < n
        self.mode_combo.setEnabled(has_selection)
        self.opacity_slider.setEnabled(has_selection)
        self.btn_dup.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)
        self.btn_merge.setEnabled(has_selection and idx < n - 1)
        self.btn_up.setEnabled(has_selection and idx > 0)
        self.btn_down.setEnabled(has_selection and idx < n - 1)
        self.btn_flatten.setEnabled(n > 0)

    # ---- recomposite / emit --------------------------------------------------

    def _emit_composite_changed(self) -> None:
        result = self.stack.composite()
        if result is not None:
            self.composite_changed.emit(result)

    # ---- list signal handlers ------------------------------------------------

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        idx = self.list.row(item)
        if not (0 <= idx < len(self.stack.layers)):
            return
        self.stack.layers[idx].visible = item.checkState() == Qt.CheckState.Checked
        self._emit_composite_changed()

    def _on_selection_changed(self, _row: int) -> None:
        self._sync_selected_controls()

    # ---- per-layer control handlers -------------------------------------------

    def _on_mode_changed(self, text: str) -> None:
        if self._updating_controls:
            return
        idx = self._selected_index()
        if 0 <= idx < len(self.stack.layers):
            self.stack.layers[idx].blend_mode = text if text in BLEND_MODES else "Normal"
            self._emit_composite_changed()

    def _on_opacity_changed(self, value: float) -> None:
        if self._updating_controls:
            return
        idx = self._selected_index()
        if 0 <= idx < len(self.stack.layers):
            self.stack.layers[idx].opacity = float(np.clip(value, 0.0, 1.0))
            self._emit_composite_changed()

    # ---- add ----------------------------------------------------------------

    def _on_add_from_current(self) -> None:
        if self._current_image_data is None:
            QMessageBox.information(
                self, "Layers", "No current image is loaded to add as a layer."
            )
            return
        layer = Layer(
            name=f"Layer {len(self.stack) + 1}",
            data=self._current_image_data.copy(),
        )
        self.stack.add(layer, index=0)
        self._rebuild_list()
        self._select_row(0)
        self._emit_composite_changed()

    def _on_add_from_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add Layer From File",
            "",
            "Images (*.fits *.fit *.fts *.xisf *.tif *.tiff *.png *.jpg *.jpeg)",
        )
        if not path:
            return
        try:
            from astraios.core.image_io import load_image

            image = load_image(path)
        except Exception as exc:  # noqa: BLE001 - surface any loader failure to the user
            QMessageBox.critical(self, "Layers", f"Could not load '{path}':\n{exc}")
            return
        layer = Layer(name=Path(path).stem, data=image.data.copy())
        self.stack.add(layer, index=0)
        self._rebuild_list()
        self._select_row(0)
        self._emit_composite_changed()

    # ---- manage ---------------------------------------------------------------

    def _on_duplicate(self) -> None:
        idx = self._selected_index()
        if not (0 <= idx < len(self.stack.layers)):
            return
        self.stack.duplicate(idx)
        self._rebuild_list()
        self._select_row(idx)
        self._emit_composite_changed()

    def _on_merge_down(self) -> None:
        idx = self._selected_index()
        if not (0 <= idx < len(self.stack.layers) - 1):
            return
        try:
            self.stack.merge_down(idx)
        except IndexError:
            return
        self._rebuild_list()
        self._select_row(idx)
        self._emit_composite_changed()

    def _on_delete(self) -> None:
        idx = self._selected_index()
        if not (0 <= idx < len(self.stack.layers)):
            return
        self.stack.remove(idx)
        self._rebuild_list()
        self._select_row(idx)
        self._emit_composite_changed()

    def _on_move(self, delta: int) -> None:
        idx = self._selected_index()
        if not (0 <= idx < len(self.stack.layers)):
            return
        new_idx = self.stack.move(idx, delta)
        self._rebuild_list()
        self._select_row(new_idx)
        self._emit_composite_changed()

    # ---- flatten ----------------------------------------------------------------

    def _on_flatten(self) -> None:
        result = self.stack.composite()
        if result is None:
            QMessageBox.information(self, "Layers", "There are no visible layers to flatten.")
            return
        self.flattened.emit(result)
