"""Processing History dialog — non-destructive, linear editing history.

Shows the recorded steps as an ordered list. Selecting a step previews the
image at that stage; the checkbox toggles a step on/off; steps can be deleted,
reordered, and exported as a reusable macro. All edits recompute the image
non-destructively from the base.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from astraios.core.processing_graph import ProcessingGraph


def _summarize_params(params: dict, max_items: int = 4) -> str:
    """A short, human-readable summary of a step's params for the tooltip."""
    if not params:
        return ""
    parts = []
    for k, v in list(params.items())[:max_items]:
        if hasattr(v, "name"):          # enum member
            sv = v.name
        elif isinstance(v, bool):
            sv = str(v)
        elif isinstance(v, float):
            sv = f"{v:.3g}"
        elif isinstance(v, (list, tuple)):
            sv = f"[{len(v)}]"
        elif isinstance(v, dict):
            continue                    # nested (e.g. curve points) — skip
        else:
            sv = str(v)
        parts.append(f"{k}={sv}")
    summary = ", ".join(parts)
    if len(params) > max_items and summary:
        summary += ", …"
    return summary


class ProcessingGraphDialog(QDialog):
    """Non-destructive editing history with stage preview and editing."""

    # Preview the image after step <index> (-1 = base image).
    view_stage = pyqtSignal(int)
    # The step list changed (toggle/delete/reorder); recompute the result.
    history_changed = pyqtSignal()
    # Export the current history as a macro.
    export_macro = pyqtSignal()

    def __init__(self, parent, graph: ProcessingGraph):
        super().__init__(parent)
        self._graph = graph
        self.setWindowTitle("Processing History")
        self.setMinimumSize(460, 440)
        self.setModal(False)
        self._suppress = False
        self._setup_ui()
        self._refresh()
        # Pick up steps recorded while the dialog is open.
        self._timer = QTimer(self)
        self._timer.setInterval(800)
        self._timer.timeout.connect(self._refresh_if_changed)
        self._timer.start()
        self._last_count = len(graph.steps)

    # ------------------------------------------------------------------ #
    def _setup_ui(self):
        lay = QVBoxLayout(self)

        info = QLabel(
            "Select a step to preview that stage. Double-click to edit its "
            "parameters. Use the checkbox to disable a step, or reorder and "
            "delete steps; the result recomputes from the original, "
            "non-destructively."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; padding: 4px;")
        lay.addWidget(info)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentRowChanged.connect(self._on_row_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(self._edit_step)
        lay.addWidget(self._list)

        row1 = QHBoxLayout()
        self._up_btn = QPushButton("Move Up")
        self._up_btn.clicked.connect(lambda: self._move(-1))
        self._down_btn = QPushButton("Move Down")
        self._down_btn.clicked.connect(lambda: self._move(1))
        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._delete)
        for b in (self._up_btn, self._down_btn, self._delete_btn):
            b.setEnabled(False)
            row1.addWidget(b)
        row1.addStretch()
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self._export_btn = QPushButton("Export as Macro…")
        self._export_btn.clicked.connect(self.export_macro.emit)
        row2.addWidget(self._export_btn)
        row2.addStretch()
        self._base_btn = QPushButton("View Original")
        self._base_btn.clicked.connect(lambda: self.view_stage.emit(-1))
        row2.addWidget(self._base_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        row2.addWidget(close_btn)
        lay.addLayout(row2)

    # ------------------------------------------------------------------ #
    def _refresh(self):
        self._suppress = True
        self._list.clear()
        for r in self._graph.list_steps():
            item = QListWidgetItem(f"{r['index'] + 1}. {r['label']}")
            item.setData(Qt.ItemDataRole.UserRole, r["index"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if r["enabled"] else Qt.CheckState.Unchecked
            )
            tips = []
            if r["replayable"]:
                summary = _summarize_params(r["params"])
                if summary:
                    tips.append(summary)
            if r["mask_name"]:
                tips.append(f"mask: {r['mask_name']}")
            if not r["replayable"]:
                tips.append("display-only (recorded before it was replayable)")
                item.setForeground(Qt.GlobalColor.darkGray)
            elif not r["enabled"]:
                item.setForeground(Qt.GlobalColor.gray)
            item.setToolTip("; ".join(tips) if tips else "Replayable step")
            self._list.addItem(item)
        self._suppress = False
        self._last_count = len(self._graph.steps)
        self._update_buttons()

    def _refresh_if_changed(self):
        if len(self._graph.steps) != self._last_count:
            self._refresh()

    def _update_buttons(self):
        row = self._list.currentRow()
        n = self._list.count()
        sel = row >= 0
        self._delete_btn.setEnabled(sel)
        self._up_btn.setEnabled(sel and row > 0)
        self._down_btn.setEnabled(sel and row < n - 1)
        self._export_btn.setEnabled(n > 0)

    # ------------------------------------------------------------------ #
    def _on_row_changed(self, row: int):
        self._update_buttons()
        if row >= 0:
            self.view_stage.emit(row)

    def _on_item_changed(self, item: QListWidgetItem):
        if self._suppress:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        enabled = item.checkState() == Qt.CheckState.Checked
        self._graph.set_enabled(int(index), enabled)
        self.history_changed.emit()

    def _delete(self):
        row = self._list.currentRow()
        if row < 0:
            return
        if self._graph.remove(row):
            self._refresh()
            self.history_changed.emit()

    def _edit_step(self, item: QListWidgetItem):
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None:
            return
        index = int(index)
        if not (0 <= index < len(self._graph.steps)):
            return
        step = self._graph.steps[index]
        if not step.replayable or not step.params:
            self._info_flash("This step has no editable parameters.")
            return
        from astraios.ui.dialogs.param_edit_dialog import ParamEditDialog

        dlg = ParamEditDialog(self, step.label, step.params)
        if dlg.exec():
            self._graph.update_params(index, dlg.get_params())
            self._refresh()
            self._list.setCurrentRow(index)
            self.history_changed.emit()

    def _info_flash(self, text: str):
        self.setWindowTitle(f"Processing History — {text}")
        QTimer.singleShot(2000, lambda: self.setWindowTitle("Processing History"))

    def _move(self, delta: int):
        row = self._list.currentRow()
        dst = row + delta
        if row < 0 or dst < 0 or dst >= self._list.count():
            return
        if self._graph.move(row, dst):
            self._refresh()
            self._list.setCurrentRow(dst)
            self.history_changed.emit()
