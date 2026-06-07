"""Processing Graph dialog — non-destructive editing history."""

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

from cosmica.core.processing_graph import ProcessingGraph


class ProcessingGraphDialog(QDialog):
    """Non-destructive editing history with navigation."""

    graph_changed = pyqtSignal()

    def __init__(self, parent, graph: ProcessingGraph):
        super().__init__(parent)
        self._graph = graph
        self.setWindowTitle("Processing History")
        self.setMinimumSize(450, 400)
        self.setModal(False)
        self._setup_ui()
        self._refresh()
        # Auto-refresh every second to pick up auto-recorded nodes
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)
        self._refresh_timer.timeout.connect(self._refresh_if_changed)
        self._refresh_timer.start()
        self._last_node_count = len(graph.nodes)

    def _setup_ui(self):
        lay = QVBoxLayout(self)

        info = QLabel(
            "Click a step to view the image at that stage. "
            "Changes cascade and invalidate downstream steps."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; padding: 4px;")
        lay.addWidget(info)

        self._history_list = QListWidget()
        self._history_list.setAlternatingRowColors(True)
        self._history_list.currentRowChanged.connect(self._on_select_step)
        lay.addWidget(self._history_list)

        btn_row = QHBoxLayout()
        self._delete_btn = QPushButton("\u2715 Delete Step")
        self._delete_btn.clicked.connect(self._delete_step)
        self._delete_btn.setEnabled(False)
        btn_row.addWidget(self._delete_btn)

        self._toggle_btn = QPushButton("\u2298 Toggle Step")
        self._toggle_btn.clicked.connect(self._toggle_step)
        self._toggle_btn.setEnabled(False)
        btn_row.addWidget(self._toggle_btn)

        self._lock_btn = QPushButton("\U0001f512 Lock Step")
        self._lock_btn.clicked.connect(self._lock_step)
        self._lock_btn.setEnabled(False)
        btn_row.addWidget(self._lock_btn)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)

        lay.addLayout(btn_row)

    def _refresh(self):
        self._history_list.blockSignals(True)
        self._history_list.clear()
        history = self._graph.list_history()
        for entry in history:
            item = QListWidgetItem(entry["display"])
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            tooltip_parts = []
            if entry["dependents"]:
                tooltip_parts.append(f"Dependents: {entry['dependents']}")
            else:
                tooltip_parts.append("Leaf node")
            item.setToolTip("; ".join(tooltip_parts))
            if not entry["enabled"]:
                item.setForeground(Qt.GlobalColor.gray)
            self._history_list.addItem(item)
        self._history_list.blockSignals(False)
        self._last_node_count = len(self._graph.nodes)

    def _refresh_if_changed(self):
        if len(self._graph.nodes) != self._last_node_count:
            self._refresh()

    def _on_select_step(self, row: int):
        has_selection = row >= 0
        self._delete_btn.setEnabled(has_selection)
        self._toggle_btn.setEnabled(has_selection)
        self._lock_btn.setEnabled(has_selection)
        if has_selection:
            node_id = self._history_list.item(row).data(Qt.ItemDataRole.UserRole)
            if node_id and node_id in self._graph.nodes:
                locked = self._graph.nodes[node_id].locked
                self._lock_btn.setText("\U0001f513 Unlock Step" if locked else "\U0001f512 Lock Step")
            self.graph_changed.emit()

    def _delete_step(self):
        row = self._history_list.currentRow()
        if row < 0:
            return
        node_id = self._history_list.item(row).data(Qt.ItemDataRole.UserRole)
        if node_id:
            self._graph.remove_node(node_id)
            self._graph.invalidate_downstream("base")
            self._refresh()
            self.graph_changed.emit()

    def _lock_step(self):
        row = self._history_list.currentRow()
        if row < 0:
            return
        node_id = self._history_list.item(row).data(Qt.ItemDataRole.UserRole)
        if node_id and node_id in self._graph.nodes:
            self._graph.nodes[node_id].locked = not self._graph.nodes[node_id].locked
            self._lock_btn.setText(
                "\U0001f513 Unlock Step" if self._graph.nodes[node_id].locked else "\U0001f512 Lock Step"
            )
            self._refresh()
            self.graph_changed.emit()

    def _toggle_step(self):
        row = self._history_list.currentRow()
        if row < 0:
            return
        node_id = self._history_list.item(row).data(Qt.ItemDataRole.UserRole)
        if node_id and node_id in self._graph.nodes:
            self._graph.nodes[node_id].enabled = not self._graph.nodes[node_id].enabled
            self._graph.invalidate_downstream(node_id)
            self._refresh()
            self.graph_changed.emit()
