"""Generic parameter editor for a processing-history step.

Builds an editable form from a params dict by inspecting each value's type
(bool, int, float, Enum, str, list of numbers), so any registered tool's step
can have its parameters re-edited without a per-tool dialog. Returns a dict with
the same value types (enums stay enum members), which the registry can replay.
"""

from __future__ import annotations

import enum
from typing import Any

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)


def _is_number_list(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


class ParamEditDialog(QDialog):
    """Edit a step's parameters generically, preserving value types."""

    def __init__(self, parent, title: str, params: dict[str, Any]):
        super().__init__(parent)
        self.setWindowTitle(f"Edit: {title}")
        self.setMinimumWidth(340)
        self._params = params
        self._editors: dict[str, Any] = {}

        root = QVBoxLayout(self)
        form = QFormLayout()
        root.addLayout(form)

        for key, value in params.items():
            label = key.replace("_", " ").title()
            widget = self._make_editor(value)
            self._editors[key] = widget
            form.addRow(label, widget)

        if not params:
            form.addRow(QLabel("This step has no editable parameters."))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------ #
    def _make_editor(self, value: Any):
        # bool must be checked before int (bool is an int subclass).
        if isinstance(value, bool):
            w = QCheckBox()
            w.setChecked(value)
            return w
        if isinstance(value, enum.Enum):
            w = QComboBox()
            for member in type(value):
                w.addItem(member.name, member)
            idx = w.findData(value)
            if idx >= 0:
                w.setCurrentIndex(idx)
            return w
        if isinstance(value, int):
            w = QSpinBox()
            w.setRange(-1_000_000, 1_000_000)
            w.setValue(value)
            return w
        if isinstance(value, float):
            w = QDoubleSpinBox()
            w.setRange(-1_000_000.0, 1_000_000.0)
            w.setDecimals(4)
            w.setSingleStep(0.01)
            w.setValue(value)
            return w
        if _is_number_list(value):
            w = QLineEdit(", ".join(str(v) for v in value))
            return w
        if isinstance(value, str):
            return QLineEdit(value)
        # Unknown / unsupported type: show read-only.
        lbl = QLineEdit(str(value))
        lbl.setReadOnly(True)
        return lbl

    def _read_editor(self, original: Any, widget) -> Any:
        if isinstance(original, bool):
            return widget.isChecked()
        if isinstance(original, enum.Enum):
            data = widget.currentData()
            return data if data is not None else original
        if isinstance(original, int):
            return widget.value()
        if isinstance(original, float):
            return widget.value()
        if _is_number_list(original):
            try:
                return [float(p.strip()) for p in widget.text().split(",") if p.strip() != ""]
            except ValueError:
                return original
        if isinstance(original, str):
            return widget.text()
        return original

    def get_params(self) -> dict[str, Any]:
        """Return the edited params, with the same value types as the input."""
        out = dict(self._params)
        for key, widget in self._editors.items():
            out[key] = self._read_editor(self._params.get(key), widget)
        return out
