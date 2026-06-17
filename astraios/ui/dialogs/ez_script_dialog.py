"""EZ Script Suite dialog — one-click processing presets."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
)

from astraios.core.ez_scripts import list_presets, run_preset


class EZScriptDialog(QDialog):
    """Dialog to select and run an EZ processing preset."""

    def __init__(self, parent, image_provider=None):
        super().__init__(parent)
        self._image_provider = image_provider
        self._result: np.ndarray | None = None
        self.setWindowTitle("EZ Script Suite")
        self.setMinimumWidth(500)
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(12)

        info = QLabel(
            "Select a one-click processing preset to apply to the current image."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        presets = list_presets()
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(presets)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        lay.addWidget(self._preset_combo)

        self._desc = QLabel("")
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet("color: #aaa; padding: 4px;")
        lay.addWidget(self._desc)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        lay.addWidget(self._progress)

        self._log_output = QPlainTextEdit()
        self._log_output.setReadOnly(True)
        self._log_output.setMaximumHeight(120)
        self._log_output.setPlaceholderText("Processing log...")
        lay.addWidget(self._log_output)

        self._btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._btn_box.accepted.connect(self._run)
        self._btn_box.rejected.connect(self.reject)
        lay.addWidget(self._btn_box)

        self._on_preset_changed(presets[0] if presets else "")

    def _on_preset_changed(self, name: str):
        from astraios.core.ez_scripts import REGISTRY

        preset = REGISTRY.get(name)
        if preset:
            steps = ", ".join(s["name"] for s in preset.steps)
            self._desc.setText(f"{preset.description}<br><b>Steps:</b> {steps}")

    def _run(self):
        if self._image_provider is None:
            self._log_output.appendPlainText("No image provider set")
            return

        name = self._preset_combo.currentText()
        img = self._image_provider()
        if img is None:
            self._log_output.appendPlainText("No image to process")
            return

        self._btn_box.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)

        def progress(p, msg):
            self._progress.setValue(int(p * 100))
            self._log_output.appendPlainText(f"[{int(p*100):3d}%] {msg}")

        try:
            result = run_preset(img, name, progress)
            self._log_output.appendPlainText(f"Done. Output shape: {result.shape}")
            self._progress.setValue(100)
            self._result = result
            self.accept()
        except Exception as e:
            self._log_output.appendPlainText(f"Error: {e}")
            self._progress.setVisible(False)
            self._btn_box.setEnabled(True)

    def dialog_result(self) -> np.ndarray | None:
        return self._result
