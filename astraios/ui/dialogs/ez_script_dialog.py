"""EZ Script Suite dialog — one-click processing presets."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
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
from astraios.ui.dialogs.dialog_workers import stop_worker


class _EZWorker(QThread):
    """Runs the preset pipeline off the GUI thread — it used to run
    synchronously in the Ok handler, freezing the whole UI for minutes."""

    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, img: np.ndarray, name: str):
        super().__init__()
        self._img = img
        self._name = name
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        def _progress(p: float, msg: str) -> None:
            if self._cancelled:
                raise InterruptedError("cancelled")
            self.progress.emit(p, msg)

        try:
            result = run_preset(self._img, self._name, _progress)
        except InterruptedError:
            return
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        if not self._cancelled:
            self.done.emit(result)


class EZScriptDialog(QDialog):
    """Dialog to select and run an EZ processing preset."""

    def __init__(self, parent, image_provider=None):
        super().__init__(parent)
        self._image_provider = image_provider
        self._result: np.ndarray | None = None
        self._worker: _EZWorker | None = None
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

        self._worker = _EZWorker(np.asarray(img), name)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, p: float, msg: str):
        self._progress.setValue(int(p * 100))
        self._log_output.appendPlainText(f"[{int(p * 100):3d}%] {msg}")

    def _on_done(self, result):
        self._log_output.appendPlainText(f"Done. Output shape: {result.shape}")
        self._progress.setValue(100)
        self._result = result
        self.accept()

    def _on_failed(self, message: str):
        self._log_output.appendPlainText(f"Error: {message}")
        self._progress.setVisible(False)
        self._btn_box.setEnabled(True)

    def reject(self) -> None:
        stop_worker(self)
        super().reject()

    def closeEvent(self, event) -> None:
        stop_worker(self)
        super().closeEvent(event)

    def dialog_result(self) -> np.ndarray | None:
        return self._result
