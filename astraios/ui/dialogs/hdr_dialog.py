"""HDR Composition Dialog — merge multiple exposures."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
)

from astraios.core.hdr import HDRMethod, HDRParams, hdr_compose
from astraios.core.image_io import load_image
from astraios.ui.dialogs.dialog_workers import stop_worker


class _HDRWorker(QThread):
    """Loads the exposures and composes the HDR off the GUI thread — both
    used to run synchronously in the dialog's button handler."""

    status = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, paths: list[Path], params: HDRParams):
        super().__init__()
        self._paths = paths
        self._params = params

    def run(self):
        try:
            n = len(self._paths)
            images = []
            for i, p in enumerate(self._paths):
                if self.isInterruptionRequested():
                    return
                self.status.emit(f"Loading image {i + 1}/{n}...")
                images.append(load_image(str(p)).data)
            self.status.emit("Composing HDR...")
            result = hdr_compose(images, self._params)
            self.done.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class HDRDialog(QDialog):
    """Dialog for HDR composition from multiple exposure images."""

    result_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDR Composition")
        self.setMinimumSize(450, 350)

        self._image_paths: list[Path] = []
        self._worker: _HDRWorker | None = None

        layout = QVBoxLayout(self)

        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.addItems(["Mertens Fusion", "Weighted Average"])
        method_row.addWidget(self._method_combo)
        layout.addLayout(method_row)

        # Image list
        layout.addWidget(QLabel("Exposure Images:"))
        self._image_list = QListWidget()
        layout.addWidget(self._image_list)

        # Add/Remove buttons
        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add Images...")
        add_btn.clicked.connect(self._add_images)
        btn_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)
        layout.addLayout(btn_row)

        # Contrast weight
        cw_row = QHBoxLayout()
        cw_row.addWidget(QLabel("Contrast weight:"))
        self._contrast_spin = QDoubleSpinBox()
        self._contrast_spin.setRange(0.0, 5.0)
        self._contrast_spin.setValue(1.0)
        self._contrast_spin.setSingleStep(0.1)
        cw_row.addWidget(self._contrast_spin)
        layout.addLayout(cw_row)

        # Run / Cancel buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Compose HDR")
        self._run_btn.clicked.connect(self._run)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._run_btn)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        layout.addWidget(self._status)

    def _add_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Exposure Images", "",
            "All Supported (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;All (*)",
        )
        for p in paths:
            path = Path(p)
            if path not in self._image_paths:
                self._image_paths.append(path)
                self._image_list.addItem(path.name)

    def _remove_selected(self):
        for item in self._image_list.selectedItems():
            idx = self._image_list.row(item)
            self._image_list.takeItem(idx)
            self._image_paths.pop(idx)

    def _run(self):
        if len(self._image_paths) < 2:
            self._status.setText("Need at least 2 images")
            return

        method_map = {0: HDRMethod.MERTENS, 1: HDRMethod.WEIGHTED_AVERAGE}
        params = HDRParams(
            method=method_map.get(self._method_combo.currentIndex(), HDRMethod.MERTENS),
            contrast_weight=self._contrast_spin.value(),
        )

        self._status.setText("Loading images...")
        self._worker = _HDRWorker(list(self._image_paths), params)
        self._worker.status.connect(self._status.setText)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_done(self, result):
        self.result_ready.emit(result)
        self._status.setText("HDR composition complete")
        self.accept()

    def _on_failed(self, message: str):
        self._status.setText(f"Error: {message}")

    def reject(self) -> None:
        stop_worker(self)
        super().reject()

    def closeEvent(self, event) -> None:
        stop_worker(self)
        super().closeEvent(event)
