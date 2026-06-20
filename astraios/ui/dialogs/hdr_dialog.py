"""HDR Composition Dialog — merge multiple exposures."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
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


class HDRDialog(QDialog):
    """Dialog for HDR composition from multiple exposure images."""

    result_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDR Composition")
        self.setMinimumSize(450, 350)

        self._image_paths: list[Path] = []

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

        self._status.setText("Loading images...")
        try:
            images = [load_image(str(p)).data for p in self._image_paths]
        except Exception as e:
            self._status.setText(f"Error loading: {e}")
            return

        method_map = {0: HDRMethod.MERTENS, 1: HDRMethod.WEIGHTED_AVERAGE}
        params = HDRParams(
            method=method_map.get(self._method_combo.currentIndex(), HDRMethod.MERTENS),
            contrast_weight=self._contrast_spin.value(),
        )

        self._status.setText("Composing HDR...")
        try:
            result = hdr_compose(images, params)
            self.result_ready.emit(result)
            self._status.setText("HDR composition complete")
            self.accept()
        except Exception as e:
            self._status.setText(f"Error: {e}")
