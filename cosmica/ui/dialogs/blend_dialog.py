"""Image Blend Dialog — combine the current image with a second layer.

The base layer is the currently-open image; the user browses for a second image
(e.g. a stretched star layer) and combines it with a chosen blend mode and
opacity. SCREEN is the default — the standard way to add stars back onto a
starless image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from cosmica.core.image_blend import BlendMode, BlendParams, blend_images
from cosmica.core.image_io import load_image

_MODE_LABELS = [
    ("Screen (add stars/light)", BlendMode.SCREEN),
    ("Normal (opacity mix)", BlendMode.NORMAL),
    ("Add (linear dodge)", BlendMode.ADD),
    ("Subtract", BlendMode.SUBTRACT),
    ("Multiply", BlendMode.MULTIPLY),
    ("Lighten (max)", BlendMode.LIGHTEN),
    ("Darken (min)", BlendMode.DARKEN),
    ("Difference", BlendMode.DIFFERENCE),
    ("Average", BlendMode.AVERAGE),
    ("Overlay", BlendMode.OVERLAY),
]


class BlendDialog(QDialog):
    """Combine the current image with a second image layer."""

    result_ready = pyqtSignal(np.ndarray)  # emits blended image

    def __init__(self, base_image: np.ndarray | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Blend")
        self.setMinimumWidth(420)

        self._base = base_image
        self._layer: np.ndarray | None = None

        layout = QVBoxLayout(self)

        # Blend layer selection
        layer_group = QGroupBox("Blend Layer")
        layer_layout = QHBoxLayout(layer_group)
        self._layer_label = QLabel("No image selected")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        layer_layout.addWidget(self._layer_label, 1)
        layer_layout.addWidget(browse_btn)
        layout.addWidget(layer_group)

        # Mode + opacity
        opts = QGroupBox("Mode")
        opts_layout = QVBoxLayout(opts)
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([label for label, _ in _MODE_LABELS])
        opts_layout.addWidget(self._mode_combo)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity"))
        self._opacity_spin = QDoubleSpinBox()
        self._opacity_spin.setRange(0.0, 1.0)
        self._opacity_spin.setSingleStep(0.05)
        self._opacity_spin.setValue(1.0)
        op_row.addWidget(self._opacity_spin)
        opts_layout.addLayout(op_row)
        layout.addWidget(opts)

        self._status = QLabel("")
        layout.addWidget(self._status)

        # Buttons
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply)
        self._apply_btn.setEnabled(False)
        btn_row.addStretch(1)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self._apply_btn)
        layout.addLayout(btn_row)

    def set_base_image(self, image: np.ndarray) -> None:
        self._base = image

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select blend layer", "",
            "Images (*.fits *.fit *.fts *.tif *.tiff *.png *.jpg *.xisf);;All files (*)",
        )
        if not path:
            return
        try:
            img = load_image(str(path))
            self._layer = img.data.astype(np.float32)
            self._layer_label.setText(Path(path).name)
            self._apply_btn.setEnabled(self._base is not None)
            self._status.setText("")
        except Exception as exc:  # pragma: no cover - UI error path
            self._status.setText(f"Could not load image: {exc}")
            self._layer = None
            self._apply_btn.setEnabled(False)

    def _selected_mode(self) -> BlendMode:
        return _MODE_LABELS[self._mode_combo.currentIndex()][1]

    def _apply(self) -> None:
        if self._base is None or self._layer is None:
            return
        params = BlendParams(mode=self._selected_mode(), opacity=self._opacity_spin.value())
        try:
            result = blend_images(self._base, self._layer, params)
        except Exception as exc:  # pragma: no cover - UI error path
            self._status.setText(f"Blend failed: {exc}")
            return
        self.result_ready.emit(result)
        self.accept()
