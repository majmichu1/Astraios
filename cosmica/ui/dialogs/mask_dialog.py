"""Mask Creation Dialog — create luminance and range masks with live preview."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from cosmica.core.masks import (
    Mask,
    blur_mask,
    create_luminance_mask,
    create_range_mask,
    invert_mask,
)

if TYPE_CHECKING:
    pass


class MaskDialog(QDialog):
    """Dialog for creating masks with live preview."""

    mask_created = pyqtSignal(object)  # Mask

    def __init__(self, image_data: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Mask")
        self.setMinimumSize(650, 500)
        self._image_data = image_data
        self._current_mask: Mask | None = None

        layout = QVBoxLayout(self)

        # Type selector
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Mask type:"))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Luminance Mask", "Range Mask"])
        self._type_combo.currentIndexChanged.connect(self._update_preview)
        type_row.addWidget(self._type_combo)
        layout.addLayout(type_row)

        # Name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        from PyQt6.QtWidgets import QLineEdit
        self._name_edit = QLineEdit("New Mask")
        name_row.addWidget(self._name_edit)
        layout.addLayout(name_row)

        # Parameters group
        self._params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(self._params_group)

        # Low threshold
        row = QHBoxLayout()
        row.addWidget(QLabel("Low:"))
        self._low_slider = QSlider(Qt.Orientation.Horizontal)
        self._low_slider.setRange(0, 1000)
        self._low_slider.setValue(0)
        self._low_slider.valueChanged.connect(self._update_preview)
        self._low_label = QLabel("0.000")
        row.addWidget(self._low_slider)
        row.addWidget(self._low_label)
        params_layout.addLayout(row)

        # High threshold
        row = QHBoxLayout()
        row.addWidget(QLabel("High:"))
        self._high_slider = QSlider(Qt.Orientation.Horizontal)
        self._high_slider.setRange(0, 1000)
        self._high_slider.setValue(1000)
        self._high_slider.valueChanged.connect(self._update_preview)
        self._high_label = QLabel("1.000")
        row.addWidget(self._high_slider)
        row.addWidget(self._high_label)
        params_layout.addLayout(row)

        # Channel selector (for range mask)
        row = QHBoxLayout()
        row.addWidget(QLabel("Channel:"))
        self._channel_combo = QComboBox()
        self._channel_combo.addItems(["Luminance", "Red", "Green", "Blue"])
        self._channel_combo.currentIndexChanged.connect(self._update_preview)
        row.addWidget(self._channel_combo)
        self._channel_row = row
        params_layout.addLayout(row)

        # Blur radius
        row = QHBoxLayout()
        row.addWidget(QLabel("Softness:"))
        self._blur_spin = QDoubleSpinBox()
        self._blur_spin.setRange(0.0, 50.0)
        self._blur_spin.setValue(0.0)
        self._blur_spin.setSingleStep(0.5)
        self._blur_spin.setToolTip("Gaussian blur radius to soften mask edges")
        self._blur_spin.valueChanged.connect(self._update_preview)
        row.addWidget(self._blur_spin)
        params_layout.addLayout(row)

        # Invert checkbox
        from PyQt6.QtWidgets import QCheckBox
        self._invert_check = QCheckBox("Invert mask")
        self._invert_check.stateChanged.connect(self._update_preview)
        params_layout.addWidget(self._invert_check)

        layout.addWidget(self._params_group)

        # Preview
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumHeight(200)
        self._preview_label.setStyleSheet("background: #1e1e1e; border: 1px solid #3c3c3c;")
        layout.addWidget(self._preview_label, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_apply = QPushButton("Create Mask")
        self._btn_apply.clicked.connect(self._on_create)
        btn_row.addWidget(self._btn_apply)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

        self._update_preview()

    def _get_low(self) -> float:
        return self._low_slider.value() / 1000.0

    def _get_high(self) -> float:
        return self._high_slider.value() / 1000.0

    def _update_preview(self):
        low = self._get_low()
        high = self._get_high()
        self._low_label.setText(f"{low:.3f}")
        self._high_label.setText(f"{high:.3f}")

        mask_type_idx = self._type_combo.currentIndex()
        name = self._name_edit.text() or "Mask"

        if mask_type_idx == 0:
            # Luminance mask
            mask = create_luminance_mask(self._image_data, low=low, high=high, name=name)
        else:
            # Range mask
            ch = self._channel_combo.currentIndex() - 1  # -1 = luminance
            mask = create_range_mask(self._image_data, channel=ch, low=low, high=high, name=name)

        # Apply blur
        blur_radius = self._blur_spin.value()
        if blur_radius > 0:
            mask = blur_mask(mask, radius=blur_radius)

        # Invert
        if self._invert_check.isChecked():
            mask = invert_mask(mask)

        self._current_mask = mask

        # Render preview
        display = mask.to_display()
        h, w, _ = display.shape

        # Scale to fit preview area
        max_h = max(self._preview_label.height() - 10, 100)
        max_w = max(self._preview_label.width() - 10, 100)
        scale = min(max_w / w, max_h / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)

        display = np.ascontiguousarray(display)
        qimg = QImage(display.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            dw, dh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self._preview_label.setPixmap(pixmap)

    def _on_create(self):
        if self._current_mask is not None:
            self._current_mask.name = self._name_edit.text() or "Mask"
            self.mask_created.emit(self._current_mask)
        self.accept()

    @property
    def mask(self) -> Mask | None:
        return self._current_mask
