"""Star Mask Dialog — generate a standalone star mask."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from cosmica.core.star_reduction import create_star_mask


class StarMaskDialog(QDialog):
    """Dialog for generating a star mask with adjustable parameters."""

    mask_ready = pyqtSignal(object)  # emits Mask

    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Star Mask")
        self.setMinimumWidth(380)
        self._image = image_data

        layout = QVBoxLayout(self)

        group = QGroupBox("Star Mask Parameters")
        g_layout = QVBoxLayout(group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Sensitivity (sigma):"))
        self._sensitivity_spin = QDoubleSpinBox()
        self._sensitivity_spin.setRange(1.0, 20.0)
        self._sensitivity_spin.setValue(5.0)
        self._sensitivity_spin.setSingleStep(0.5)
        self._sensitivity_spin.setToolTip("Detection threshold — lower detects more stars")
        row1.addWidget(self._sensitivity_spin)
        g_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Max stars:"))
        self._max_stars_spin = QSpinBox()
        self._max_stars_spin.setRange(10, 5000)
        self._max_stars_spin.setValue(500)
        row2.addWidget(self._max_stars_spin)
        g_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Softness:"))
        self._softness_spin = QDoubleSpinBox()
        self._softness_spin.setRange(0.0, 20.0)
        self._softness_spin.setValue(5.0)
        self._softness_spin.setSingleStep(0.5)
        self._softness_spin.setToolTip("Gaussian feathering radius")
        row3.addWidget(self._softness_spin)
        g_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Scale:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.5, 5.0)
        self._scale_spin.setValue(1.5)
        self._scale_spin.setSingleStep(0.1)
        self._scale_spin.setToolTip("Star blob size relative to FWHM")
        row4.addWidget(self._scale_spin)
        g_layout.addLayout(row4)

        layout.addWidget(group)

        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: #80c0ff; font-size: 11px;")
        layout.addWidget(self._info_label)

        btn_row = QHBoxLayout()
        generate_btn = QPushButton("Generate Mask")
        generate_btn.clicked.connect(self._generate)
        btn_row.addWidget(generate_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _generate(self):
        mask = create_star_mask(
            self._image,
            sensitivity=self._sensitivity_spin.value(),
            max_stars=self._max_stars_spin.value(),
            softness=self._softness_spin.value(),
            scale=self._scale_spin.value(),
        )
        coverage = float(mask.data.mean()) * 100
        self._info_label.setText(f"Mask generated — {coverage:.1f}% coverage")
        self.mask_ready.emit(mask)
