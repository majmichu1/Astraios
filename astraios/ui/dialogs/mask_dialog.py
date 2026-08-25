"""Mask Creation Dialog — create luminance and range masks with live preview."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
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

from astraios.core.masks import (
    Mask,
    blur_mask,
    create_luminance_mask,
    create_range_mask,
    invert_mask,
)
from astraios.ui.widgets.ui_kit import form_label, param_help

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

        # Debounce slider drags: a full-resolution mask rebuild + blur per
        # tick used to lock the GUI for seconds on large images.
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._update_preview)

        layout = QVBoxLayout(self)

        # Type selector
        type_row = QHBoxLayout()
        type_row.addWidget(form_label("Mask type:", param_help(
            "How the mask decides which pixels a tool may touch.",
            how="Luminance Mask follows brightness smoothly: bright areas get "
                "weight 1, dark sky weight 0, with a gentle taper at the Low "
                "and High limits. Range Mask keeps only pixels whose brightness "
                "falls between Low and High, with a hard edge you can soften.",
            tip="Use a luminance mask to sharpen or denoise the nebula without "
                "touching the background, and its inverse to work on the "
                "background alone.",
        )))
        self._type_combo = QComboBox()
        self._type_combo.addItems(["Luminance Mask", "Range Mask"])
        self._type_combo.currentIndexChanged.connect(self._schedule_preview)
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
        row.addWidget(form_label("Low:", param_help(
            "Pixels darker than this are left out of the mask (weight 0).",
            higher="More of the faint background is protected; the mask "
                   "covers only the brighter structure.",
            lower="Faint areas join the mask and will be processed too.",
            default="0 includes everything from black upward.",
        )))
        self._low_slider = QSlider(Qt.Orientation.Horizontal)
        self._low_slider.setRange(0, 1000)
        self._low_slider.setValue(0)
        self._low_slider.valueChanged.connect(self._schedule_preview)
        self._low_label = QLabel("0.000")
        row.addWidget(self._low_slider)
        row.addWidget(self._low_label)
        params_layout.addLayout(row)

        # High threshold
        row = QHBoxLayout()
        row.addWidget(form_label("High:", param_help(
            "Pixels brighter than this are left out of the mask (weight 0).",
            higher="Bright cores and stars stay inside the mask.",
            lower="Star cores and the brightest nebula are protected from "
                  "the tool, which stops sharpening or stretching from "
                  "blowing them out.",
            default="1 includes everything up to white.",
        )))
        self._high_slider = QSlider(Qt.Orientation.Horizontal)
        self._high_slider.setRange(0, 1000)
        self._high_slider.setValue(1000)
        self._high_slider.valueChanged.connect(self._schedule_preview)
        self._high_label = QLabel("1.000")
        row.addWidget(self._high_slider)
        row.addWidget(self._high_label)
        params_layout.addLayout(row)

        # Channel selector (for range mask)
        row = QHBoxLayout()
        row.addWidget(form_label("Channel:", param_help(
            "Which channel's brightness the mask is built from.",
            how="Luminance is the usual choice. Pick a single colour to mask "
                "by that channel alone, for example Red to isolate Ha "
                "emission in an OSC image.",
        )))
        self._channel_combo = QComboBox()
        self._channel_combo.addItems(["Luminance", "Red", "Green", "Blue"])
        self._channel_combo.currentIndexChanged.connect(self._schedule_preview)
        row.addWidget(self._channel_combo)
        self._channel_row = row
        params_layout.addLayout(row)

        # Blur radius
        row = QHBoxLayout()
        row.addWidget(form_label("Softness:", param_help(
            "Blurs the edge of the mask so processed and protected areas "
            "merge without a visible line.",
            how="A Gaussian blur of this radius, in pixels, is applied to "
                "the mask after it is built.",
            higher="Smoother, wider transitions; small features are no "
                   "longer isolated.",
            lower="Crisper selection; at 0 the edge is hard and can show as "
                  "a halo after strong processing.",
            default="5 to 15 px is a good start on a full-size image.",
        )))
        self._blur_spin = QDoubleSpinBox()
        self._blur_spin.setRange(0.0, 50.0)
        self._blur_spin.setValue(0.0)
        self._blur_spin.setSingleStep(0.5)
        self._blur_spin.setToolTip("Gaussian blur radius to soften mask edges")
        self._blur_spin.valueChanged.connect(self._schedule_preview)
        row.addWidget(self._blur_spin)
        params_layout.addLayout(row)

        # Invert checkbox
        from PyQt6.QtWidgets import QCheckBox
        self._invert_check = QCheckBox("Invert mask")
        self._invert_check.setToolTip("<qt>" + param_help(
            "Swaps protected and processed areas.",
            how="What the mask selected becomes protected and everything "
                "else becomes editable. A luminance mask inverted is a "
                "background mask.",
        ) + "</qt>")
        self._invert_check.stateChanged.connect(self._schedule_preview)
        params_layout.addWidget(self._invert_check)

        layout.addWidget(self._params_group)

        # Preview
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumHeight(200)
        self._preview_label.setStyleSheet("background: #0d1117; border: 1px solid #30363d;")
        layout.addWidget(self._preview_label, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._btn_apply = QPushButton("Create Mask")
        self._btn_apply.setDefault(True)  # the dialog's primary action: Enter runs it, drawn in accent
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

    def _schedule_preview(self, *_args):
        """Instant label feedback; the heavy mask recompute is debounced."""
        self._low_label.setText(f"{self._get_low():.3f}")
        self._high_label.setText(f"{self._get_high():.3f}")
        self._debounce_timer.start()

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
