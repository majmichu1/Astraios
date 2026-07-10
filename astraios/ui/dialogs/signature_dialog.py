"""Signature / watermark dialog — sign the finished image.

The compositing core is ported from Seti Astro Suite Pro (GPL-3.0, Franklin
Marek); this dialog drives astraios.core.signature.
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

_POSITIONS = [
    ("Bottom right", "bottom_right"), ("Bottom left", "bottom_left"),
    ("Bottom center", "bottom_center"), ("Top right", "top_right"),
    ("Top left", "top_left"), ("Top center", "top_center"),
    ("Center", "center"), ("Center left", "center_left"),
    ("Center right", "center_right"),
]

_FONTS = [
    ("Simplex", "simplex"), ("Duplex", "duplex"), ("Triplex", "triplex"),
    ("Complex", "complex"), ("Complex small", "complex_small"),
    ("Plain", "plain"), ("Script simplex", "script_simplex"),
    ("Script complex", "script_complex"),
]


class SignatureDialog(QDialog):
    """Insert a text or logo signature into the current image."""

    # Emitted with the signed ndarray when the user applies.
    result_ready = pyqtSignal(object)

    def __init__(self, image_data: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Signature / Watermark")
        self.setMinimumWidth(440)
        self._data = image_data
        self._color = (1.0, 1.0, 1.0)

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Signs the finished image with your name or logo before "
            "export. Applied as a normal (undoable) processing step."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        mode_row = QHBoxLayout()
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Text", "Image / logo"])
        self._mode_combo.currentIndexChanged.connect(self._update_mode)
        mode_row.addWidget(QLabel("Type"))
        mode_row.addWidget(self._mode_combo, 1)
        mode_row.addWidget(help_dot(
            "Text renders your name with an outline for readability; "
            "Image places a PNG logo (transparency respected)."))
        lay.addLayout(mode_row)

        self._text_group = QGroupBox("Text")
        tform = QFormLayout(self._text_group)
        self._text_edit = QLineEdit("Imaged by ...")
        tform.addRow("Text", self._text_edit)
        self._font_combo = QComboBox()
        for label, _key in _FONTS:
            self._font_combo.addItem(label)
        tform.addRow("Font", self._font_combo)
        self._size_spin = QSpinBox()
        self._size_spin.setRange(8, 300)
        self._size_spin.setValue(28)
        tform.addRow("Size (px)", self._size_spin)
        style_row = QHBoxLayout()
        self._bold_check = QCheckBox("Bold")
        self._italic_check = QCheckBox("Italic")
        self._color_btn = QPushButton("Color...")
        self._color_btn.clicked.connect(self._pick_color)
        style_row.addWidget(self._bold_check)
        style_row.addWidget(self._italic_check)
        style_row.addWidget(self._color_btn)
        style_row.addStretch()
        tform.addRow("Style", style_row)
        self._outline_spin = QSpinBox()
        self._outline_spin.setRange(0, 10)
        self._outline_spin.setValue(2)
        row = QHBoxLayout()
        row.addWidget(self._outline_spin)
        row.addWidget(help_dot(param_help(
            "Dark outline thickness around the letters so the text stays "
            "readable over stars.",
            higher="A thicker, more visible outline.",
            lower="A thinner outline; 0 disables it.",
        )))
        row.addStretch()
        tform.addRow("Outline", row)
        lay.addWidget(self._text_group)

        self._image_group = QGroupBox("Logo image")
        iform = QFormLayout(self._image_group)
        logo_row = QHBoxLayout()
        self._logo_edit = QLineEdit()
        self._logo_edit.setPlaceholderText("PNG with transparency works best")
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_logo)
        logo_row.addWidget(self._logo_edit, 1)
        logo_row.addWidget(btn)
        iform.addRow("File", logo_row)
        lay.addWidget(self._image_group)

        place = QGroupBox("Placement")
        pform = QFormLayout(place)
        self._pos_combo = QComboBox()
        for label, _key in _POSITIONS:
            self._pos_combo.addItem(label)
        pform.addRow("Position", self._pos_combo)
        self._margin_spin = QSpinBox()
        self._margin_spin.setRange(0, 500)
        self._margin_spin.setValue(24)
        pform.addRow("Margin (px)", self._margin_spin)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(1.0, 400.0)
        self._scale_spin.setValue(100.0)
        self._scale_spin.setSuffix(" %")
        pform.addRow("Scale", self._scale_spin)
        self._rotation_spin = QDoubleSpinBox()
        self._rotation_spin.setRange(-180.0, 180.0)
        self._rotation_spin.setValue(0.0)
        pform.addRow("Rotation", self._rotation_spin)
        self._opacity_spin = QDoubleSpinBox()
        self._opacity_spin.setRange(0.0, 100.0)
        self._opacity_spin.setValue(90.0)
        self._opacity_spin.setSuffix(" %")
        pform.addRow("Opacity", self._opacity_spin)
        lay.addWidget(place)

        self._status = QLabel("")
        lay.addWidget(self._status)
        btns = QHBoxLayout()
        apply_btn = QPushButton("Apply Signature")
        apply_btn.clicked.connect(self._apply)
        btns.addWidget(apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self._update_mode()

    def _update_mode(self):
        text_mode = self._mode_combo.currentIndex() == 0
        self._text_group.setVisible(text_mode)
        self._image_group.setVisible(not text_mode)

    def _pick_color(self):
        from PyQt6.QtGui import QColor
        c = QColorDialog.getColor(QColor.fromRgbF(*self._color), self)
        if c.isValid():
            self._color = (c.redF(), c.greenF(), c.blueF())

    def _browse_logo(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Logo", "", "Images (*.png *.jpg *.jpeg)"
        )
        if path:
            self._logo_edit.setText(path)

    def get_params(self):
        from astraios.core.signature import SignatureParams

        text_mode = self._mode_combo.currentIndex() == 0
        return SignatureParams(
            mode="text" if text_mode else "image",
            text=self._text_edit.text(),
            font_face=_FONTS[self._font_combo.currentIndex()][1],
            font_size=float(self._size_spin.value()),
            bold=self._bold_check.isChecked(),
            italic=self._italic_check.isChecked(),
            color=self._color,
            outline_width=int(self._outline_spin.value()),
            image_path=self._logo_edit.text().strip() or None,
            position=_POSITIONS[self._pos_combo.currentIndex()][1],
            margin_x=int(self._margin_spin.value()),
            margin_y=int(self._margin_spin.value()),
            scale=float(self._scale_spin.value()),
            rotation=float(self._rotation_spin.value()),
            opacity=float(self._opacity_spin.value()),
        )

    def _apply(self):
        from astraios.core.signature import insert_signature

        params = self.get_params()
        if params.mode == "image" and not params.image_path:
            self._status.setText("Choose a logo file first.")
            return
        try:
            result = insert_signature(self._data, params)
        except Exception as exc:
            log.exception("Signature failed")
            self._status.setText(f"Failed: {exc}")
            return
        self.result_ready.emit(result)
        self.accept()
