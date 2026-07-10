"""Image Combine dialog.

Drives astraios.core.image_combine (pixel arithmetic between two images —
average/add/subtract/blend/multiply/divide/screen/overlay/difference/min/max
with independent per-input weights), ported from Seti Astro Suite Pro
(GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

# (label, CombineOperation) — human-readable names for every supported op.
_OP_LABELS = [
    ("Average", "average"),
    ("Add (linear dodge)", "add"),
    ("Subtract", "subtract"),
    ("Blend (weighted mix)", "blend"),
    ("Multiply", "multiply"),
    ("Divide", "divide"),
    ("Screen", "screen"),
    ("Overlay", "overlay"),
    ("Difference", "difference"),
    ("Min (darkest)", "min"),
    ("Max (lightest)", "max"),
]


class _Worker(QThread):
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image_a, image_b, params):
        super().__init__()
        self._a, self._b, self._params = image_a, image_b, params

    def run(self):
        try:
            from astraios.core.image_combine import combine_images
            self.done.emit(combine_images(self._a, self._b, self._params))
        except Exception as exc:
            log.exception("Image combine failed")
            self.failed.emit(str(exc))


class ImageCombineDialog(QDialog):
    """Combine the current image (A) with a second image (B) via pixel arithmetic."""

    result_ready = pyqtSignal(object)

    def __init__(self, image_a: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Combine")
        self.setMinimumWidth(440)
        self._a = image_a
        self._b: np.ndarray | None = None
        self._worker: _Worker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Combines the current image (A) with a second image (B) using "
            "pixel arithmetic — average, add, subtract, multiply, screen, "
            "and more. Mono/color operands broadcast automatically; both "
            "images must share the same pixel dimensions."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        b_row = QHBoxLayout()
        self._b_edit = QLineEdit()
        self._b_edit.setPlaceholderText("Load image B...")
        self._b_edit.setReadOnly(True)
        browse = QPushButton("Load...")
        browse.clicked.connect(self._browse)
        b_row.addWidget(self._b_edit, 1)
        b_row.addWidget(browse)
        lay.addLayout(b_row)

        form = QFormLayout()
        self._op = QComboBox()
        self._op.addItems([label for label, _ in _OP_LABELS])
        form.addRow(*self._r("Operation", self._op,
                    "Which pixel-arithmetic operation combines A and B. "
                    "Average/Add/Blend are additive; Subtract/Difference "
                    "compare levels; Multiply/Screen/Overlay/Min/Max are "
                    "the usual compositing modes."))

        self._weight_a = QDoubleSpinBox()
        self._weight_a.setRange(0.0, 10.0)
        self._weight_a.setSingleStep(0.05)
        self._weight_a.setValue(1.0)
        form.addRow(*self._r("Weight A", self._weight_a, param_help(
            "Multiplier applied to image A before the operation.",
            higher="Image A contributes more strongly to the result.",
            lower="Image A contributes less; 0 removes it entirely.",
            default="1.0 leaves A unscaled.",
        )))

        self._weight_b = QDoubleSpinBox()
        self._weight_b.setRange(0.0, 10.0)
        self._weight_b.setSingleStep(0.05)
        self._weight_b.setValue(1.0)
        form.addRow(*self._r("Weight B", self._weight_b, param_help(
            "Multiplier applied to image B before the operation.",
            higher="Image B contributes more strongly to the result.",
            lower="Image B contributes less; 0 removes it entirely.",
            default="1.0 leaves B unscaled. For an opacity cross-fade with "
                    "Blend, set Weight A = 1 - alpha and Weight B = alpha.",
        )))

        self._clip = QCheckBox("Clip result to [0, 1]")
        self._clip.setChecked(True)
        self._clip.setToolTip(
            "<qt>Clamp the combined result into the normal image range. "
            "Ignored when Rescale is checked.</qt>")
        form.addRow(self._clip)

        self._rescale = QCheckBox("Rescale result to [0, 1]")
        self._rescale.setToolTip(
            "<qt>Instead of clamping, min-max stretch the raw result back "
            "into [0, 1]. Takes priority over Clip when both are "
            "checked — use when weights/operations push values well "
            "outside range and a full dynamic-range remap is preferred "
            "over hard clipping.</qt>")
        form.addRow(self._rescale)
        lay.addLayout(form)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._apply_btn = QPushButton("Combine")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        btns.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    @staticmethod
    def _r(label, widget, tip):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return label, row

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Image B", "",
            "Images (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png)"
        )
        if not path:
            return
        from astraios.core.image_io import load_image
        try:
            img = load_image(path)
        except Exception as exc:
            self._status.setText(f"Could not load: {exc}")
            return
        b = img.data
        if b.shape[-2:] != self._a.shape[-2:]:
            self._status.setText(
                f"Image B size {b.shape[-2:]} does not match image A "
                f"{self._a.shape[-2:]}."
            )
            return
        self._b = b
        from pathlib import Path
        self._b_edit.setText(Path(path).name)
        self._apply_btn.setEnabled(True)
        self._status.setText("")

    def _selected_operation(self):
        from astraios.core.image_combine import CombineOperation
        return CombineOperation(_OP_LABELS[self._op.currentIndex()][1])

    def _apply(self):
        from astraios.core.image_combine import ImageCombineParams

        if self._b is None:
            return
        params = ImageCombineParams(
            operation=self._selected_operation(),
            weight_a=float(self._weight_a.value()),
            weight_b=float(self._weight_b.value()),
            clip=self._clip.isChecked(),
            rescale=self._rescale.isChecked(),
        )
        self._apply_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        self._status.setText("Combining...")
        self._worker = _Worker(self._a, self._b, params)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        if not isinstance(result, np.ndarray):
            self._status.setText("No result produced.")
            self._apply_btn.setEnabled(True)
            return
        self.result_ready.emit(result)
        self.accept()

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._apply_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
