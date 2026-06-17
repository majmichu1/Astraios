"""Export dialog — save image as TIFF, PNG, JPEG with format options."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

FORMAT_FILTERS = {
    ".tif": "TIFF 16-bit (*.tif *.tiff)",
    ".png": "PNG 16-bit (*.png)",
    ".jpg": "JPEG 95% (*.jpg *.jpeg)",
    ".fits": "FITS 32-bit float (*.fits *.fts)",
    ".xisf": "XISF 32-bit float (*.xisf)",
}


class ExportDialog(QDialog):
    """Dialog for exporting images to various formats with options."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Image")
        self.setMinimumWidth(420)
        self._output_path = ""
        self._setup_ui()

    @property
    def output_path(self) -> str:
        return self._output_path

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File path
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select output file...")
        self._path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_path)
        path_layout.addWidget(QLabel("File:"))
        path_layout.addWidget(self._path_edit, 1)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Options section
        layout.addWidget(QLabel("Options"))

        bit_layout = QHBoxLayout()
        bit_layout.addWidget(QLabel("Bit depth:"))
        self._bit_combo = QComboBox()
        self._bit_combo.addItems(["8-bit", "16-bit"])
        self._bit_combo.setCurrentIndex(1)
        bit_layout.addWidget(self._bit_combo)
        layout.addLayout(bit_layout)

        qual_layout = QHBoxLayout()
        qual_layout.addWidget(QLabel("JPEG quality:"))
        self._quality_spin = QSpinBox()
        self._quality_spin.setRange(1, 100)
        self._quality_spin.setValue(95)
        self._quality_spin.setSuffix("%")
        qual_layout.addWidget(self._quality_spin)
        layout.addLayout(qual_layout)

        # Info label
        info_label = QLabel(
            "<span style='color: #888;'>TIFF and PNG support 16-bit for "
            "high-quality output. JPEG is always 8-bit.</span>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _browse_path(self):
        # Build filter string
        filters = ";;".join(FORMAT_FILTERS.values())
        default_filter = FORMAT_FILTERS[".tif"]

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Image As",
            "",
            filters,
            default_filter,
        )
        if path:
            # Ensure extension matches filter
            if "TIFF" in selected_filter and not path.lower().endswith((".tif", ".tiff")):
                path += ".tif"
            elif "PNG" in selected_filter and not path.lower().endswith(".png"):
                path += ".png"
            elif "JPEG" in selected_filter and not path.lower().endswith((".jpg", ".jpeg")):
                path += ".jpg"
            elif "FITS" in selected_filter and not path.lower().endswith((".fits", ".fts")):
                path += ".fits"
            elif "XISF" in selected_filter and not path.lower().endswith(".xisf"):
                path += ".xisf"

            self._path_edit.setText(path)
            self._output_path = path

    def _accept(self):
        if not self._output_path:
            self._browse_path()
            if not self._output_path:
                return
        self.accept()

    def get_export_params(self) -> dict:
        """Return export parameters dict for use by the caller."""
        bit_text = self._bit_combo.currentText()
        return {
            "path": Path(self._output_path),
            "bit_depth": 16 if "16" in bit_text else 8,
            "jpeg_quality": self._quality_spin.value(),
        }
