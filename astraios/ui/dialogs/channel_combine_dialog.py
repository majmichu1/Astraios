"""Channel Combine Dialog — combine mono Ha/OIII/SII/R/G/B channels into color.

Supported palettes:
  RGB        — standard red/green/blue
  HSO (Hubble) — Ha→G, OIII→B, SII→R  (cyan/blue tones)
  SHO        — SII→R, Ha→G, OIII→B   (gold tones, most popular)
  HOO        — Ha→R, OIII→G, OIII→B  (reddish/green)
  HOS        — Ha→R, OIII→G, SII→B
  HOO (mapped) — Ha→R, OIII→G+B mixture
  Custom     — user picks which file goes to R, G, B
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

FILE_FILTERS = "FITS / TIFF (*.fit *.fits *.fts *.xisf *.tif *.tiff);;All Files (*)"

# Palette definitions: list of (output_channel, source_label, weight)
# output_channel: 'R' | 'G' | 'B'
# source_label: user-facing name for the channel slot
PALETTES: dict[str, list[tuple[str, str]]] = {
    "RGB": [("R", "Red (R)"), ("G", "Green (G)"), ("B", "Blue (B)")],
    "SHO (Gold — most popular)": [("R", "SII"), ("G", "Ha"), ("B", "OIII")],
    "HSO (Hubble palette)": [("R", "SII"), ("G", "Ha / OIII blend"), ("B", "OIII")],
    "HOO (Red Ha)": [("R", "Ha"), ("G", "OIII"), ("B", "OIII")],
    "HOS": [("R", "Ha"), ("G", "OIII"), ("B", "SII")],
    "Custom": [("R", "Channel R"), ("G", "Channel G"), ("B", "Channel B")],
}

DEFAULT_PALETTE = "SHO (Gold — most popular)"


class _ChannelRow(QWidget):
    """A single channel input row: label + file picker + weight."""

    def __init__(self, output_channel: str, source_label: str, parent=None):
        super().__init__(parent)
        self.output_channel = output_channel
        self._path: Path | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        colors = {"R": "#cc4444", "G": "#44aa44", "B": "#4466cc"}
        ch_lbl = QLabel(f"<b style='color:{colors.get(output_channel,'#888')}'>{output_channel}</b>")
        ch_lbl.setFixedWidth(14)
        row.addWidget(ch_lbl)

        src_lbl = QLabel(source_label)
        src_lbl.setFixedWidth(110)
        row.addWidget(src_lbl)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("No file selected")
        self._path_edit.setReadOnly(True)
        row.addWidget(self._path_edit, stretch=1)

        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(24)
        btn_browse.clicked.connect(self._browse)
        row.addWidget(btn_browse)

        self._weight_spin = QDoubleSpinBox()
        self._weight_spin.setRange(0.0, 5.0)
        self._weight_spin.setValue(1.0)
        self._weight_spin.setSingleStep(0.05)
        self._weight_spin.setDecimals(2)
        self._weight_spin.setFixedWidth(58)
        self._weight_spin.setToolTip("Channel weight (1.0 = normal)")
        row.addWidget(self._weight_spin)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Channel Image", "", FILE_FILTERS)
        if path:
            self._path = Path(path)
            self._path_edit.setText(path)
            dialog = self.parent()
            while dialog and not isinstance(dialog, ChannelCombineDialog):
                dialog = dialog.parent()
            if dialog:
                dialog._cached_channels.pop(str(path), None)
                dialog._schedule_preview()

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def weight(self) -> float:
        return self._weight_spin.value()

    def set_path(self, path: Path | None):
        self._path = path
        self._path_edit.setText(str(path) if path else "")


class ChannelCombineDialog(QDialog):
    """Dialog for combining mono channel images into a color composite."""

    def __init__(self, current_image=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine Channels")
        self.setMinimumWidth(620)
        self._current_image = current_image  # ImageData for "use current" option
        self._result: np.ndarray | None = None  # float32 (3, H, W) after combine
        self._channel_rows: list[_ChannelRow] = []

        main_layout = QVBoxLayout(self)

        # Palette selector
        palette_group = QGroupBox("Palette / Color Mapping")
        pal_layout = QFormLayout(palette_group)
        self._palette_combo = QComboBox()
        self._palette_combo.addItems(list(PALETTES.keys()))
        self._palette_combo.setCurrentText(DEFAULT_PALETTE)
        self._palette_combo.currentTextChanged.connect(self._rebuild_channels)
        pal_layout.addRow("Palette:", self._palette_combo)
        main_layout.addWidget(palette_group)

        # Channel rows (rebuilt when palette changes)
        self._channels_group = QGroupBox("Channel Files")
        self._channels_layout = QVBoxLayout(self._channels_group)
        lbl_hdr = QHBoxLayout()
        lbl_hdr.addWidget(QLabel("<b>Ch</b>"), stretch=0)
        lbl_hdr.addSpacing(14)
        lbl_hdr.addWidget(QLabel("<b>Source</b>"), stretch=0)
        lbl_hdr.addSpacing(110)
        lbl_hdr.addWidget(QLabel("<b>File</b>"), stretch=1)
        lbl_hdr.addWidget(QLabel("<b>W</b>"))
        self._channels_layout.addLayout(lbl_hdr)
        main_layout.addWidget(self._channels_group)

        # HOO note
        self._hoo_note = QLabel(
            "ℹ For HOO: OIII is used for both G and B. Load the same OIII file in both G and B slots."
        )
        self._hoo_note.setWordWrap(True)
        self._hoo_note.setStyleSheet("color: #aaa; font-size: 11px;")
        self._hoo_note.setVisible(False)
        main_layout.addWidget(self._hoo_note)

        # Live preview
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumHeight(180)
        self._preview_label.setStyleSheet("background: #1e1e1e; border: 1px solid #3c3c3c;")
        main_layout.addWidget(self._preview_label)

        self._cached_channels: dict[str, np.ndarray | None] = {}
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(300)
        self._preview_timer.timeout.connect(self._update_preview)

        # Options
        options_group = QGroupBox("Options")
        opt_layout = QFormLayout(options_group)

        self._bit_depth_combo = QComboBox()
        self._bit_depth_combo.addItems(["Float32 (lossless)", "16-bit FITS", "8-bit TIFF"])
        opt_layout.addRow("Output bit depth:", self._bit_depth_combo)

        self._normalize_check_combo = QComboBox()
        self._normalize_check_combo.addItems(["None", "Per-channel (match median)", "Global (common scale)"])
        self._normalize_check_combo.setCurrentIndex(1)
        self._normalize_check_combo.currentIndexChanged.connect(self._schedule_preview)
        opt_layout.addRow("Normalize channels:", self._normalize_check_combo)

        main_layout.addWidget(options_group)

        # Buttons
        btn_box = QDialogButtonBox()
        self._combine_btn = btn_box.addButton("Combine", QDialogButtonBox.ButtonRole.AcceptRole)
        self._combine_btn.clicked.connect(self._do_combine)
        btn_box.addButton(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)
        main_layout.addWidget(btn_box)

        self._rebuild_channels(DEFAULT_PALETTE)
        QTimer.singleShot(500, self._schedule_preview)

    def _rebuild_channels(self, palette_name: str):
        for row in self._channel_rows:
            row.setParent(None)
        self._channel_rows.clear()
        self._cached_channels.clear()

        definition = PALETTES.get(palette_name, PALETTES["RGB"])
        for out_ch, src_label in definition:
            row = _ChannelRow(out_ch, src_label, self)
            row._weight_spin.valueChanged.connect(self._schedule_preview)
            self._channels_layout.addWidget(row)
            self._channel_rows.append(row)

        self._hoo_note.setVisible("HOO" in palette_name)
        self._schedule_preview()

    def set_current_image(self, image_data):
        self._current_image = image_data

    def _schedule_preview(self):
        self._preview_timer.start()

    def _update_preview(self):
        palette = self._palette_combo.currentText()
        definition = PALETTES.get(palette, PALETTES["RGB"])
        channels: dict[str, np.ndarray] = {}
        channel_counts: dict[str, int] = {}

        for row in self._channel_rows:
            if row.path is None:
                continue
            cache_key = str(row.path)
            if cache_key not in self._cached_channels:
                self._cached_channels[cache_key] = self._load_channel(row.path)
            data = self._cached_channels.get(cache_key)
            if data is None:
                continue
            data = data * row.weight
            if row.output_channel in channels:
                prev_n = channel_counts[row.output_channel]
                channels[row.output_channel] = (
                    channels[row.output_channel] * prev_n + data
                ) / (prev_n + 1)
                channel_counts[row.output_channel] = prev_n + 1
            else:
                channels[row.output_channel] = data
                channel_counts[row.output_channel] = 1

        if len(channels) < 3:
            self._preview_label.clear()
            return

        norm_idx = self._normalize_check_combo.currentIndex()
        if norm_idx == 1:
            medians = {k: float(np.median(v)) for k, v in channels.items()}
            target = float(np.mean(list(medians.values())))
            for k, v in channels.items():
                channels[k] = np.clip(v + target - medians[k], 0, 1)
        elif norm_idx == 2:
            combined_max = max(ch.max() for ch in channels.values())
            if combined_max > 0:
                channels = {k: v / combined_max for k, v in channels.items()}

        try:
            rgb = np.stack([channels["R"], channels["G"], channels["B"]], axis=0)
        except (ValueError, KeyError):
            self._preview_label.clear()
            return

        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        display = np.transpose(rgb, (1, 2, 0))
        h, w = display.shape[:2]
        max_h = max(self._preview_label.height() - 10, 100)
        max_w = max(self._preview_label.width() - 10, 100)
        scale = min(max_w / w, max_h / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)

        display = np.ascontiguousarray((display * 255).astype(np.uint8))
        qimg = QImage(display.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            dw, dh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self._preview_label.setPixmap(pixmap)

    def _load_channel(self, path: Path) -> np.ndarray | None:
        """Load a mono (or color-collapsed) channel as 2D float32."""
        try:
            from astraios.core.image_io import load_image
            img = load_image(str(path))
            d = img.data.astype(np.float32)
            if d.ndim == 3:
                # Take luminance of any color image loaded as a channel
                d = d.mean(axis=0)
            return d
        except Exception as exc:
            log.error("Could not load channel %s: %s", path, exc)
            return None

    def _do_combine(self):
        from PyQt6.QtWidgets import QMessageBox

        channels: dict[str, np.ndarray] = {}  # 'R'/'G'/'B' → 2D array
        channel_counts: dict[str, int] = {}  # count of contributions per output channel

        for row in self._channel_rows:
            if row.path is None:
                QMessageBox.warning(
                    self,
                    "Missing channel",
                    f"No file selected for channel {row.output_channel} ({row.output_channel}).",
                )
                return
            data = self._load_channel(row.path)
            if data is None:
                QMessageBox.critical(self, "Load error", f"Could not load: {row.path}")
                return
            # Apply weight
            data = data * row.weight
            # Accumulate (HOO may have two rows for the same output channel)
            if row.output_channel in channels:
                prev_n = channel_counts[row.output_channel]
                channels[row.output_channel] = (
                    channels[row.output_channel] * prev_n + data
                ) / (prev_n + 1)
                channel_counts[row.output_channel] = prev_n + 1
            else:
                channels[row.output_channel] = data
                channel_counts[row.output_channel] = 1

        # Validate all R, G, B are present
        for ch in ("R", "G", "B"):
            if ch not in channels:
                QMessageBox.warning(self, "Missing channel", f"Channel {ch} has no data.")
                return

        # Normalize channels if requested
        norm_idx = self._normalize_check_combo.currentIndex()
        if norm_idx == 1:
            # Per-channel: match medians
            channels = self._normalize_per_channel(channels)
        elif norm_idx == 2:
            # Global: scale all to [0, 1] jointly
            combined_max = max(ch.max() for ch in channels.values())
            if combined_max > 0:
                channels = {k: v / combined_max for k, v in channels.items()}

        # Stack into (3, H, W) — ensure all same shape
        try:
            rgb = np.stack([channels["R"], channels["G"], channels["B"]], axis=0)
        except ValueError as exc:
            QMessageBox.critical(
                self,
                "Shape mismatch",
                f"Channels have incompatible sizes: {exc}\n"
                "All channels must have the same pixel dimensions.",
            )
            return

        rgb = np.clip(rgb, 0, 1).astype(np.float32)
        self._result = rgb
        self.accept()

    def _normalize_per_channel(self, channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Shift each channel so medians match the mean of all medians."""
        medians = {k: float(np.median(v)) for k, v in channels.items()}
        target = float(np.mean(list(medians.values())))
        out = {}
        for k, v in channels.items():
            shift = target - medians[k]
            out[k] = np.clip(v + shift, 0, 1)
        return out

    def result_data(self) -> np.ndarray | None:
        """Return the combined (3, H, W) float32 array, or None if not run."""
        return self._result
