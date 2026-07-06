"""Perfect Palette Picker dialog — blend Ha/OIII/SII into a false-color palette.

Drives astraios.core.palette_picker, ported from Seti Astro Suite Pro's
Perfect Palette Picker (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from astraios.core.palette_picker import (
    PALETTE_DESCRIPTIONS,
    PALETTE_LABELS,
    Palette,
    PalettePickerParams,
)
from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

_ROLES = ("Ha", "OIII", "SII")


def _to_mono(data: np.ndarray) -> np.ndarray:
    """Collapse a possibly-color array to a single (H, W) plane, like
    NarrowbandDialog._load_filter does for loaded filter images."""
    a = np.asarray(data, dtype=np.float32)
    if a.ndim == 3:
        if a.shape[0] == 1:
            return a[0]
        return np.mean(a, axis=0).astype(np.float32)
    return a


class _Worker(QThread):
    """Runs apply_palette off the GUI thread (large images)."""

    done = pyqtSignal(object)  # ndarray
    failed = pyqtSignal(str)

    def __init__(self, ha, oiii, sii, stars, params):
        super().__init__()
        self._ha, self._oiii, self._sii = ha, oiii, sii
        self._stars = stars
        self._params = params

    def run(self):
        try:
            from astraios.core.palette_picker import apply_palette

            result = apply_palette(self._ha, self._oiii, self._sii, self._stars, self._params)
            self.done.emit(result)
        except Exception as exc:
            log.exception("Perfect Palette Picker failed")
            self.failed.emit(str(exc))


class PalettePickerDialog(QDialog):
    """Blend Ha/OIII/SII narrowband channels into a named false-color palette.

    The image passed in is the primary narrowband input (its role -- Ha,
    OIII, or SII -- is chosen below); the remaining channel(s) are loaded
    from file and size-checked against it.
    """

    result_ready = pyqtSignal(object)

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Perfect Palette Picker")
        self.setMinimumWidth(500)

        self._primary = _to_mono(image)
        self._shape = self._primary.shape
        # role name -> loaded mono ndarray (or None); one entry always
        # points at self._primary (the role picked in _role_combo).
        self._channels: dict[str, np.ndarray | None] = {"Ha": None, "OIII": None, "SII": None}
        self._stars: np.ndarray | None = None
        self._worker: _Worker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Blends narrowband Ha/OIII/SII channels into a named false-color palette "
            "(Hubble SHO, bicolor HOO, dynamic Foraxx, and more). The current image "
            "supplies one channel; load the other one or two from file."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        # ---------------- Channels ----------------
        chan_group = QGroupBox("Channels")
        chan_form = QFormLayout(chan_group)

        self._role_combo = QComboBox()
        self._role_combo.addItems(_ROLES)
        self._role_combo.currentTextChanged.connect(self._on_role_changed)
        chan_form.addRow("Current image is", self._row(
            self._role_combo,
            "Which narrowband filter the currently-open image is. The other "
            "two channels are loaded from file below.",
        ))

        self._chan_labels: dict[str, QLabel] = {}
        self._chan_buttons: dict[str, QPushButton] = {}
        for role in _ROLES:
            label = QLabel("")
            btn = QPushButton(f"Load {role}...")
            btn.clicked.connect(lambda _=None, r=role: self._load_channel(r))
            row = QHBoxLayout()
            row.addWidget(label, 1)
            row.addWidget(btn)
            chan_form.addRow(f"{role}:", row)
            self._chan_labels[role] = label
            self._chan_buttons[role] = btn

        lay.addWidget(chan_group)
        self._on_role_changed(self._role_combo.currentText())

        # ---------------- Palette ----------------
        palette_group = QGroupBox("Palette")
        palette_layout = QVBoxLayout(palette_group)

        combo_row = QHBoxLayout()
        self._palette_combo = QComboBox()
        for p in Palette:
            self._palette_combo.addItem(PALETTE_LABELS[p], p)
        self._palette_combo.setCurrentIndex(0)
        self._palette_combo.currentIndexChanged.connect(self._on_palette_changed)
        combo_row.addWidget(self._palette_combo, 1)
        combo_row.addWidget(help_dot(
            "<br>".join(f"<b>{PALETTE_LABELS[p]}</b>: {PALETTE_DESCRIPTIONS[p]}" for p in Palette)
        ))
        palette_layout.addLayout(combo_row)

        self._palette_desc = QLabel("")
        self._palette_desc.setWordWrap(True)
        self._palette_desc.setStyleSheet("color: #8b949e;")
        palette_layout.addWidget(self._palette_desc)

        # Custom weight matrix (only shown/used for Palette.CUSTOM).
        self._custom_group = QGroupBox("Custom weights (rows = R/G/B, columns = Ha/OIII/SII)")
        custom_grid = QGridLayout(self._custom_group)
        self._custom_spins: list[list[QDoubleSpinBox]] = []
        default = np.eye(3, dtype=np.float32)
        for col, name in enumerate(("Ha", "OIII", "SII")):
            custom_grid.addWidget(QLabel(name), 0, col + 1)
        for grid_row, name in enumerate(("R", "G", "B")):
            custom_grid.addWidget(QLabel(name), grid_row + 1, 0)
            spins = []
            for col in range(3):
                spin = QDoubleSpinBox()
                spin.setRange(-2.0, 2.0)
                spin.setSingleStep(0.05)
                spin.setDecimals(3)
                spin.setValue(float(default[grid_row, col]))
                custom_grid.addWidget(spin, grid_row + 1, col + 1)
                spins.append(spin)
            self._custom_spins.append(spins)
        custom_help_row = QHBoxLayout()
        custom_help_row.addStretch()
        custom_help_row.addWidget(help_dot(param_help(
            "Free-form mix of Ha/OIII/SII into R/G/B, for palettes not covered "
            "by the preset list.",
            how="Each output row is weight_Ha*Ha + weight_OIII*OIII + weight_SII*SII.",
            default="Identity (R=Ha, G=OIII, B=SII) — the same as the HOS preset.",
        )))
        custom_grid.addLayout(custom_help_row, 4, 0, 1, 4)
        palette_layout.addWidget(self._custom_group)

        lay.addWidget(palette_group)
        self._on_palette_changed(self._palette_combo.currentIndex())

        # ---------------- Options ----------------
        opt_group = QGroupBox("Options")
        opt_form = QFormLayout(opt_group)

        self._linear_check = QCheckBox("Linear input (stretch each channel before building)")
        self._linear_check.setChecked(True)
        self._linear_check.toggled.connect(self._update_enabled_state)
        opt_form.addRow(self._row(
            self._linear_check,
            param_help(
                "Applies a statistical (target-median) stretch to each channel "
                "before mixing colors.",
                how="Narrowband subs are usually still linear when loaded here; "
                "mixing raw linear data produces a near-black, uninformative preview.",
                higher="Leave checked for linear/unstretched subs (the normal case).",
                lower="Uncheck only if you're feeding in already-stretched channels.",
                default="Checked.",
            ),
        ))

        self._target_median_spin = QDoubleSpinBox()
        self._target_median_spin.setRange(0.05, 0.6)
        self._target_median_spin.setSingleStep(0.01)
        self._target_median_spin.setValue(0.25)
        opt_form.addRow("Target median", self._row(
            self._target_median_spin,
            param_help(
                "Background brightness each channel is stretched to before mixing.",
                higher="Brighter, more aggressively stretched background.",
                lower="Darker, more conservative stretch.",
                default="0.25, matching Perfect Palette Picker's default.",
            ),
        ))

        self._normalize_check = QCheckBox("Normalize to peak")
        self._normalize_check.setChecked(True)
        opt_form.addRow(self._row(
            self._normalize_check,
            "Divides the finished palette by its own brightest pixel so the "
            "preview is never fully clipped.",
        ))

        lay.addWidget(opt_group)

        # ---------------- Stars layer (optional, Astraios addition) ----------------
        stars_group = QGroupBox("Star color (optional)")
        stars_form = QFormLayout(stars_group)
        stars_row = QHBoxLayout()
        self._stars_label = QLabel("No layer loaded.")
        self._stars_label.setWordWrap(True)
        stars_btn = QPushButton("Load stars/broadband layer...")
        stars_btn.clicked.connect(self._load_stars)
        stars_row.addWidget(self._stars_label, 1)
        stars_row.addWidget(stars_btn)
        stars_form.addRow(self._row_layout(
            stars_row,
            "An RGB (or OSC) frame — typically a stars-only extraction — to "
            "screen-blend over the finished palette so stars keep natural "
            "color instead of the narrowband false-color mix.",
        ))

        self._stars_opacity_spin = QDoubleSpinBox()
        self._stars_opacity_spin.setRange(0.0, 1.0)
        self._stars_opacity_spin.setSingleStep(0.05)
        self._stars_opacity_spin.setValue(0.0)
        stars_form.addRow("Blend opacity", self._row(
            self._stars_opacity_spin,
            param_help(
                "Strength of the star-color screen blend.",
                higher="Stars look closer to their true broadband color.",
                lower="Stars keep more of the narrowband false color.",
                default="0.0 (no blend) until a layer is loaded.",
            ),
        ))
        lay.addWidget(stars_group)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._apply_btn = QPushButton("Build Palette")
        self._apply_btn.clicked.connect(self._apply)
        btns.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self._update_enabled_state()

    # ---------------- helpers ----------------

    @staticmethod
    def _row(widget, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return row

    @staticmethod
    def _row_layout(inner: QHBoxLayout, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addLayout(inner, 1)
        row.addWidget(help_dot(tip))
        return row

    def _update_enabled_state(self):
        self._target_median_spin.setEnabled(self._linear_check.isChecked())

    def _on_role_changed(self, role: str):
        for r in _ROLES:
            is_primary = r == role
            self._chan_buttons[r].setEnabled(not is_primary)
            if is_primary:
                self._chan_labels[r].setText("(current image)")
                self._channels[r] = self._primary
            elif self._channels[r] is self._primary:
                # was the primary role before switching; clear it back out
                self._channels[r] = None
                self._chan_labels[r].setText("Not loaded.")

    def _on_palette_changed(self, index: int):
        palette = self._palette_combo.itemData(index)
        self._palette_desc.setText(PALETTE_DESCRIPTIONS.get(palette, ""))
        self._custom_group.setVisible(palette == Palette.CUSTOM)

    def _load_channel(self, role: str):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {role} File", "",
            "Images (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;All files (*)",
        )
        if not path:
            return
        from astraios.core.image_io import load_image

        try:
            img = load_image(path)
        except Exception as exc:
            self._status.setText(f"Could not load {role}: {exc}")
            return

        data = _to_mono(img.data)
        if data.shape != self._shape:
            self._status.setText(
                f"Size mismatch: {role} is {data.shape[1]}x{data.shape[0]}, but the "
                f"reference channel is {self._shape[1]}x{self._shape[0]}. They must match."
            )
            return

        self._channels[role] = data
        self._chan_labels[role].setText(Path(path).name)
        self._status.setText("")

    def _load_stars(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Stars/Broadband Layer", "",
            "Images (*.fit *.fits *.fts *.xisf *.tif *.tiff *.png);;All files (*)",
        )
        if not path:
            return
        from astraios.core.image_io import load_image

        try:
            img = load_image(path)
        except Exception as exc:
            self._status.setText(f"Could not load stars layer: {exc}")
            return

        data = np.asarray(img.data, dtype=np.float32)
        if data.shape[-2:] != self._shape:
            self._status.setText(
                f"Size mismatch: stars layer is {data.shape[-1]}x{data.shape[-2]}, but "
                f"the reference channel is {self._shape[1]}x{self._shape[0]}. They must match."
            )
            return

        self._stars = data
        self._stars_label.setText(Path(path).name)
        if self._stars_opacity_spin.value() <= 0:
            self._stars_opacity_spin.setValue(1.0)
        self._status.setText("")

    def _get_params(self) -> PalettePickerParams:
        palette = self._palette_combo.currentData()
        custom = np.array(
            [[spin.value() for spin in row] for row in self._custom_spins], dtype=np.float32
        )
        return PalettePickerParams(
            palette=palette,
            custom_weights=custom,
            linear_input=self._linear_check.isChecked(),
            target_median=float(self._target_median_spin.value()),
            normalize=self._normalize_check.isChecked(),
            stars_opacity=float(self._stars_opacity_spin.value()),
        )

    def _apply(self):
        ha, oiii, sii = self._channels["Ha"], self._channels["OIII"], self._channels["SII"]
        if oiii is None:
            self._status.setText("OIII is required (load it, or use it as the current image).")
            return
        if ha is None and sii is None:
            self._status.setText("At least one of Ha or SII is required (in addition to OIII).")
            return

        params = self._get_params()
        stars = self._stars if self._stars_opacity_spin.value() > 0 else None

        self._apply_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        self._status.setText("Building palette...")

        self._worker = _Worker(ha, oiii, sii, stars, params)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._apply_btn.setEnabled(True)
        if not isinstance(result, np.ndarray):
            self._status.setText("No result produced.")
            return
        self._status.setText("Done.")
        self.result_ready.emit(result)

    def _on_fail(self, msg: str):
        self._progress.setVisible(False)
        self._apply_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
