"""Narrowband Normalization dialog — normalize Ha/OIII/SII channels relative
to each other before palette combination (SHO/HSO/HOS/HOO mapping).

The normalization core is ported from Seti Astro Suite Pro (GPL-3.0, Franklin
Marek); this dialog drives astraios.core.narrowband_normalization.
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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

_SCENARIOS = ["SHO", "HSO", "HOS", "HOO"]

# lightness combo options per scenario, mapping display index -> NBNParams.lightness
_LIGHTNESS_OPTIONS = {
    "HOO": [
        ("Assembled RGB (CIE L*)", 0),
        ("Raw RGB CIE lightness", 1),
        ("Ha", 2),
        ("OIII", 3),
    ],
    "SHO": [
        ("Assembled RGB (CIE L*)", 0),
        ("Raw RGB CIE lightness", 1),
        ("Ha", 2),
        ("SII", 3),
        ("OIII", 4),
    ],
}
_LIGHTNESS_OPTIONS["HSO"] = _LIGHTNESS_OPTIONS["SHO"]
_LIGHTNESS_OPTIONS["HOS"] = _LIGHTNESS_OPTIONS["SHO"]

_BLENDMODES = ["Screen-like", "Add-like", "Linear-Dodge-like"]


class _NormalizeWorker(QThread):
    """Runs normalize_narrowband off the GUI thread (large images)."""

    finished_ok = pyqtSignal(object)  # ndarray
    failed = pyqtSignal(str)

    def __init__(self, ha, oiii, sii, params):
        super().__init__()
        self._ha = ha
        self._oiii = oiii
        self._sii = sii
        self._params = params

    def run(self):
        try:
            from astraios.core.narrowband_normalization import normalize_narrowband

            result = normalize_narrowband(self._ha, self._oiii, self._sii, self._params)
            self.finished_ok.emit(result)
        except Exception as exc:
            log.exception("Narrowband normalization failed")
            self.failed.emit(str(exc))


class _ChannelRow(QHBoxLayout):
    """A single Browse.../path-label row for one narrowband channel."""

    def __init__(self, dialog: "NBNormalizationDialog", label: str, attr: str):
        super().__init__()
        self._dialog = dialog
        self._attr = attr
        lbl = QLabel(label)
        lbl.setFixedWidth(40)
        self._path_label = QLabel("No file loaded")
        self._path_label.setWordWrap(True)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        self.addWidget(lbl)
        self.addWidget(self._path_label, 1)
        self.addWidget(browse_btn)
        self.addWidget(clear_btn)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self._dialog, "Select Channel Image", "",
            "Images (*.fits *.fit *.fts *.tif *.tiff *.png *.jpg *.xisf);;All files (*)",
        )
        if not path:
            return
        from astraios.core.image_io import load_image

        try:
            img = load_image(path)
        except Exception as exc:
            self._dialog._status.setText(f"Could not load {Path(path).name}: {exc}")
            return

        data = img.data.astype(np.float32)
        if data.ndim == 3:
            data = data.mean(axis=0)  # collapse any color channel image to mono

        setattr(self._dialog, self._attr, data)
        self._path_label.setText(Path(path).name)
        self._dialog._update_apply_enabled()

    def _clear(self):
        setattr(self._dialog, self._attr, None)
        self._path_label.setText("No file loaded")
        self._dialog._update_apply_enabled()


class NBNormalizationDialog(QDialog):
    """Normalize Ha/OIII/SII narrowband channels and compose an RGB palette."""

    # Emitted with the composed (3, H, W) RGB ndarray when Apply succeeds.
    result_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Narrowband Normalization")
        self.setMinimumWidth(480)
        self._ha: np.ndarray | None = None
        self._oiii: np.ndarray | None = None
        self._sii: np.ndarray | None = None
        self._worker: _NormalizeWorker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Normalizes Ha/OIII/SII channels relative to each other before "
            "mapping them to a palette (SHO/HSO/HOS/HOO), reproducing SASpro's "
            "PixelMath-derived normalization exactly."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        src = QGroupBox("Channels")
        srow = QVBoxLayout(src)
        srow.addLayout(_ChannelRow(self, "Ha", "_ha"))
        srow.addLayout(_ChannelRow(self, "OIII", "_oiii"))
        srow.addLayout(_ChannelRow(self, "SII", "_sii"))
        lay.addWidget(src)

        form_group = QGroupBox("Normalization")
        form = QFormLayout(form_group)

        self._scenario_combo = QComboBox()
        self._scenario_combo.addItems(_SCENARIOS)
        self._scenario_combo.currentTextChanged.connect(self._on_scenario_changed)
        form.addRow("Scenario", self._row(
            self._scenario_combo,
            "Palette mapping: SHO (SII-R/Ha-G/OIII-B), HSO (Ha-R/SII-G/OIII-B), "
            "HOS (Ha-R/OIII-G/SII-B), or HOO (Ha-R, OIII-G/B, no SII needed).",
        ))

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Linear", "Non-linear (Lab)"])
        self._mode_combo.currentIndexChanged.connect(self._update_enabled_state)
        form.addRow("Mode", self._row(
            self._mode_combo,
            "Linear assembles RGB directly from the normalized channels. "
            "Non-linear (Lab) additionally replaces the CIE L*a*b* lightness "
            "of the assembled RGB with a chosen lightness source.",
        ))

        self._lightness_combo = QComboBox()
        form.addRow("Lightness source", self._row(
            self._lightness_combo,
            "Which lightness drives the Non-linear (Lab) replacement pass. "
            "Only used when Mode is Non-linear (Lab).",
        ))

        self._blackpoint_spin = QDoubleSpinBox()
        self._blackpoint_spin.setRange(0.0, 1.0)
        self._blackpoint_spin.setSingleStep(0.05)
        self._blackpoint_spin.setDecimals(2)
        self._blackpoint_spin.setValue(0.0)
        form.addRow("Blackpoint", self._row(
            self._blackpoint_spin, param_help(
                "Where between each channel's min and median the "
                "normalization blackpoint sits.",
                higher="Pulls the blackpoint toward the median — crushes "
                       "more of the faint background to black.",
                lower="Keeps the blackpoint near the channel minimum — "
                      "preserves more faint signal near black.",
                default="0 = min, 1 = median.",
            ),
        ))

        self._hlrecover_spin = QDoubleSpinBox()
        self._hlrecover_spin.setRange(0.25, 4.0)
        self._hlrecover_spin.setSingleStep(0.05)
        self._hlrecover_spin.setDecimals(2)
        self._hlrecover_spin.setValue(1.0)
        form.addRow("Highlight recover", self._row(
            self._hlrecover_spin, param_help(
                "Highlight recovery scale in the finishing stage.",
                higher="Recovers more highlight detail, pulling blown "
                       "areas back down.",
                lower="Less recovery — highlights stay closer to their "
                      "combined level.",
                default="1.0 is neutral; minimum is 0.25.",
            ),
        ))

        self._hlreduct_spin = QDoubleSpinBox()
        self._hlreduct_spin.setRange(0.25, 4.0)
        self._hlreduct_spin.setSingleStep(0.05)
        self._hlreduct_spin.setDecimals(2)
        self._hlreduct_spin.setValue(1.0)
        form.addRow("Highlight reduction", self._row(
            self._hlreduct_spin, param_help(
                "Highlight reduction strength in the finishing stage.",
                higher="Compresses highlights more aggressively, taming "
                       "bright cores.",
                lower="Less compression — highlights stay brighter.",
                default="1.0 is neutral; minimum is 0.25.",
            ),
        ))

        self._brightness_spin = QDoubleSpinBox()
        self._brightness_spin.setRange(0.25, 4.0)
        self._brightness_spin.setSingleStep(0.05)
        self._brightness_spin.setDecimals(2)
        self._brightness_spin.setValue(1.0)
        form.addRow("Brightness", self._row(
            self._brightness_spin, param_help(
                "Overall brightness multiplier in the finishing stage.",
                higher="Brightens the whole combined result.",
                lower="Darkens the whole combined result.",
                default="1.0 is neutral; minimum is 0.25.",
            ),
        ))

        self._siiboost_spin = QDoubleSpinBox()
        self._siiboost_spin.setRange(0.1, 10.0)
        self._siiboost_spin.setSingleStep(0.1)
        self._siiboost_spin.setDecimals(2)
        self._siiboost_spin.setValue(1.0)
        form.addRow("SII boost", self._row(
            self._siiboost_spin, param_help(
                "SHO/HSO/HOS only — SII normalization boost divisor.",
                higher="Boosts the SII channel's contribution more "
                       "strongly.",
                lower="Reduces SII's contribution to the mix.",
                default="1.0 is neutral.",
            ),
        ))

        self._oiiiboost2_spin = QDoubleSpinBox()
        self._oiiiboost2_spin.setRange(0.1, 10.0)
        self._oiiiboost2_spin.setSingleStep(0.1)
        self._oiiiboost2_spin.setDecimals(2)
        self._oiiiboost2_spin.setValue(1.0)
        form.addRow("OIII boost", self._row(
            self._oiiiboost2_spin, param_help(
                "SHO/HSO/HOS only — OIII normalization boost divisor.",
                higher="Boosts the OIII channel's contribution more "
                       "strongly.",
                lower="Reduces OIII's contribution to the mix.",
                default="1.0 is neutral.",
            ),
        ))

        self._oiiiboost_spin = QDoubleSpinBox()
        self._oiiiboost_spin.setRange(0.1, 10.0)
        self._oiiiboost_spin.setSingleStep(0.1)
        self._oiiiboost_spin.setDecimals(2)
        self._oiiiboost_spin.setValue(1.0)
        form.addRow("OIII boost (HOO)", self._row(
            self._oiiiboost_spin, param_help(
                "HOO only — OIII normalization boost divisor.",
                higher="Boosts the OIII channel's contribution more "
                       "strongly.",
                lower="Reduces OIII's contribution to the mix.",
                default="1.0 is neutral.",
            ),
        ))

        self._blendmode_combo = QComboBox()
        self._blendmode_combo.addItems(_BLENDMODES)
        form.addRow("Ha blend mode", self._row(
            self._blendmode_combo,
            "HOO only -- how Ha is blended into the normalized OIII channel.",
        ))

        self._hablend_spin = QDoubleSpinBox()
        self._hablend_spin.setRange(0.0, 1.0)
        self._hablend_spin.setSingleStep(0.05)
        self._hablend_spin.setDecimals(2)
        self._hablend_spin.setValue(0.6)
        form.addRow("Ha blend", self._row(
            self._hablend_spin, param_help(
                "HOO only — mix ratio for Ha blend mode.",
                higher="Ha contributes more strongly to the blend.",
                lower="Ha contributes less; the blend leans on OIII "
                      "instead.",
                default="0..1 range.",
            ),
        ))

        self._scnr_check = QCheckBox("Apply SCNR (green-cast reduction)")
        form.addRow(self._row(
            self._scnr_check,
            "SHO/HSO/HOS only -- caps the green channel to G = min((R+B)/2, G) "
            "to suppress a green color cast.",
        ))

        lay.addWidget(form_group)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        btns.addWidget(self._apply_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self._on_scenario_changed(self._scenario_combo.currentText())
        self._update_enabled_state()

    @staticmethod
    def _row(widget, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return row

    def _on_scenario_changed(self, scenario: str):
        self._lightness_combo.blockSignals(True)
        self._lightness_combo.clear()
        for label, _val in _LIGHTNESS_OPTIONS.get(scenario, _LIGHTNESS_OPTIONS["SHO"]):
            self._lightness_combo.addItem(label)
        self._lightness_combo.blockSignals(False)
        self._update_enabled_state()
        self._update_apply_enabled()

    def _update_enabled_state(self):
        scenario = self._scenario_combo.currentText()
        is_hoo = scenario == "HOO"
        is_nonlinear = self._mode_combo.currentIndex() == 1

        self._lightness_combo.setEnabled(is_nonlinear)

        self._oiiiboost_spin.setEnabled(is_hoo)
        self._blendmode_combo.setEnabled(is_hoo)
        self._hablend_spin.setEnabled(is_hoo)

        self._siiboost_spin.setEnabled(not is_hoo)
        self._oiiiboost2_spin.setEnabled(not is_hoo)
        self._scnr_check.setEnabled(not is_hoo)

    def _update_apply_enabled(self):
        scenario = self._scenario_combo.currentText()
        if scenario == "HOO":
            ok = self._ha is not None and self._oiii is not None
        else:
            ok = self._ha is not None and self._oiii is not None and self._sii is not None
        # At minimum require at least one channel loaded (validated more
        # precisely -- and with a clear error -- by the core on Apply).
        ok = ok and any(c is not None for c in (self._ha, self._oiii, self._sii))
        self._apply_btn.setEnabled(ok)

    def get_params(self):
        from astraios.core.narrowband_normalization import NBNParams

        scenario = self._scenario_combo.currentText()
        lightness_opts = _LIGHTNESS_OPTIONS.get(scenario, _LIGHTNESS_OPTIONS["SHO"])
        lightness_idx = self._lightness_combo.currentIndex()
        in_range = 0 <= lightness_idx < len(lightness_opts)
        lightness = lightness_opts[lightness_idx][1] if in_range else 0

        return NBNParams(
            scenario=scenario,
            mode=self._mode_combo.currentIndex(),
            lightness=lightness,
            blackpoint=float(self._blackpoint_spin.value()),
            hlrecover=float(self._hlrecover_spin.value()),
            hlreduct=float(self._hlreduct_spin.value()),
            brightness=float(self._brightness_spin.value()),
            blendmode=self._blendmode_combo.currentIndex(),
            hablend=float(self._hablend_spin.value()),
            oiiiboost=float(self._oiiiboost_spin.value()),
            siiboost=float(self._siiboost_spin.value()),
            oiiiboost2=float(self._oiiiboost2_spin.value()),
            scnr=self._scnr_check.isChecked(),
        )

    def _apply(self):
        if not any(c is not None for c in (self._ha, self._oiii, self._sii)):
            self._status.setText("Load at least one narrowband channel first.")
            return
        params = self.get_params()
        self._apply_btn.setEnabled(False)
        self._status.setText("Normalizing...")

        self._worker = _NormalizeWorker(self._ha, self._oiii, self._sii, params)
        self._worker.finished_ok.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_done(self, result):
        self.result_ready.emit(result)
        self._apply_btn.setEnabled(True)
        self._status.setText("Done.")

    def _on_fail(self, msg: str):
        self._apply_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
