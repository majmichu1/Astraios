"""SFCC (Spectral Flux Color Calibration) dialog.

Drives astraios.core.sfcc, ported/adapted from Seti Astro Suite Pro's SFCC
(GPL-3.0, Franklin Marek). See astraios.core.sfcc's module docstring for
what physics was ported faithfully vs. reduced (bundled representative
filter/sensor curves instead of SASpro's proprietary FITS curve library).
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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

_NONE_LABEL = "(None)"


class _SFCCWorker(QThread):
    """Runs apply_sfcc off the GUI thread (star detection + catalog query)."""

    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, params, wcs_header, catalog_stars):
        super().__init__()
        self._image = image
        self._params = params
        self._wcs_header = wcs_header
        self._catalog_stars = catalog_stars

    def run(self):
        try:
            from astraios.core.sfcc import apply_sfcc

            result = apply_sfcc(
                self._image,
                params=self._params,
                wcs_header=self._wcs_header,
                catalog_stars=self._catalog_stars,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.done.emit(result)
        except Exception as exc:
            log.exception("SFCC failed")
            self.failed.emit(str(exc))


class SFCCDialog(QDialog):
    """Spectral Flux Color Calibration: per-channel color correction from
    filter transmission x sensor QE x stellar-flux integration.

    Requires either a plate-solved WCS (``wcs_header``) so it can query Gaia
    DR3 itself, or a pre-resolved ``catalog_stars`` list (e.g. an existing
    WCS star overlay) — see astraios.core.sfcc.apply_sfcc.
    """

    result_ready = pyqtSignal(object)

    def __init__(
        self,
        image: np.ndarray,
        parent=None,
        wcs_header: dict | None = None,
        catalog_stars: list[tuple[float, float, float]] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spectral Flux Color Calibration (SFCC)")
        self.setMinimumWidth(520)
        self._image = image
        self._wcs_header = wcs_header
        self._catalog_stars = catalog_stars
        self._worker: _SFCCWorker | None = None
        self._custom_filter_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._custom_sensor_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        from astraios.core.sfcc import (
            FILTER_CURVE_NAMES,
            SENSOR_QE_NAMES,
            WHITE_REFERENCE_NAMES,
        )

        self._filter_names = list(FILTER_CURVE_NAMES)
        self._sensor_names = list(SENSOR_QE_NAMES)

        lay = QVBoxLayout(self)

        is_color = image is not None and image.ndim == 3 and image.shape[0] == 3
        if not is_color:
            lay.addWidget(QLabel(
                "SFCC requires a 3-channel RGB image — the loaded image is mono "
                "or has an unexpected shape."
            ))

        if not (self._wcs_header or self._catalog_stars):
            warn = QLabel(
                "No plate solution / catalog stars supplied — run Solve & "
                "Calibrate (PCC) or SPCC first so SFCC can locate reference "
                "stars by RA/Dec."
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #cc8800;")
            lay.addWidget(warn)

        # ── Filters / sensor group ────────────────────────────────────────
        filt_group = QGroupBox("Filter transmission x Sensor QE")
        filt_form = QFormLayout(filt_group)

        self._filter_r_combo = self._filter_combo(default="Broadband-R (generic LRGB interference)")
        filt_form.addRow("R filter", self._row(self._filter_r_combo, param_help(
            "Filter transmission curve integrated against the stellar flux "
            "for the red channel.",
            how="S_expected_R = integral(flux_star(lambda) x T_filter_R(lambda) "
                "x QE_sensor(lambda)) dlambda.",
            tip="Bundled curves are representative/approximate, not digitized "
                "vendor data — load a real curve via 'Load custom filter CSV...' "
                "for production accuracy.",
        )))

        self._filter_g_combo = self._filter_combo(default="Broadband-G (generic LRGB interference)")
        filt_form.addRow("G filter", self._row(self._filter_g_combo, param_help(
            "Filter transmission curve for the green channel (the reference "
            "channel — its scale is always fixed at the fitted value; R and B "
            "are corrected relative to it).",
        )))

        self._filter_b_combo = self._filter_combo(default="Broadband-B (generic LRGB interference)")
        filt_form.addRow("B filter", self._row(self._filter_b_combo, param_help(
            "Filter transmission curve for the blue channel.",
        )))

        self._lp1_combo = self._filter_combo(default=None, include_none=True)
        filt_form.addRow("LP/cut filter 1", self._row(self._lp1_combo, param_help(
            "Optional extra filter (e.g. a UV/IR-cut or light-pollution filter) "
            "stacked multiplicatively into every channel's system response.",
        )))

        self._lp2_combo = self._filter_combo(default=None, include_none=True)
        filt_form.addRow("LP/cut filter 2", self._row(self._lp2_combo, param_help(
            "A second stackable filter, same as LP/cut filter 1.",
        )))

        self._sensor_combo = QComboBox()
        self._sensor_combo.addItems(self._sensor_names)
        idx = self._sensor_combo.findText("Generic CMOS back-illuminated (Sony IMX-class)")
        if idx >= 0:
            self._sensor_combo.setCurrentIndex(idx)
        filt_form.addRow("Sensor QE", self._row(self._sensor_combo, param_help(
            "Sensor quantum-efficiency curve, multiplied into every channel's "
            "system response alongside the filter transmission.",
            tip="Manufacturer QE curves are usually optimistic and measured "
                "at room temperature, not your sensor's operating temperature "
                "— treat this as a starting point, not ground truth.",
        )))

        load_filter_btn = QPushButton("Load custom filter CSV...")
        load_filter_btn.clicked.connect(self._load_custom_filter_csv)
        load_sensor_btn = QPushButton("Load custom sensor CSV...")
        load_sensor_btn.clicked.connect(self._load_custom_sensor_csv)
        load_row = QHBoxLayout()
        load_row.addWidget(load_filter_btn)
        load_row.addWidget(load_sensor_btn)
        filt_form.addRow(load_row)

        lay.addWidget(filt_group)

        # ── Reference / catalog group ──────────────────────────────────────
        ref_group = QGroupBox("Reference star + catalog")
        ref_form = QFormLayout(ref_group)

        self._white_ref_combo = QComboBox()
        self._white_ref_combo.addItems(WHITE_REFERENCE_NAMES)
        ref_form.addRow("White reference", self._row(self._white_ref_combo, param_help(
            "Stellar type used only for a diagnostic reference color ratio "
            "(not fed into the fit itself) — shows what 'neutral gray' looks "
            "like under the current filter/QE choice.",
            default="G2V (Sun-like) is the conventional astrophotography white point.",
        )))

        self._catalog_combo = QComboBox()
        self._catalog_combo.addItem("Gaia DR3 (Vizier, online)", "vizier_gaia_dr3")
        self._catalog_combo.addItem("Gaia offline catalog (no BP/RP — unsupported)", "offline_gaia")
        ref_form.addRow("Catalog", self._row(self._catalog_combo, param_help(
            "Where reference star positions + colors come from. Only used "
            "when no catalog_stars were already supplied (e.g. from a "
            "previous PCC/SPCC run).",
            tip="The offline catalog only stores G magnitude, not BP/RP color, "
                "so it cannot drive this tool yet — it will raise a clear error "
                "if selected without catalog_stars.",
        )))

        self._search_radius_spin = QDoubleSpinBox()
        self._search_radius_spin.setRange(0.05, 5.0)
        self._search_radius_spin.setSingleStep(0.05)
        self._search_radius_spin.setValue(0.5)
        ref_form.addRow("Search radius (deg)", self._row(self._search_radius_spin, param_help(
            "Cone-search radius around the plate-solved field center for "
            "catalog stars.",
            higher="More candidate stars (better statistics) but slower queries.",
            lower="Faster, but may not find enough usable stars.",
        )))

        self._mag_limit_spin = QDoubleSpinBox()
        self._mag_limit_spin.setRange(5.0, 20.0)
        self._mag_limit_spin.setSingleStep(0.5)
        self._mag_limit_spin.setValue(16.0)
        ref_form.addRow("Magnitude limit", self._row(self._mag_limit_spin, param_help(
            "Faintest catalog G magnitude to consider.",
            higher="More (fainter) candidate stars.",
            lower="Only bright, high-SNR stars — more reliable per-star photometry.",
        )))

        lay.addWidget(ref_group)

        # ── Detection group ─────────────────────────────────────────────────
        det_group = QGroupBox("Star detection + photometry")
        det_form = QFormLayout(det_group)

        self._detection_sigma_spin = QDoubleSpinBox()
        self._detection_sigma_spin.setRange(1.0, 50.0)
        self._detection_sigma_spin.setSingleStep(0.5)
        self._detection_sigma_spin.setValue(8.0)
        det_form.addRow("Detection sigma", self._row(self._detection_sigma_spin, param_help(
            "Star-detection threshold in noise-sigma above background.",
            higher="Stricter — only bright, unambiguous stars are detected.",
            lower="More detections, including faint/noisy ones.",
        )))

        self._max_stars_spin = QSpinBox()
        self._max_stars_spin.setRange(10, 5000)
        self._max_stars_spin.setValue(300)
        det_form.addRow("Max stars detected", self._row(self._max_stars_spin, param_help(
            "Cap on the number of detected stars considered for catalog matching.",
        )))

        self._match_radius_spin = QDoubleSpinBox()
        self._match_radius_spin.setRange(0.5, 20.0)
        self._match_radius_spin.setSingleStep(0.5)
        self._match_radius_spin.setValue(3.0)
        det_form.addRow("Match radius (px)", self._row(self._match_radius_spin, param_help(
            "How close a detected star must be to a WCS-projected catalog "
            "position to count as a match.",
        )))

        self._saturation_spin = QDoubleSpinBox()
        self._saturation_spin.setRange(0.5, 1.0)
        self._saturation_spin.setSingleStep(0.01)
        self._saturation_spin.setValue(0.98)
        det_form.addRow("Saturation limit", self._row(self._saturation_spin, param_help(
            "Stars whose peak pixel (any channel) meets or exceeds this "
            "normalized value are excluded — a saturated star's color ratio "
            "is meaningless.",
        )))

        self._aperture_spin = QDoubleSpinBox()
        self._aperture_spin.setRange(1.0, 50.0)
        self._aperture_spin.setSingleStep(0.5)
        self._aperture_spin.setValue(5.0)
        det_form.addRow("Aperture radius (px)", self._row(self._aperture_spin, param_help(
            "Photometry aperture radius for measuring instrumental star flux.",
        )))

        self._max_phot_spin = QSpinBox()
        self._max_phot_spin.setRange(8, 5000)
        self._max_phot_spin.setValue(500)
        det_form.addRow("Max stars used", self._row(self._max_phot_spin, param_help(
            "Cap to the brightest N matched stars used for the color fit.",
        )))

        self._min_stars_spin = QSpinBox()
        self._min_stars_spin.setRange(3, 1000)
        self._min_stars_spin.setValue(8)
        det_form.addRow("Min stars required", self._row(self._min_stars_spin, param_help(
            "Fit aborts with an error if fewer usable stars are found than this.",
        )))

        self._neutralize_chk = QCheckBox("Neutralize background")
        self._neutralize_chk.setChecked(False)
        det_form.addRow(self._row(self._neutralize_chk, param_help(
            "Subtracts a per-channel background offset after the color "
            "correction so the sky background is neutral gray.",
        )))

        lay.addWidget(det_group)

        # ── Progress / status / buttons ──────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._run_btn = QPushButton("Run SFCC")
        self._run_btn.setEnabled(bool(is_color))
        self._run_btn.clicked.connect(self._run)
        btns.addWidget(self._run_btn)
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

    # ── helpers ──────────────────────────────────────────────────────────

    def _filter_combo(self, default: str | None, include_none: bool = False) -> QComboBox:
        combo = QComboBox()
        if include_none:
            combo.addItem(_NONE_LABEL)
        combo.addItems(self._filter_names)
        if default is not None:
            idx = combo.findText(default)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        return combo

    @staticmethod
    def _row(widget, tip: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return row

    def _load_custom_filter_csv(self):
        self._load_custom_csv(
            self._custom_filter_curves,
            combos=(self._filter_r_combo, self._filter_g_combo, self._filter_b_combo,
                    self._lp1_combo, self._lp2_combo),
        )

    def _load_custom_sensor_csv(self):
        self._load_custom_csv(self._custom_sensor_curves, combos=(self._sensor_combo,))

    def _load_custom_csv(self, target: dict, combos: tuple[QComboBox, ...]):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select 2-column CSV (wavelength_nm, response)", "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        from astraios.core.sfcc import load_curve_csv

        try:
            curve = load_curve_csv(path)
        except Exception as exc:
            self._status.setText(f"Could not load curve: {exc}")
            return

        name = f"Custom: {Path(path).stem}"
        target[name] = curve
        for combo in combos:
            if combo.findText(name) < 0:
                combo.addItem(name)
        self._status.setText(f"Loaded '{name}' — now selectable above.")

    def _selected_or_none(self, combo: QComboBox) -> str | None:
        text = combo.currentText()
        return None if text == _NONE_LABEL else text

    def get_params(self):
        from astraios.core.sfcc import SFCCParams

        return SFCCParams(
            filter_r=self._filter_r_combo.currentText(),
            filter_g=self._filter_g_combo.currentText(),
            filter_b=self._filter_b_combo.currentText(),
            lp_filter_1=self._selected_or_none(self._lp1_combo),
            lp_filter_2=self._selected_or_none(self._lp2_combo),
            sensor=self._sensor_combo.currentText(),
            white_reference=self._white_ref_combo.currentText(),
            custom_filter_curves=dict(self._custom_filter_curves) or None,
            custom_sensor_curves=dict(self._custom_sensor_curves) or None,
            catalog=self._catalog_combo.currentData(),
            search_radius_deg=float(self._search_radius_spin.value()),
            mag_limit=float(self._mag_limit_spin.value()),
            detection_sigma=float(self._detection_sigma_spin.value()),
            max_stars_detect=int(self._max_stars_spin.value()),
            match_radius_px=float(self._match_radius_spin.value()),
            saturation_threshold=float(self._saturation_spin.value()),
            max_phot_stars=int(self._max_phot_spin.value()),
            min_stars=int(self._min_stars_spin.value()),
            aperture_radius_px=float(self._aperture_spin.value()),
            neutralize_background=self._neutralize_chk.isChecked(),
        )

    def _run(self):
        if self._image is None or self._image.ndim != 3 or self._image.shape[0] != 3:
            self._status.setText("SFCC requires a 3-channel RGB image.")
            return
        if not (self._wcs_header or self._catalog_stars):
            self._status.setText(
                "No plate solution / catalog stars available — run PCC/SPCC first."
            )
            return

        params = self.get_params()
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._status.setText("Running SFCC...")

        self._worker = _SFCCWorker(self._image, params, self._wcs_header, self._catalog_stars)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_progress(self, fraction: float, message: str):
        self._progress.setValue(int(fraction * 100))
        self._status.setText(message)

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText("SFCC complete.")
        self.result_ready.emit(result)

    def _on_fail(self, msg: str):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
