"""Alt/Az Field Rotation dialog.

Drives astraios.core.field_rotation (the rate at which the star field spins
underneath an Alt/Az-mounted telescope while it tracks a target — the
reason alt-az rigs need a de-rotator or short subs near the zenith), ported
from Seti Astro Suite Pro (GPL-3.0, Franklin Marek).
"""

from __future__ import annotations

import logging
import math

from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtWidgets import (
    QDateTimeEdit,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


class FieldRotationDialog(QDialog):
    """Compute Alt/Az field-rotation rate, total rotation, and parallactic angle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Alt/Az Field Rotation Calculator")
        self.setMinimumSize(560, 560)
        self.result = None  # exposed for scripted verification

        outer = QVBoxLayout(self)
        intro = QLabel(
            "Computes how fast the star field rotates under an Alt/Az-mounted "
            "telescope tracking a target — the effect that limits sub-exposure "
            "length (or requires a field de-rotator) as a target approaches the "
            "zenith. Rate = 15.04 x cos(lat) x cos(az) / cos(alt) arcsec/sec."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        outer.addWidget(intro)

        form = QFormLayout()

        self._ra = QDoubleSpinBox()
        self._ra.setRange(0.0, 360.0)
        self._ra.setDecimals(4)
        self._ra.setSuffix(" deg")
        form.addRow(*self._r("Target RA", self._ra,
                    "Right ascension of the target, in degrees (ICRS/J2000)."))

        self._dec = QDoubleSpinBox()
        self._dec.setRange(-90.0, 90.0)
        self._dec.setDecimals(4)
        self._dec.setSuffix(" deg")
        form.addRow(*self._r("Target Dec", self._dec,
                    "Declination of the target, in degrees (ICRS/J2000)."))

        self._lat = QDoubleSpinBox()
        self._lat.setRange(-90.0, 90.0)
        self._lat.setDecimals(4)
        self._lat.setSuffix(" deg")
        form.addRow(*self._r("Observer latitude", self._lat,
                    "Observing site latitude, north positive. Rate is zero at "
                    "the geographic pole — an Alt/Az mount there behaves like "
                    "an equatorial one."))

        self._lon = QDoubleSpinBox()
        self._lon.setRange(-180.0, 180.0)
        self._lon.setDecimals(4)
        self._lon.setSuffix(" deg")
        form.addRow(*self._r("Observer longitude", self._lon,
                    "Observing site longitude, east positive."))

        self._time = QDateTimeEdit(QDateTime.currentDateTimeUtc())
        self._time.setCalendarPopup(True)
        self._time.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self._time.setTimeSpec(Qt.TimeSpec.UTC)
        form.addRow(*self._r("Observation time (UTC)", self._time,
                    "Date and time of the exposure, in UTC. Field rotation "
                    "rate depends on where the target sits in the sky at "
                    "this instant, via its altitude/azimuth."))

        self._exposure = QDoubleSpinBox()
        self._exposure.setRange(0.0, 36000.0)
        self._exposure.setDecimals(1)
        self._exposure.setSuffix(" s")
        self._exposure.setValue(60.0)
        form.addRow(*self._r("Sub-exposure length", self._exposure,
                    "Length of a single sub, in seconds — used to report the "
                    "total rotation accumulated over one sub (rate x time)."))

        outer.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._compute_btn = QPushButton("Compute")
        self._compute_btn.setDefault(True)
        self._compute_btn.clicked.connect(self._compute)
        btn_row.addWidget(self._compute_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        outer.addWidget(sep)

        results_box = QGroupBox("Results")
        self._result_grid = QGridLayout(results_box)
        self._result_grid.setHorizontalSpacing(16)
        self._result_grid.setVerticalSpacing(4)
        outer.addWidget(results_box)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #cc8844;")
        outer.addWidget(self._status)
        outer.addStretch(1)

        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        close_row.addWidget(close_btn)
        outer.addLayout(close_row)

        self._row_widgets: list[tuple[QLabel, QLabel]] = []

    @staticmethod
    def _r(label, widget, tip):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return label, row

    def _set_row(self, row: int, label: str, value: str, bold: bool = False) -> None:
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size: 11px; color: palette(placeholderText);")
        val = QLabel(value)
        weight = "700" if bold else "500"
        val.setStyleSheet(f"font-size: 12px; font-weight: {weight};")
        self._result_grid.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignRight)
        self._result_grid.addWidget(val, row, 1)

    def _compute(self) -> None:
        while self._result_grid.count():
            item = self._result_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._status.setText("")

        try:
            from astraios.core.field_rotation import FieldRotationParams, compute_field_rotation

            qt_dt = self._time.dateTime().toUTC()
            iso_time = qt_dt.toString("yyyy-MM-ddTHH:mm:ss")

            params = FieldRotationParams(
                lat_deg=self._lat.value(),
                lon_deg=self._lon.value(),
                time=iso_time,
                ra_deg=self._ra.value(),
                dec_deg=self._dec.value(),
                exposure_s=self._exposure.value(),
            )
            result = compute_field_rotation(params)
        except Exception as exc:
            log.exception("Field rotation calculation failed")
            self._status.setText(f"Could not compute: {exc}")
            return

        self.result = result

        def _rate_str(v: float) -> str:
            return "no limit (E/W)" if math.isinf(v) else f"{v:.4f}"

        self._set_row(0, "Altitude / Azimuth:",
                      f"{result.alt_deg:.2f} deg / {result.az_deg:.2f} deg")
        self._set_row(1, "Hour angle:", f"{result.hour_angle_deg:+.2f} deg")
        self._set_row(2, "Parallactic angle:",
                      f"{result.parallactic_angle_deg:+.2f} deg", bold=True)
        self._set_row(3, "Field rotation rate:",
                      f"{_rate_str(result.rate_arcsec_per_sec)} arcsec/sec  "
                      f"({_rate_str(result.rate_arcsec_per_min)} arcsec/min, "
                      f"{_rate_str(result.rate_deg_per_min)} deg/min)", bold=True)
        self._set_row(4, f"Total rotation over {result.exposure_s:.1f} s exposure:",
                      f"{_rate_str(result.total_rotation_deg)} deg "
                      f"({_rate_str(result.total_rotation_arcsec)} arcsec)", bold=True)

        if math.isinf(result.rate_arcsec_per_sec):
            self._status.setText(
                "Target is at (or within a hair of) the zenith — field "
                "rotation rate is unbounded there."
            )
