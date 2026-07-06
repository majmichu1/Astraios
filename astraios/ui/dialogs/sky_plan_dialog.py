"""What's In My Sky observation-planning dialog.

Drives astraios.core.sky_plan (visibility/transit/rise-set computation over
the embedded DSO catalog for a given observer location and night), ported
from Seti Astro Suite Pro (GPL-3.0, Franklin Marek).

Standalone planning tool: no image needs to be loaded to use it.
"""

from __future__ import annotations

import logging
from zoneinfo import available_timezones

from PyQt6.QtCore import QDate, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from astraios.core.sky_plan import ALL_OBJECT_TYPES, SkyPlanParams
from astraios.ui.widgets.ui_kit import help_dot, param_help

log = logging.getLogger(__name__)

_TYPE_LABELS = {
    "G": "Galaxy",
    "N": "Nebula",
    "OC": "Open Cluster",
    "GC": "Globular Cluster",
    "PN": "Planetary Nebula",
    "SNR": "Supernova Remnant",
    "EN": "Emission Nebula",
}

_TWILIGHT_ITEMS = [
    ("Civil (-6 deg)", "civil"),
    ("Nautical (-12 deg)", "nautical"),
    ("Astronomical (-18 deg)", "astronomical"),
]


class _NumericItem(QTableWidgetItem):
    """A table item that sorts by its underlying numeric value, not its text."""

    def __init__(self, value: float, text: str):
        super().__init__(text)
        self._value = float(value)

    def __lt__(self, other):
        if isinstance(other, _NumericItem):
            return self._value < other._value
        return super().__lt__(other)


class _AltitudeCurveWidget(QWidget):
    """Small custom-painted altitude-vs-time plot for one selected object.

    Draws the object's altitude track (yellow), the Moon's (dashed grey), the
    horizon (0 deg), the night's minimum-altitude threshold (dashed red), and
    shades the portion of the night darker than the selected twilight band —
    a lightweight stand-in for SASpro's pyqtgraph plot in
    ``ObjectVisibilityDialog``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self._curve = None
        self._min_altitude_deg = 20.0
        self._label = ""

    def set_data(self, curve, min_altitude_deg: float, label: str) -> None:
        self._curve = curve
        self._min_altitude_deg = min_altitude_deg
        self._label = label
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, Qt.GlobalColor.black)

        if self._curve is None:
            painter.setPen(QPen(Qt.GlobalColor.gray))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "Select an object above to plot its altitude tonight",
            )
            return

        margin = 28.0
        hours = self._curve.hours
        h_min, h_max = float(hours[0]), float(hours[-1])
        a_min, a_max = -20.0, 90.0

        def to_px(hour: float, alt: float) -> tuple[float, float]:
            x = margin + (hour - h_min) / (h_max - h_min) * (w - 2 * margin)
            y = h - margin - (alt - a_min) / (a_max - a_min) * (h - 2 * margin)
            return x, y

        # Night shading: darker where the Sun is below the twilight threshold.
        night = self._curve.sun_alt_deg < self._curve.twilight_threshold_deg
        i = 0
        n = len(hours)
        while i < n:
            if night[i]:
                j = i
                while j < n and night[j]:
                    j += 1
                x0, _ = to_px(float(hours[i]), 0.0)
                x1, _ = to_px(float(hours[min(j, n - 1)]), 0.0)
                painter.fillRect(int(x0), int(margin), max(1, int(x1 - x0)),
                                  int(h - 2 * margin), Qt.GlobalColor.darkBlue)
                i = j
            else:
                i += 1

        # Horizon (0 deg) and minimum-altitude threshold guide lines.
        _, y0 = to_px(h_min, 0.0)
        painter.setPen(QPen(Qt.GlobalColor.gray, 1))
        painter.drawLine(int(margin), int(y0), int(w - margin), int(y0))
        _, y_min = to_px(h_min, self._min_altitude_deg)
        painter.setPen(QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(margin), int(y_min), int(w - margin), int(y_min))

        # Moon altitude (dashed grey).
        painter.setPen(QPen(Qt.GlobalColor.lightGray, 1, Qt.PenStyle.DashLine))
        prev = None
        for hr, alt in zip(hours, self._curve.moon_alt_deg, strict=False):
            x, y = to_px(float(hr), float(alt))
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)

        # Object altitude (solid yellow).
        painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
        prev = None
        for hr, alt in zip(hours, self._curve.object_alt_deg, strict=False):
            x, y = to_px(float(hr), float(alt))
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)

        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.drawText(int(margin), 14, f"{self._label}  (yellow = object, grey dashed = Moon)")


class _Worker(QThread):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, params: SkyPlanParams):
        super().__init__()
        self._params = params

    def run(self):
        try:
            from astraios.core.sky_plan import plan_sky
            result = plan_sky(self._params, progress=lambda f, m: self.progress.emit(f, m))
            self.done.emit(result)
        except Exception as exc:
            log.exception("Sky plan calculation failed")
            self.failed.emit(str(exc))


class SkyPlanDialog(QDialog):
    """What's In My Sky — plan tonight's observable deep-sky objects.

    Standalone planning tool (no working image required): given an observer
    location, night, and filters, lists which catalog objects are observable,
    with transit time, peak altitude, and hours above the horizon, plus a
    sun/moon summary for the night.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("What's In My Sky")
        self.setMinimumSize(760, 700)
        self._worker: _Worker | None = None
        self._result = None  # set on completion; exposed for scripted verification
        self._last_params: SkyPlanParams | None = None
        self._settings = QSettings("Astraios", "Astraios")

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Lists which deep-sky objects (from the built-in Messier/NGC/IC "
            "catalog) are observable tonight from a given location — transit "
            "time, peak altitude, and hours above your minimum altitude "
            "during dark sky — plus Sun/Moon rise, set, and phase. Computed "
            "locally with astropy; no network access."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        lay.addLayout(self._build_location_group())
        lay.addWidget(self._build_filters_group())

        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Plan Tonight's Sky")
        self._run_btn.clicked.connect(self._run)
        run_row.addWidget(self._run_btn)
        run_row.addStretch()
        lay.addLayout(run_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Type", "Mag", "Transit", "Max Alt", "Hours Visible", "Moon Sep"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        lay.addWidget(self._table, 2)

        self._curve_widget = _AltitudeCurveWidget()
        lay.addWidget(self._curve_widget, 1)

        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setMaximumHeight(140)
        lay.addWidget(self._summary)

        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btns.addWidget(close_btn)
        lay.addLayout(btns)

        self._load_settings()

    # ── UI construction ─────────────────────────────────────────────────────

    @staticmethod
    def _r(label, widget, tip):
        row = QHBoxLayout()
        row.addWidget(widget)
        row.addWidget(help_dot(tip))
        row.addStretch()
        return label, row

    def _build_location_group(self) -> QHBoxLayout:
        group = QGroupBox("Observer Location && Date")
        form = QFormLayout(group)

        self._lat = QDoubleSpinBox()
        self._lat.setRange(-90.0, 90.0)
        self._lat.setDecimals(4)
        self._lat.setSuffix(" deg")
        form.addRow(*self._r("Latitude", self._lat,
                    "Observer latitude, degrees north (negative for south)."))

        self._lon = QDoubleSpinBox()
        self._lon.setRange(-180.0, 180.0)
        self._lon.setDecimals(4)
        self._lon.setSuffix(" deg")
        form.addRow(*self._r("Longitude", self._lon,
                    "Observer longitude, degrees east (negative for west)."))

        self._elevation = QDoubleSpinBox()
        self._elevation.setRange(-500.0, 9000.0)
        self._elevation.setDecimals(0)
        self._elevation.setSuffix(" m")
        form.addRow(*self._r("Elevation", self._elevation,
                    "Site elevation above sea level. Only affects horizon "
                    "geometry slightly — not critical to get exact."))

        self._tz = QComboBox()
        self._tz.setEditable(True)
        self._tz.addItems(sorted(available_timezones()))
        self._tz.setCurrentText("UTC")
        form.addRow(*self._r("Timezone", self._tz,
                    "IANA timezone name (e.g. \"America/Denver\"). All times "
                    "in the results are shown in this timezone. Falls back "
                    "to UTC if not recognized."))

        self._date = QDateEdit()
        self._date.setCalendarPopup(True)
        self._date.setDate(QDate.currentDate())
        form.addRow(*self._r("Observing date", self._date,
                    "Local calendar date of the observing night. The plan "
                    "covers local noon on this date through local noon the "
                    "next day, so the whole night is one contiguous window."))

        outer = QHBoxLayout()
        outer.addWidget(group)
        return outer

    def _build_filters_group(self) -> QGroupBox:
        group = QGroupBox("Filters")
        outer = QVBoxLayout(group)
        form = QFormLayout()

        self._min_alt = QDoubleSpinBox()
        self._min_alt.setRange(0.0, 89.0)
        self._min_alt.setValue(20.0)
        self._min_alt.setSuffix(" deg")
        form.addRow(*self._r("Minimum altitude", self._min_alt, param_help(
            "How high above the horizon an object must get, during the "
            "dark part of the night, to count as observable.",
            higher="Stricter — excludes objects that only clear the murky, "
                   "high-extinction air near the horizon.",
            lower="More permissive — includes low passes, at the cost of "
                  "atmospheric extinction and often obstructed horizons.",
            default="20 deg is a reasonable default for most sites/horizons.",
        )))

        mag_row = QHBoxLayout()
        self._limit_mag = QCheckBox("Limit to magnitude")
        self._max_mag = QDoubleSpinBox()
        self._max_mag.setRange(-5.0, 20.0)
        self._max_mag.setValue(9.0)
        self._max_mag.setEnabled(False)
        self._limit_mag.toggled.connect(self._max_mag.setEnabled)
        mag_row.addWidget(self._limit_mag)
        mag_row.addWidget(self._max_mag)
        mag_row.addWidget(help_dot(param_help(
            "Faintest catalog magnitude to include.",
            how="Objects with no reliably-established integrated magnitude "
                "in the catalog (many diffuse nebulae) are never excluded "
                "by this filter, since we don't know they'd fail it.",
            higher="Fainter limit — includes dimmer targets that need more "
                   "aperture/exposure or darker skies.",
            lower="Brighter-only — restricts the list to easy, bright "
                  "targets.",
        )))
        mag_row.addStretch()
        form.addRow(mag_row)

        self._twilight = QComboBox()
        for text, _key in _TWILIGHT_ITEMS:
            self._twilight.addItem(text)
        self._twilight.setCurrentIndex(2)
        form.addRow(*self._r("Twilight definition", self._twilight, param_help(
            "Which twilight band counts as \"night\" for the hours-visible "
            "calculation.",
            how="Civil / nautical / astronomical correspond to the Sun's "
                "altitude falling below -6 / -12 / -18 degrees.",
            higher="Astronomical (-18 deg) — strictest, truest dark-sky "
                   "imaging window; the deepest-sky targets need this.",
            lower="Civil (-6 deg) — most permissive; useful for the Moon, "
                  "planets, and bright clusters that tolerate some twilight.",
        )))
        outer.addLayout(form)

        types_row = QHBoxLayout()
        types_row.addWidget(QLabel("Object types:"))
        self._type_checks: dict[str, QCheckBox] = {}
        for code in ALL_OBJECT_TYPES:
            cb = QCheckBox(_TYPE_LABELS.get(code, code))
            cb.setChecked(True)
            self._type_checks[code] = cb
            types_row.addWidget(cb)
        types_row.addStretch()
        outer.addLayout(types_row)

        return group

    # ── settings persistence ────────────────────────────────────────────────

    def _load_settings(self) -> None:
        s = self._settings
        self._lat.setValue(float(s.value("sky_plan/latitude_deg", 0.0)))
        self._lon.setValue(float(s.value("sky_plan/longitude_deg", 0.0)))
        self._elevation.setValue(float(s.value("sky_plan/elevation_m", 0.0)))
        tz = str(s.value("sky_plan/timezone", "UTC"))
        self._tz.setCurrentText(tz)
        self._min_alt.setValue(float(s.value("sky_plan/min_altitude_deg", 20.0)))
        limit_mag = str(s.value("sky_plan/limit_magnitude", "false")).lower() == "true"
        self._limit_mag.setChecked(limit_mag)
        self._max_mag.setValue(float(s.value("sky_plan/max_magnitude", 9.0)))
        twilight_idx = int(s.value("sky_plan/twilight_index", 2))
        if 0 <= twilight_idx < self._twilight.count():
            self._twilight.setCurrentIndex(twilight_idx)

    def _save_settings(self) -> None:
        s = self._settings
        s.setValue("sky_plan/latitude_deg", self._lat.value())
        s.setValue("sky_plan/longitude_deg", self._lon.value())
        s.setValue("sky_plan/elevation_m", self._elevation.value())
        s.setValue("sky_plan/timezone", self._tz.currentText().strip())
        s.setValue("sky_plan/min_altitude_deg", self._min_alt.value())
        s.setValue("sky_plan/limit_magnitude", "true" if self._limit_mag.isChecked() else "false")
        s.setValue("sky_plan/max_magnitude", self._max_mag.value())
        s.setValue("sky_plan/twilight_index", self._twilight.currentIndex())

    # ── run ──────────────────────────────────────────────────────────────────

    def _run(self):
        checked_types = [code for code, cb in self._type_checks.items() if cb.isChecked()]
        if not checked_types:
            self._status.setText("Select at least one object type.")
            return
        object_types = None if len(checked_types) == len(self._type_checks) else tuple(
            checked_types
        )

        params = SkyPlanParams(
            latitude_deg=float(self._lat.value()),
            longitude_deg=float(self._lon.value()),
            elevation_m=float(self._elevation.value()),
            date=self._date.date().toString("yyyy-MM-dd"),
            timezone=self._tz.currentText().strip() or "UTC",
            min_altitude_deg=float(self._min_alt.value()),
            max_magnitude=float(self._max_mag.value()) if self._limit_mag.isChecked() else None,
            object_types=object_types,
            twilight=_TWILIGHT_ITEMS[self._twilight.currentIndex()][1],
        )
        self._last_params = params
        self._save_settings()

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.setText("Calculating...")
        self._table.setRowCount(0)
        self._curve_widget.set_data(None, params.min_altitude_deg, "")

        self._worker = _Worker(params)
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
        self._result = result

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(result.objects))
        for row, obj in enumerate(result.objects):
            name_item = QTableWidgetItem(obj.name)
            name_item.setData(Qt.ItemDataRole.UserRole, (obj.ra_deg, obj.dec_deg, obj.name))
            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, QTableWidgetItem(obj.type_label))
            mag_text = f"{obj.magnitude:.1f}" if obj.magnitude is not None else "-"
            mag_val = obj.magnitude if obj.magnitude is not None else float("inf")
            self._table.setItem(row, 2, _NumericItem(mag_val, mag_text))
            self._table.setItem(row, 3, QTableWidgetItem(obj.transit_time_local))
            self._table.setItem(
                row, 4, _NumericItem(obj.max_altitude_deg, f"{obj.max_altitude_deg:.1f}")
            )
            self._table.setItem(
                row, 5, _NumericItem(obj.hours_visible, f"{obj.hours_visible:.2f}")
            )
            self._table.setItem(
                row, 6, _NumericItem(obj.moon_separation_deg, f"{obj.moon_separation_deg:.0f}")
            )
        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()

        self._summary.setPlainText(self._format_summary(result))
        self._status.setText(result.message)
        if result.warnings:
            self._status.setText(result.message + "  (" + "; ".join(result.warnings) + ")")

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")

    def _on_selection_changed(self):
        items = self._table.selectedItems()
        if not items or self._last_params is None:
            return
        row = items[0].row()
        name_item = self._table.item(row, 0)
        if name_item is None:
            return
        data = name_item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        ra_deg, dec_deg, name = data
        try:
            from astraios.core.sky_plan import object_altitude_curve
            curve = object_altitude_curve(ra_deg, dec_deg, self._last_params)
            self._curve_widget.set_data(curve, self._last_params.min_altitude_deg, name)
        except Exception as exc:
            log.warning("Could not compute altitude curve for %s: %s", name, exc)

    @staticmethod
    def _format_summary(result) -> str:
        def _t(v):
            return v if v else "n/a (does not rise/set in this window)"

        return (
            f"Local Sidereal Time (at local noon): {result.local_sidereal_time}\n"
            f"Sun:   rise {_t(result.sun_rise_local)}   set {_t(result.sun_set_local)}\n"
            f"Twilight (selected band): evening begins {_t(result.twilight_evening_local)}   "
            f"morning ends {_t(result.twilight_morning_local)}\n"
            f"Moon:  rise {_t(result.moon_rise_local)}   "
            f"transit {_t(result.moon_transit_local)}   set {_t(result.moon_set_local)}\n"
            f"Moon phase: {result.moon_phase_pct}%  ({result.moon_phase_label})\n"
            f"Catalog: {result.n_after_filters} of {result.n_catalog_total} objects "
            f"matched filters; {len(result.objects)} observable tonight."
        )
