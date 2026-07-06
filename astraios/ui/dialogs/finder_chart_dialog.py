"""Finder Chart dialog.

Drives astraios.core.finder_chart (render an annotated star chart from a
plate-solved image: catalog star/DSO markers, a north/east compass, a scale
bar, a field marker, an optional pixel grid, and an optional imaging-train
field-of-view box), ported from Seti Astro Suite Pro (GPL-3.0, Franklin
Marek).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from astraios.ui.widgets.ui_kit import help_dot

log = logging.getLogger(__name__)


def _to_astropy_wcs(wcs_header: Any):
    """Build an astropy WCS from a header, header-like dict, or WCS instance.

    Small local re-implementation of the equivalent (private) helper in
    astraios.core.finder_chart, so the dialog can read the field center and
    pixel scale for the Gaia/DSO cone queries without reaching into a core
    module's private API.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    if isinstance(wcs_header, WCS):
        return wcs_header
    if isinstance(wcs_header, fits.Header):
        return WCS(wcs_header, relax=True)
    if isinstance(wcs_header, dict):
        return WCS(fits.Header(wcs_header), relax=True)
    raise TypeError(f"Unsupported WCS header type: {type(wcs_header)!r}")


class _Worker(QThread):
    progress = pyqtSignal(float, str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, image, wcs_header, params, catalog_stars, dso_list):
        super().__init__()
        self._image = image
        self._wcs_header = wcs_header
        self._params = params
        self._catalog_stars = catalog_stars
        self._dso_list = dso_list

    def run(self):
        try:
            from astraios.core.finder_chart import render_finder_chart
            result = render_finder_chart(
                self._image, self._wcs_header, self._params,
                catalog_stars=self._catalog_stars, dso_list=self._dso_list,
                progress=lambda f, m: self.progress.emit(f, m),
            )
            self.done.emit(result)
        except Exception as exc:
            log.exception("Finder chart render failed")
            self.failed.emit(str(exc))


class FinderChartDialog(QDialog):
    """Render an annotated finder chart from a plate-solved image."""

    result_ready = pyqtSignal(object)

    def __init__(self, image: np.ndarray, wcs_header: Any = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Finder Chart")
        self.setMinimumWidth(460)
        self._image = image
        self._wcs_header = wcs_header
        self._worker: _Worker | None = None

        lay = QVBoxLayout(self)
        intro = QLabel(
            "Renders an annotated finder chart from the plate-solved image: "
            "catalog star/DSO markers, a north/east compass, a scale bar, a "
            "field marker, and an optional imaging-train field-of-view box."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        lay.addWidget(intro)

        if self._wcs_header is None:
            no_wcs = QLabel("No WCS solution available — plate solve first.")
            no_wcs.setStyleSheet("color: #e06666;")
            no_wcs.setWordWrap(True)
            lay.addWidget(no_wcs)

        # --- Annotation checkboxes ---
        ann_form = QFormLayout()
        self._show_stars = QCheckBox("Catalog stars")
        self._show_dso = QCheckBox("Deep-sky objects")
        self._show_dso.setChecked(True)
        self._show_compass = QCheckBox("Compass (N/E)")
        self._show_compass.setChecked(True)
        self._show_scale_bar = QCheckBox("Scale bar")
        self._show_scale_bar.setChecked(True)
        self._show_field_marker = QCheckBox("Field marker")
        self._show_field_marker.setChecked(True)
        self._show_grid = QCheckBox("Pixel grid")
        self._show_fov_box = QCheckBox("Imaging FOV box")

        self._gaia_available = self._check_gaia_available()
        if self._gaia_available:
            self._show_stars.setChecked(True)
        else:
            self._show_stars.setChecked(False)
            self._show_stars.setEnabled(False)
            self._show_stars.setToolTip(
                "<qt>No local Gaia catalog installed. Install a magnitude "
                "band from Tools &gt; GAIA Catalog Manager to overlay "
                "catalog stars. WCS-only annotations (compass, scale bar, "
                "field marker, grid, FOV box) still work without it.</qt>"
            )

        ann_form.addRow(*self._r("", self._show_stars,
                    "Overlay Gaia DR3 star positions, queried from the "
                    "local catalog around the field center."))
        ann_form.addRow(*self._r("", self._show_dso,
                    "Overlay the embedded Messier/NGC deep-sky object "
                    "catalog for objects that fall inside the field, with "
                    "size circles where known."))
        ann_form.addRow(*self._r("", self._show_compass,
                    "Draw north/east direction arrows derived from the "
                    "WCS."))
        ann_form.addRow(*self._r("", self._show_scale_bar,
                    "Draw a labeled angular scale bar."))
        ann_form.addRow(*self._r("", self._show_field_marker,
                    "Draw a crosshair + circle at the field center."))
        ann_form.addRow(*self._r("", self._show_grid,
                    "Overlay a fixed-spacing pixel grid."))
        ann_form.addRow(*self._r("", self._show_fov_box,
                    "Overlay the rectangular field of view of a different "
                    "imaging train (focal length / sensor below) — useful "
                    "for planning a mosaic panel or comparing against a "
                    "different scope/camera combo."))
        lay.addLayout(ann_form)

        # --- Background ---
        bg_form = QFormLayout()
        self._background = QComboBox()
        self._background.addItems(["Image", "Black", "White"])
        bg_form.addRow(*self._r("Background", self._background,
                    "What to draw the annotations over: the plate-solved "
                    "image itself, or a plain black/white canvas."))
        self._invert = QCheckBox("Invert tones")
        bg_form.addRow(self._invert)
        self._stretch_bg = QCheckBox("Auto-stretch background")
        self._stretch_bg.setChecked(True)
        bg_form.addRow(self._stretch_bg)
        lay.addLayout(bg_form)

        # --- FOV box optics ---
        self._optics_form = QFormLayout()
        self._focal_length = QDoubleSpinBox()
        self._focal_length.setRange(10.0, 10000.0)
        self._focal_length.setValue(500.0)
        self._focal_length.setSuffix(" mm")
        self._optics_form.addRow(*self._r("Focal length", self._focal_length,
                    "Focal length of the imaging train being previewed."))
        self._pixel_pitch = QDoubleSpinBox()
        self._pixel_pitch.setRange(0.5, 30.0)
        self._pixel_pitch.setSingleStep(0.1)
        self._pixel_pitch.setValue(3.76)
        self._pixel_pitch.setSuffix(" µm")
        self._optics_form.addRow(*self._r("Pixel pitch", self._pixel_pitch,
                    "Sensor pixel size of the imaging train being "
                    "previewed."))
        self._sensor_w = QSpinBox()
        self._sensor_w.setRange(16, 20000)
        self._sensor_w.setValue(6248)
        self._optics_form.addRow(*self._r("Sensor width", self._sensor_w,
                    "Sensor width, in pixels."))
        self._sensor_h = QSpinBox()
        self._sensor_h.setRange(16, 20000)
        self._sensor_h.setValue(4176)
        self._optics_form.addRow(*self._r("Sensor height", self._sensor_h,
                    "Sensor height, in pixels."))
        self._rotation = QDoubleSpinBox()
        self._rotation.setRange(-180.0, 180.0)
        self._rotation.setValue(0.0)
        self._rotation.setSuffix(" deg")
        self._optics_form.addRow(*self._r("Rotation", self._rotation,
                    "Camera rotation angle, clockwise from north."))
        lay.addLayout(self._optics_form)
        self._set_optics_enabled(self._show_fov_box.isChecked())
        self._show_fov_box.toggled.connect(self._set_optics_enabled)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QHBoxLayout()
        self._render_btn = QPushButton("Render Finder Chart")
        self._render_btn.setEnabled(self._wcs_header is not None)
        self._render_btn.clicked.connect(self._render)
        btns.addWidget(self._render_btn)
        btns.addStretch()
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

    def _set_optics_enabled(self, enabled: bool):
        for w in (self._focal_length, self._pixel_pitch, self._sensor_w,
                  self._sensor_h, self._rotation):
            w.setEnabled(enabled)

    @staticmethod
    def _check_gaia_available() -> bool:
        try:
            from astraios.core.gaia_catalog import installed_files
            return bool(installed_files())
        except Exception:
            log.debug("Gaia catalog availability check failed", exc_info=True)
            return False

    def _field_center_and_fov(self):
        """Return ((ra_deg, dec_deg), fov_deg) for the current WCS + image, or None."""
        try:
            from astropy.wcs.utils import proj_plane_pixel_scales

            wcs = _to_astropy_wcs(self._wcs_header)
            h, w = self._image.shape[-2], self._image.shape[-1]
            center = wcs.pixel_to_world(w / 2.0, h / 2.0)
            scales = proj_plane_pixel_scales(wcs)
            deg_per_px = float(np.nanmedian(scales))
            if not np.isfinite(deg_per_px) or deg_per_px <= 0:
                return None
            fov_deg = max(w, h) * deg_per_px * 1.5
            return (float(center.ra.deg), float(center.dec.deg)), fov_deg
        except Exception:
            log.debug("Finder chart: could not derive field center", exc_info=True)
            return None

    def _fetch_catalog_stars(self):
        """Cone-query the local Gaia catalog around the field center."""
        if not (self._gaia_available and self._show_stars.isChecked()
                and self._wcs_header is not None):
            return None
        info = self._field_center_and_fov()
        if info is None:
            return None
        (ra, dec), fov_deg = info
        radius_deg = max(fov_deg / 2.0, 0.05)
        try:
            from PyQt6.QtCore import QSettings

            from astraios.core.gaia_catalog import GaiaCatalog

            settings = QSettings("Astraios", "Astraios")
            raw_dir = settings.value("gaia/catalog_dir", "")
            catalog_dir = raw_dir.strip() or None if isinstance(raw_dir, str) else None

            with GaiaCatalog(catalog_dir) as cat:
                return cat.cone_query(ra, dec, radius_deg, max_stars=2000)
        except Exception:
            log.exception("Finder chart: Gaia cone query failed")
            return None

    def _fetch_dso_list(self):
        if not (self._show_dso.isChecked() and self._wcs_header is not None):
            return None
        info = self._field_center_and_fov()
        if info is None:
            return None
        (ra, dec), fov_deg = info
        try:
            from astraios.core.dso_catalog import query_dso_in_field
            return query_dso_in_field(ra, dec, fov_deg)
        except Exception:
            log.exception("Finder chart: DSO query failed")
            return None

    def _render(self):
        from astraios.core.finder_chart import FinderChartParams

        if self._wcs_header is None:
            self._status.setText("No WCS solution — plate solve first.")
            return

        params = FinderChartParams(
            background=self._background.currentText().lower(),
            invert=self._invert.isChecked(),
            stretch_background=self._stretch_bg.isChecked(),
            show_stars=self._show_stars.isChecked() and self._gaia_available,
            show_star_labels=False,
            show_dso=self._show_dso.isChecked(),
            show_compass=self._show_compass.isChecked(),
            show_scale_bar=self._show_scale_bar.isChecked(),
            show_field_marker=self._show_field_marker.isChecked(),
            show_grid=self._show_grid.isChecked(),
            show_fov_box=self._show_fov_box.isChecked(),
            focal_length_mm=float(self._focal_length.value()),
            pixel_pitch_um=float(self._pixel_pitch.value()),
            sensor_w_px=int(self._sensor_w.value()),
            sensor_h_px=int(self._sensor_h.value()),
            rotation_deg=float(self._rotation.value()),
        )

        catalog_stars = self._fetch_catalog_stars()
        dso_list = self._fetch_dso_list()

        self._render_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 100)
        self._status.setText("Rendering finder chart...")
        self._worker = _Worker(self._image, self._wcs_header, params, catalog_stars, dso_list)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_fail)
        self._worker.start()

    def _on_progress(self, fraction: float, message: str):
        self._progress.setValue(int(fraction * 100))
        self._status.setText(message)

    def _on_done(self, result):
        self._progress.setVisible(False)
        self._render_btn.setEnabled(True)
        if not isinstance(result, np.ndarray):
            self._status.setText("No result produced.")
            return
        self._status.setText("Finder chart rendered.")
        self.result_ready.emit(result)

    def _on_fail(self, msg):
        self._progress.setVisible(False)
        self._render_btn.setEnabled(True)
        self._status.setText(f"Failed: {msg}")
