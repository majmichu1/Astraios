"""Finder Chart — annotated star chart rendered from a plate-solved image.

Draws catalog star/DSO markers, labels, a north/east compass, a scale bar,
a field crosshair/circle, an optional pixel grid, and an optional imaging-train
field-of-view box onto a rendered copy of the image (or a neutral background).

This is a headless, network-free port: SASpro's original tool fetches a HiPS
survey cutout as the chart background and queries SIMBAD/local CSV catalogs
live. Here the plate-solved image itself is the background, and catalog data
(stars, DSOs) is passed in by the caller (e.g. from
``astraios.core.gaia_catalog`` / ``astraios.core.dso_catalog`` or a live
query) so this module has no network dependency and is fully testable with
synthetic data.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

import astropy.units as u
import cv2
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class FinderChartParams:
    """All user-facing settings for :func:`render_finder_chart`.

    Catalog data (``catalog_stars`` / ``dso_list``) is supplied separately so
    this stays network-free; each entry may be a dict or any object exposing
    ``ra_deg``/``ra``, ``dec_deg``/``dec``, ``mag``/``g_mag``, ``name`` and
    (for DSOs) ``size_arcmin`` attributes/keys.
    """

    # --- Background / canvas ---
    background: str = "image"  # "image" | "black" | "white"
    invert: bool = False  # invert image tones before drawing
    stretch_background: bool = True  # percentile autostretch when background="image"
    out_size: int = 0  # output canvas size in px, square; 0 = keep native image size

    # --- Star annotations ---
    show_stars: bool = True
    star_mag_limit: float = 12.0  # faintest catalog magnitude to plot
    star_max_labels: int = 40  # cap on labelled/plotted stars (brightest first)
    show_star_labels: bool = True

    # --- DSO annotations ---
    show_dso: bool = True
    dso_mag_limit: float = 14.0  # faintest catalog magnitude to plot (None mags always pass)
    dso_max_labels: int = 40
    show_dso_labels: bool = True
    show_dso_size_circles: bool = True  # draw a circle for the DSO's angular size

    # --- Compass / scale bar / field marker / grid ---
    show_compass: bool = True
    show_scale_bar: bool = True
    show_field_marker: bool = True  # crosshair + circle at the field center
    show_grid: bool = False
    grid_spacing_px: int = 150

    # --- Imaging-train FOV box overlay ---
    show_fov_box: bool = False
    focal_length_mm: float = 500.0
    pixel_pitch_um: float = 3.76
    sensor_w_px: int = 6248
    sensor_h_px: int = 4176
    rotation_deg: float = 0.0  # clockwise from north

    # --- Style ---
    star_color: tuple[int, int, int] = (255, 176, 0)
    dso_color: tuple[int, int, int] = (102, 204, 255)
    dso_circle_color: tuple[int, int, int] = (81, 69, 255)
    compass_color: tuple[int, int, int] = (255, 255, 255)
    scale_bar_color: tuple[int, int, int] = (255, 255, 255)
    field_marker_color: tuple[int, int, int] = (255, 255, 255)
    grid_color: tuple[int, int, int] = (120, 120, 120)
    fov_box_color: tuple[int, int, int] = (0, 255, 136)
    label_font_scale: float = 0.4
    marker_radius_px: int = 4
    declutter_cell_px: int = 28  # min pixel spacing between kept labels


def _emit(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress is None:
        return
    try:
        progress(fraction, message)
    except Exception:
        log.debug("Finder chart progress callback raised", exc_info=True)


# --------------------------------------------------------------------------
# WCS helpers
# --------------------------------------------------------------------------


def _to_wcs(wcs_header: Any) -> WCS:
    """Build an astropy WCS from a header, header-like dict, or WCS instance."""
    if isinstance(wcs_header, WCS):
        return wcs_header
    if isinstance(wcs_header, fits.Header):
        return WCS(wcs_header, relax=True)
    if isinstance(wcs_header, dict):
        return WCS(fits.Header(wcs_header), relax=True)
    raise TypeError(f"Unsupported wcs_header type for finder chart: {type(wcs_header)!r}")


def _pixel_scale_arcsec(wcs: WCS) -> float | None:
    try:
        scales = proj_plane_pixel_scales(wcs)
        deg_per_px = float(np.nanmedian(scales))
        if not np.isfinite(deg_per_px) or deg_per_px <= 0:
            return None
        return deg_per_px * 3600.0
    except Exception:
        return None


def _world_to_pixel(
    wcs: WCS, ra_deg: np.ndarray, dec_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coords = SkyCoord(
        np.atleast_1d(ra_deg) * u.deg, np.atleast_1d(dec_deg) * u.deg, frame="icrs"
    )
    x, y = wcs.world_to_pixel(coords)
    return np.atleast_1d(x).astype(np.float64), np.atleast_1d(y).astype(np.float64)


# --------------------------------------------------------------------------
# Catalog entry duck-typing
# --------------------------------------------------------------------------


def _get_field(entry: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(entry, dict):
            if name in entry and entry[name] is not None:
                return entry[name]
        elif hasattr(entry, name) and getattr(entry, name) is not None:
            return getattr(entry, name)
    return default


def _normalize_entries(entries: list[Any], mag_limit: float | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        ra = _get_field(e, "ra_deg", "ra")
        dec = _get_field(e, "dec_deg", "dec")
        if ra is None or dec is None:
            continue
        mag = _get_field(e, "mag", "g_mag", "vmag", "v_mag")
        if mag is not None and mag_limit is not None and float(mag) > float(mag_limit):
            continue
        name = _get_field(e, "name", "source_id", default="")
        size_arcmin = _get_field(e, "size_arcmin")
        out.append({
            "ra": float(ra),
            "dec": float(dec),
            "mag": float(mag) if mag is not None else None,
            "name": str(name),
            "size_arcmin": float(size_arcmin) if size_arcmin is not None else None,
        })
    return out


def _prepare_markers(
    entries: list[Any],
    wcs: WCS,
    width: int,
    height: int,
    mag_limit: float | None,
    max_labels: int,
    cell_px: int,
) -> list[dict[str, Any]]:
    """Filter by magnitude, project to pixels, declutter, and cap the count."""
    norm = _normalize_entries(entries, mag_limit)
    if not norm:
        return []

    # Brightest first; entries without a magnitude sort last.
    norm.sort(key=lambda r: r["mag"] if r["mag"] is not None else float("inf"))

    ra_arr = np.array([r["ra"] for r in norm], dtype=np.float64)
    dec_arr = np.array([r["dec"] for r in norm], dtype=np.float64)
    xs, ys = _world_to_pixel(wcs, ra_arr, dec_arr)

    cell_px = max(1, int(cell_px))
    used_cells: set[tuple[int, int]] = set()
    kept: list[dict[str, Any]] = []

    for row, x, y in zip(norm, xs, ys):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if not (0.0 <= x < width and 0.0 <= y < height):
            continue
        cell = (int(x // cell_px), int(y // cell_px))
        if cell in used_cells:
            continue
        used_cells.add(cell)
        entry = dict(row)
        entry["x"] = float(x)
        entry["y"] = float(y)
        kept.append(entry)
        if len(kept) >= max(1, int(max_labels)):
            break

    return kept


# --------------------------------------------------------------------------
# Canvas preparation
# --------------------------------------------------------------------------


def _image_to_hwc(image: np.ndarray) -> np.ndarray:
    """Convert an astraios (H,W) or (C,H,W) float image to (H,W,3) float."""
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1).astype(np.float32, copy=False)
    if arr.ndim == 3:
        n_channels = arr.shape[0]
        if n_channels == 1:
            return np.repeat(arr[0][..., None], 3, axis=-1).astype(np.float32, copy=False)
        return np.transpose(arr[:3], (1, 2, 0)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported image shape for finder chart: {arr.shape}")


def _make_background_canvas(image: np.ndarray, params: FinderChartParams) -> np.ndarray:
    """Build the HxWx3 uint8 background canvas at the image's native resolution."""
    arr = np.asarray(image)
    height = arr.shape[-2]
    width = arr.shape[-1]

    if params.background == "image":
        rgb = _image_to_hwc(arr)
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        if params.stretch_background:
            lo, hi = np.percentile(rgb, [1.0, 99.5])
            if hi > lo:
                rgb = (rgb - lo) / (hi - lo)
        rgb = np.clip(rgb, 0.0, 1.0)
        if params.invert:
            rgb = 1.0 - rgb
        canvas = (rgb * 255.0 + 0.5).astype(np.uint8)
    else:
        fill = 255 if params.background == "white" else 0
        canvas = np.full((height, width, 3), fill, dtype=np.uint8)

    return np.ascontiguousarray(canvas)


def _fit_canvas(canvas: np.ndarray, wcs: WCS, out_size: int) -> tuple[np.ndarray, WCS]:
    """Center-crop or center-pad the canvas to a square ``out_size``, adjusting WCS CRPIX."""
    height, width = canvas.shape[:2]
    if out_size <= 0 or (out_size == height and out_size == width):
        return canvas, wcs

    wcs_out = wcs.deepcopy()

    if out_size <= min(height, width):
        x0 = (width - out_size) // 2
        y0 = (height - out_size) // 2
        cropped = canvas[y0:y0 + out_size, x0:x0 + out_size]
        offset = np.array([x0, y0], dtype=float)
        wcs_out.wcs.crpix = np.array(wcs_out.wcs.crpix, dtype=float) - offset
        return np.ascontiguousarray(cropped), wcs_out

    pad_x = (out_size - width) // 2
    pad_y = (out_size - height) // 2
    padded = np.zeros((out_size, out_size, 3), dtype=canvas.dtype)
    padded[pad_y:pad_y + height, pad_x:pad_x + width] = canvas
    pad = np.array([pad_x, pad_y], dtype=float)
    wcs_out.wcs.crpix = np.array(wcs_out.wcs.crpix, dtype=float) + pad
    return padded, wcs_out


# --------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------


def _draw_marker(
    canvas: np.ndarray,
    x: float,
    y: float,
    color: tuple[int, int, int],
    radius: int,
    label: str | None,
    font_scale: float,
    shape: str = "circle",
) -> None:
    center = (int(round(x)), int(round(y)))
    if shape == "square":
        half = max(1, radius)
        cv2.rectangle(canvas, (center[0] - half, center[1] - half),
                      (center[0] + half, center[1] + half), color, 1, cv2.LINE_AA)
    else:
        cv2.circle(canvas, center, max(1, radius), color, 1, cv2.LINE_AA)

    if label:
        pos = (center[0] + radius + 3, center[1] - radius - 3)
        # Black outline pass then colored pass, mimicking matplotlib's path effects.
        cv2.putText(
            canvas, label, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(canvas, label, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)


def _draw_grid(canvas: np.ndarray, width: int, height: int, params: FinderChartParams) -> None:
    step = max(8, int(params.grid_spacing_px))
    for gx in range(0, width, step):
        cv2.line(canvas, (gx, 0), (gx, height - 1), params.grid_color, 1, cv2.LINE_AA)
    for gy in range(0, height, step):
        cv2.line(canvas, (0, gy), (width - 1, gy), params.grid_color, 1, cv2.LINE_AA)


def _draw_field_marker(
    canvas: np.ndarray, width: int, height: int, color: tuple[int, int, int]
) -> None:
    cx, cy = width // 2, height // 2
    size = max(6, int(min(width, height) * 0.05))
    cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), size, color, 1, cv2.LINE_AA)


def _draw_compass(
    canvas: np.ndarray, wcs: WCS, params: FinderChartParams, width: int, height: int
) -> None:
    color = params.compass_color
    cx = width * 0.88
    cy = height * 0.88
    base_len = max(10.0, min(width, height) * 0.08)

    try:
        center = wcs.pixel_to_world(width / 2.0, height / 2.0)
        ra = center.ra
        dec = center.dec
        step = 0.05 * u.deg
        north = SkyCoord(ra, dec + step, frame="icrs")
        cosd = max(0.15, float(np.cos(dec.to(u.rad).value)))
        east = SkyCoord(ra + step / cosd, dec, frame="icrs")

        x0, y0 = wcs.world_to_pixel(SkyCoord(ra, dec, frame="icrs"))
        xn, yn = wcs.world_to_pixel(north)
        xe, ye = wcs.world_to_pixel(east)

        vn = np.array([float(xn - x0), float(yn - y0)], dtype=np.float64)
        ve = np.array([float(xe - x0), float(ye - y0)], dtype=np.float64)

        def _unit(v: np.ndarray) -> np.ndarray:
            n = float(np.hypot(v[0], v[1])) or 1.0
            return v / n

        vn = _unit(vn) * base_len
        ve = _unit(ve) * base_len
    except Exception:
        log.debug("Finder chart compass fallback (WCS direction failed)", exc_info=True)
        vn = np.array([0.0, -base_len])
        ve = np.array([base_len, 0.0])

    p0 = (int(round(cx)), int(round(cy)))
    pn = (int(round(cx + vn[0])), int(round(cy + vn[1])))
    pe = (int(round(cx + ve[0])), int(round(cy + ve[1])))

    cv2.arrowedLine(canvas, p0, pn, color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(canvas, p0, pe, color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.putText(canvas, "N", (pn[0] + 4, pn[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv2.LINE_AA)
    cv2.putText(canvas, "E", (pe[0] + 4, pe[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv2.LINE_AA)


_SCALE_BAR_ARCMIN_CANDIDATES = (1, 2, 5, 10, 20, 30, 60, 120, 300, 600)


def _draw_scale_bar(
    canvas: np.ndarray, wcs: WCS, params: FinderChartParams, width: int, height: int
) -> None:
    arcsec_per_px = _pixel_scale_arcsec(wcs)
    if arcsec_per_px is None or arcsec_per_px <= 0:
        return

    max_px = 0.5 * width
    bar_arcmin = _SCALE_BAR_ARCMIN_CANDIDATES[0]
    for candidate in _SCALE_BAR_ARCMIN_CANDIDATES:
        px = (candidate * 60.0) / arcsec_per_px
        if px <= max_px:
            bar_arcmin = candidate
        else:
            break

    bar_px = (bar_arcmin * 60.0) / arcsec_per_px
    if not np.isfinite(bar_px) or bar_px < 2:
        return

    x0 = width * 0.08
    y0 = height * 0.90
    x1 = x0 + bar_px
    color = params.scale_bar_color

    cv2.line(canvas, (int(x0), int(y0)), (int(x1), int(y0)), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (int(x0), int(y0) - 4), (int(x0), int(y0) + 4), color, 2, cv2.LINE_AA)
    cv2.line(canvas, (int(x1), int(y0) - 4), (int(x1), int(y0) + 4), color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{bar_arcmin}' ({arcsec_per_px:.2f}\"/px)", (int(x0), int(y0) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _draw_fov_box(
    canvas: np.ndarray, wcs: WCS, params: FinderChartParams, width: int, height: int
) -> None:
    fl = float(params.focal_length_mm)
    pitch = float(params.pixel_pitch_um)
    sw = int(params.sensor_w_px)
    sh = int(params.sensor_h_px)
    rot = float(params.rotation_deg)
    if fl <= 0 or pitch <= 0 or sw <= 0 or sh <= 0:
        return

    try:
        half_w_deg = math.degrees(math.atan((pitch * sw) / (2000.0 * fl)))
        half_h_deg = math.degrees(math.atan((pitch * sh) / (2000.0 * fl)))

        center = wcs.pixel_to_world(width / 2.0, height / 2.0)
        ra0 = float(center.ra.deg)
        dec0 = float(center.dec.deg)

        rot_rad = math.radians(-rot)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        cosd = max(0.01, math.cos(math.radians(dec0)))

        corners_en = [
            (-half_w_deg, half_h_deg),
            (half_w_deg, half_h_deg),
            (half_w_deg, -half_h_deg),
            (-half_w_deg, -half_h_deg),
        ]
        ra_corners = []
        dec_corners = []
        for de, dn in corners_en:
            re = de * cos_r - dn * sin_r
            rn = de * sin_r + dn * cos_r
            ra_corners.append(ra0 - re / cosd)
            dec_corners.append(dec0 + rn)

        xs, ys = _world_to_pixel(wcs, np.array(ra_corners), np.array(dec_corners))
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(
            canvas, [pts], isClosed=True, color=params.fov_box_color,
            thickness=1, lineType=cv2.LINE_AA,
        )
    except Exception:
        log.debug("Finder chart FOV box draw failed", exc_info=True)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def render_finder_chart(
    image: np.ndarray,
    wcs_header: Any,
    params: FinderChartParams | None = None,
    catalog_stars: list[Any] | None = None,
    dso_list: list[Any] | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Render an annotated finder chart from a plate-solved image.

    Parameters
    ----------
    image : ndarray
        (H, W) mono or (C, H, W) color float32 image in [0, 1]. Used as the
        chart background unless ``params.background`` is "black"/"white".
    wcs_header : astropy.io.fits.Header | dict | astropy.wcs.WCS
        The plate solution for ``image``.
    params : FinderChartParams, optional
        Rendering options. Defaults to :class:`FinderChartParams`.
    catalog_stars, dso_list : list, optional
        Pre-fetched catalog entries (dicts or objects) with
        ``ra_deg``/``ra``, ``dec_deg``/``dec``, optional ``mag``, ``name``,
        and (DSOs) ``size_arcmin``. No network access happens here — callers
        fetch from ``astraios.core.gaia_catalog`` / ``astraios.core.dso_catalog``
        or elsewhere and pass the results in.
    progress : callable(fraction, message), optional

    Returns
    -------
    ndarray
        (3, H, W) float32 RGB image in [0, 1].
    """
    params = params or FinderChartParams()
    wcs = _to_wcs(wcs_header)

    _emit(progress, 0.0, "Preparing background")
    canvas = _make_background_canvas(image, params)
    canvas, wcs = _fit_canvas(canvas, wcs, int(params.out_size))
    height, width = canvas.shape[:2]

    if params.show_grid:
        _draw_grid(canvas, width, height, params)

    if params.show_field_marker:
        _draw_field_marker(canvas, width, height, params.field_marker_color)

    _emit(progress, 0.3, "Plotting catalog stars")
    if params.show_stars and catalog_stars:
        stars = _prepare_markers(
            catalog_stars, wcs, width, height,
            params.star_mag_limit, params.star_max_labels, params.declutter_cell_px,
        )
        for star in stars:
            _draw_marker(
                canvas, star["x"], star["y"], params.star_color, params.marker_radius_px,
                label=star["name"] if (params.show_star_labels and star["name"]) else None,
                font_scale=params.label_font_scale, shape="circle",
            )

    _emit(progress, 0.55, "Plotting deep-sky objects")
    arcsec_per_px = _pixel_scale_arcsec(wcs)
    if params.show_dso and dso_list:
        dsos = _prepare_markers(
            dso_list, wcs, width, height,
            params.dso_mag_limit, params.dso_max_labels, params.declutter_cell_px,
        )
        for dso in dsos:
            _draw_marker(
                canvas, dso["x"], dso["y"], params.dso_color, params.marker_radius_px,
                label=dso["name"] if (params.show_dso_labels and dso["name"]) else None,
                font_scale=params.label_font_scale, shape="square",
            )
            if params.show_dso_size_circles and dso["size_arcmin"] and arcsec_per_px:
                diam_px = (dso["size_arcmin"] * 60.0) / arcsec_per_px
                radius_px = 0.5 * diam_px
                if np.isfinite(radius_px) and radius_px > 2:
                    cv2.circle(
                        canvas, (int(round(dso["x"])), int(round(dso["y"]))),
                        int(round(radius_px)), params.dso_circle_color, 1, cv2.LINE_AA,
                    )

    _emit(progress, 0.8, "Drawing compass and scale bar")
    if params.show_compass:
        _draw_compass(canvas, wcs, params, width, height)
    if params.show_scale_bar:
        _draw_scale_bar(canvas, wcs, params, width, height)
    if params.show_fov_box:
        _draw_fov_box(canvas, wcs, params, width, height)

    _emit(progress, 1.0, "Done")

    rgb01 = canvas.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb01, (2, 0, 1)))
