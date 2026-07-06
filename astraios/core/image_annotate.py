"""Image Annotate ("What's In My Image") — structured catalog identification core.

SASpro's WIMI tool takes a plate-solved image, live-queries SIMBAD/Vizier for
every catalog object in the field, and renders/labels them over the image
inside a large interactive dialog (object tree, HR diagrams, 3D views, an
"open on SIMBAD/AstroBin" context menu, minor-body search, etc). Astraios
already has an annotated-chart renderer (:mod:`astraios.core.finder_chart`,
fed by :func:`astraios.core.dso_catalog.query_dso_in_field` and the local
Gaia catalog) that covers WIMI's *rendering* half.

What finder_chart does not provide is a **structured, queryable** result: a
plain list of "what is in this field" with pixel positions and metadata that
a caller can put in a table, click to re-center on, or filter — independent
of any drawing. That is what this module ports: :func:`identify_objects`
matches the embedded Messier/NGC/IC catalog (and, opt-in, a locally
installed Gaia DR3 catalog for bright-star identification) against a WCS +
image footprint and returns :class:`IdentifiedObject` entries.

This is a headless, network-free port: SASpro's SIMBAD/Vizier TAP queries,
AstroBin/website links, 3D minor-body viewer, and HR-diagram tooling are
intentionally not ported — ``astraios.core.simbad_lookup`` already covers
opt-in online lookups when a user wants more than the embedded catalog.

:func:`split_for_finder_chart` converts identified objects back into the
dict shape :func:`astraios.core.finder_chart.render_finder_chart` expects,
so a caller can annotate onto the image without reimplementing any drawing.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from astraios.core.dso_catalog import query_dso_in_field

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _emit(progress: ProgressCallback | None, fraction: float, message: str) -> None:
    if progress is None:
        return
    try:
        progress(fraction, message)
    except Exception:
        log.debug("Image annotate progress callback raised", exc_info=True)


@dataclass
class IdentifiedObject:
    """A single catalog object matched inside the image footprint."""

    name: str
    catalog: str  # "Messier" | "NGC" | "IC" | "DSO" | "Gaia DR3"
    type: str  # DSO type_code (G/N/OC/GC/PN/SNR/EN) or "Star"
    ra: float  # degrees, ICRS
    dec: float  # degrees, ICRS
    x: float  # pixel column (0-indexed)
    y: float  # pixel row (0-indexed)
    magnitude: float | None = None
    size: float | None = None  # angular size, arcmin (DSOs only; None for stars)


@dataclass
class AnnotateParams:
    """Settings controlling :func:`identify_objects`."""

    include_dso: bool = True
    max_dso: int = 300

    # Bright stars require a locally installed Gaia DR3 catalog (see
    # astraios.core.gaia_catalog) — opt-in and never touches the network.
    include_bright_stars: bool = False
    star_mag_limit: float = 9.0
    max_stars: int = 50
    gaia_catalog_dir: str | None = None

    # Field-of-view padding multiplier applied when deriving the search
    # radius from the image footprint (catches objects whose center falls
    # just outside the frame but whose marker/label would still be relevant).
    fov_margin: float = 1.5


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
    raise TypeError(f"Unsupported wcs_header type for image annotate: {type(wcs_header)!r}")


def _image_hw(image_shape: Any) -> tuple[int, int]:
    """Return (height, width) from a (H,W) or (C,H,W) shape tuple."""
    shape = tuple(image_shape)
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[-2]), int(shape[-1])
    raise ValueError(f"Unsupported image_shape for image annotate: {image_shape!r}")


def _field_center_and_fov(
    wcs: WCS, height: int, width: int, margin: float
) -> tuple[float, float, float] | None:
    """Return (ra_deg, dec_deg, fov_deg) for the image footprint, or None."""
    try:
        center = wcs.pixel_to_world(width / 2.0, height / 2.0)
        ra = float(center.ra.deg)
        dec = float(center.dec.deg)
    except Exception:
        log.debug("Image annotate: could not derive field center from WCS", exc_info=True)
        return None

    try:
        scales = proj_plane_pixel_scales(wcs)
        deg_per_px = float(np.nanmedian(scales))
    except Exception:
        deg_per_px = float("nan")
    if not np.isfinite(deg_per_px) or deg_per_px <= 0:
        return None

    fov_deg = max(width, height) * deg_per_px * float(margin)
    if not np.isfinite(fov_deg) or fov_deg <= 0:
        return None
    return ra, dec, fov_deg


def _world_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(coord)
    return float(x), float(y)


# --------------------------------------------------------------------------
# DSO identification (embedded Messier/NGC/IC catalog — always network-free)
# --------------------------------------------------------------------------


def _dso_catalog_name(name: str) -> str:
    if name.startswith("NGC"):
        return "NGC"
    if name.startswith("IC"):
        return "IC"
    if len(name) > 1 and name[0] == "M" and name[1].isdigit():
        return "Messier"
    return "DSO"


def _identify_dso(
    wcs: WCS, ra: float, dec: float, fov_deg: float, height: int, width: int,
    params: AnnotateParams,
) -> list[IdentifiedObject]:
    entries = query_dso_in_field(ra, dec, fov_deg)
    out: list[IdentifiedObject] = []
    for entry in entries:
        try:
            x, y = _world_to_pixel(wcs, entry.ra_deg, entry.dec_deg)
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if not (0.0 <= x < width and 0.0 <= y < height):
            continue
        out.append(IdentifiedObject(
            name=entry.name,
            catalog=_dso_catalog_name(entry.name),
            type=entry.type_code,
            ra=float(entry.ra_deg),
            dec=float(entry.dec_deg),
            x=x, y=y,
            magnitude=None,
            size=float(entry.size_arcmin) if entry.size_arcmin is not None else None,
        ))
        if len(out) >= max(1, int(params.max_dso)):
            break
    out.sort(key=lambda o: o.name)
    return out


# --------------------------------------------------------------------------
# Bright-star identification (local Gaia DR3 catalog — opt-in, no network)
# --------------------------------------------------------------------------


def _identify_bright_stars(
    wcs: WCS, ra: float, dec: float, fov_deg: float, height: int, width: int,
    params: AnnotateParams,
) -> list[IdentifiedObject]:
    try:
        from astraios.core.gaia_catalog import GaiaCatalog, GaiaCatalogError
    except Exception:
        log.debug("Image annotate: gaia_catalog module unavailable", exc_info=True)
        return []

    try:
        with GaiaCatalog(params.gaia_catalog_dir) as cat:
            sources = cat.cone_query(
                ra, dec, max(fov_deg / 2.0, 0.01),
                mag_limit=params.star_mag_limit, max_stars=params.max_stars,
            )
    except GaiaCatalogError:
        log.info("Image annotate: no local Gaia catalog installed for bright-star ID")
        return []
    except Exception:
        log.exception("Image annotate: bright-star cone query failed")
        return []

    out: list[IdentifiedObject] = []
    for src in sources:
        try:
            x, y = _world_to_pixel(wcs, src.ra, src.dec)
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if not (0.0 <= x < width and 0.0 <= y < height):
            continue
        out.append(IdentifiedObject(
            name=f"Gaia DR3 {src.source_id}",
            catalog="Gaia DR3",
            type="Star",
            ra=float(src.ra), dec=float(src.dec),
            x=x, y=y,
            magnitude=float(src.mag) if src.mag is not None else None,
            size=None,
        ))
    out.sort(key=lambda o: o.magnitude if o.magnitude is not None else float("inf"))
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def identify_objects(
    image_shape: Any,
    wcs_header: Any,
    params: AnnotateParams | None = None,
    progress: ProgressCallback | None = None,
) -> list[IdentifiedObject]:
    """Identify catalog objects that fall inside a plate-solved image's footprint.

    Parameters
    ----------
    image_shape : tuple
        ``(H, W)`` or ``(C, H, W)`` — only the spatial extent is used.
    wcs_header : astropy.io.fits.Header | dict | astropy.wcs.WCS
        The plate solution for the image.
    params : AnnotateParams, optional
        Which catalogs to match and how (magnitude limits, caps). Defaults
        to DSO-only, network-free identification.
    progress : callable(fraction, message), optional

    Returns
    -------
    list[IdentifiedObject]
        Deterministically ordered: DSOs sorted by name, then bright stars
        (if requested) sorted brightest-first. Empty if the WCS has no
        usable pixel scale, the footprint is degenerate, or no catalog
        entries fall inside the frame.
    """
    params = params or AnnotateParams()
    _emit(progress, 0.0, "Resolving field")

    try:
        wcs = _to_wcs(wcs_header)
    except Exception:
        log.debug("Image annotate: could not build WCS", exc_info=True)
        return []

    height, width = _image_hw(image_shape)
    if height <= 0 or width <= 0:
        return []

    field = _field_center_and_fov(wcs, height, width, params.fov_margin)
    if field is None:
        return []
    ra, dec, fov_deg = field

    results: list[IdentifiedObject] = []

    if params.include_dso:
        _emit(progress, 0.2, "Matching deep-sky catalog")
        results.extend(_identify_dso(wcs, ra, dec, fov_deg, height, width, params))

    if params.include_bright_stars:
        _emit(progress, 0.6, "Matching bright stars")
        results.extend(_identify_bright_stars(wcs, ra, dec, fov_deg, height, width, params))

    _emit(progress, 1.0, "Done")
    return results


def split_for_finder_chart(
    objects: list[IdentifiedObject],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert :func:`identify_objects` output into finder_chart-ready dicts.

    Returns ``(catalog_stars, dso_list)`` — plain dicts using the
    ``ra_deg``/``dec_deg``/``mag``/``name``/``size_arcmin`` keys that
    :func:`astraios.core.finder_chart.render_finder_chart`'s duck-typed
    entry reader expects, so identified objects can be annotated onto the
    image without reimplementing any marker drawing.
    """
    stars: list[dict[str, Any]] = []
    dsos: list[dict[str, Any]] = []
    for obj in objects:
        entry = {
            "ra_deg": obj.ra,
            "dec_deg": obj.dec,
            "mag": obj.magnitude,
            "name": obj.name,
            "size_arcmin": obj.size,
        }
        (stars if obj.type == "Star" else dsos).append(entry)
    return stars, dsos
