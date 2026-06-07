"""Equipment Profile System — camera, telescope, and filter data for Smart Processor.

Stores and manages astrophotography equipment profiles used by the
AI-based Smart Processor to make informed stacking and post-processing
decisions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraProfile:
    """An astrophotography camera sensor profile."""

    name: str
    sensor: str
    pixel_size_um: float  # micrometers
    read_noise_e: float  # electrons (at typical unity-gain setting)
    dark_current_e_per_s: float  # e-/s at 0 °C
    full_well_e: int
    qe_peak: float  # peak quantum efficiency 0-1
    qe_curve: list[tuple[float, float]]  # [(wavelength_nm, qe), ...]
    bayer_pattern: str | None  # "RGGB", "GRBG", … or None for mono
    resolution_x: int  # sensor width in pixels
    resolution_y: int  # sensor height in pixels
    camera_type: str  # "mono" or "color"

    # -- helpers -------------------------------------------------------------

    def qe_at(self, wavelength_nm: float) -> float:
        """Linearly interpolate QE at a given wavelength from *qe_curve*.

        Values outside the curve range are clamped to the nearest endpoint.
        """
        if not self.qe_curve:
            return self.qe_peak
        pts = sorted(self.qe_curve, key=lambda p: p[0])
        if wavelength_nm <= pts[0][0]:
            return pts[0][1]
        if wavelength_nm >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            w0, q0 = pts[i]
            w1, q1 = pts[i + 1]
            if w0 <= wavelength_nm <= w1:
                t = (wavelength_nm - w0) / (w1 - w0)
                return q0 + t * (q1 - q0)
        return self.qe_peak  # fallback

    @property
    def is_mono(self) -> bool:
        return self.camera_type == "mono"

    def to_dict(self) -> dict:
        d = asdict(self)
        # qe_curve: list of 2-element lists (JSON-friendly)
        d["qe_curve"] = [list(p) for p in self.qe_curve]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CameraProfile:
        d = dict(d)  # shallow copy
        d["qe_curve"] = [tuple(p) for p in d.get("qe_curve", [])]
        return cls(**d)


@dataclass
class TelescopeProfile:
    """An optical tube assembly / telescope profile."""

    name: str
    aperture_mm: float
    focal_length_mm: float
    focal_ratio: float
    telescope_type: str  # "refractor", "reflector", "catadioptric", "camera_lens"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TelescopeProfile:
        return cls(**d)


@dataclass
class FilterProfile:
    """A photographic filter profile (broadband, narrowband, or multi-band)."""

    name: str
    filter_type: str  # "broadband" | "narrowband" | "dual_narrowband" | "triple_narrowband" | "light_pollution"
    center_nm: float | None = None  # centre wavelength for single narrowband
    bandwidth_nm: float | None = None  # FWHM for single narrowband
    range_nm: tuple[float, float] | None = None  # (min, max) for broadband
    peak_transmission: float = 0.95  # 0-1
    pass_bands: list[dict] | None = None  # for dual/triple: [{"center_nm": …, "bandwidth_nm": …}, …]

    @property
    def effective_bandwidth_nm(self) -> float:
        """Return an effective bandwidth figure regardless of filter type."""
        if self.bandwidth_nm is not None:
            return self.bandwidth_nm
        if self.range_nm is not None:
            return self.range_nm[1] - self.range_nm[0]
        if self.pass_bands:
            return sum(b.get("bandwidth_nm", 0.0) for b in self.pass_bands)
        return 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert range_nm tuple to list for JSON serialisation.
        if d.get("range_nm") is not None:
            d["range_nm"] = list(d["range_nm"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FilterProfile:
        d = dict(d)
        if d.get("range_nm") is not None:
            d["range_nm"] = tuple(d["range_nm"])
        return cls(**d)


@dataclass
class EquipmentProfile:
    """A complete imaging setup: camera + telescope + filter set."""

    camera: CameraProfile
    telescope: TelescopeProfile
    filters: dict[str, FilterProfile] = field(default_factory=dict)

    # -- optical calculations ------------------------------------------------

    def plate_scale(self) -> float:
        """Image scale in arcseconds per pixel.

        Uses the standard formula:  plate_scale = 206.265 * pixel_um / FL_mm
        """
        return 206.265 * self.camera.pixel_size_um / self.telescope.focal_length_mm

    def theoretical_resolution(self, wavelength_nm: float = 550.0) -> float:
        """Rayleigh criterion angular resolution in arcseconds.

        theta = 1.22 * lambda / D  (in radians), converted to arcsec.
        Both wavelength and aperture are converted to metres before division.
        """
        wavelength_m = wavelength_nm * 1e-9  # nm -> m
        aperture_m = self.telescope.aperture_mm * 1e-3  # mm -> m
        theta_rad = 1.22 * wavelength_m / aperture_m
        return theta_rad * 206265.0

    def is_oversampled(self, measured_fwhm_arcsec: float | None = None) -> bool:
        """Determine whether the pixel scale over-samples the resolution element.

        If *measured_fwhm_arcsec* (e.g. from star FWHM) is provided it is used
        as the resolution element; otherwise the theoretical Rayleigh limit is
        used.  Over-sampling is defined as the pixel scale being less than half
        the resolution element (i.e. more than 2 pixels per FWHM).
        """
        resolution = measured_fwhm_arcsec if measured_fwhm_arcsec else self.theoretical_resolution()
        ps = self.plate_scale()
        # Nyquist: need ~2 pixels per FWHM.  Over-sampled when pixel scale
        # is significantly smaller than FWHM/2.
        return ps < resolution / 2.0

    def effective_throughput(self, filter_name: str) -> float:
        """Compute *QE * filter_transmission * bandwidth* for the named filter.

        For narrowband filters the bandwidth is taken directly; for broadband
        filters the passband width is used.  The QE is evaluated at the
        filter's central wavelength (or mid-range for broadband).
        """
        filt = self.filters.get(filter_name)
        if filt is None:
            raise KeyError(f"Filter '{filter_name}' not in equipment profile")

        # Determine the representative wavelength for QE lookup.
        if filt.center_nm is not None:
            wl = filt.center_nm
        elif filt.range_nm is not None:
            wl = (filt.range_nm[0] + filt.range_nm[1]) / 2.0
        elif filt.pass_bands:
            wl = filt.pass_bands[0]["center_nm"]
        else:
            wl = 550.0  # sensible default

        qe = self.camera.qe_at(wl)
        bandwidth = filt.effective_bandwidth_nm
        transmission = filt.peak_transmission

        return qe * transmission * bandwidth

    def field_of_view_arcmin(self) -> tuple[float, float]:
        """Return (width, height) field of view in arcminutes."""
        ps = self.plate_scale()  # arcsec/px
        fov_x = ps * self.camera.resolution_x / 60.0
        fov_y = ps * self.camera.resolution_y / 60.0
        return (fov_x, fov_y)

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "camera": self.camera.to_dict(),
            "telescope": self.telescope.to_dict(),
            "filters": {k: v.to_dict() for k, v in self.filters.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> EquipmentProfile:
        camera = CameraProfile.from_dict(d["camera"])
        telescope = TelescopeProfile.from_dict(d["telescope"])
        filters = {k: FilterProfile.from_dict(v) for k, v in d.get("filters", {}).items()}
        return cls(camera=camera, telescope=telescope, filters=filters)

    def save(self, path: Path) -> None:
        """Persist the profile to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        log.info("Equipment profile saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> EquipmentProfile:
        """Load a profile from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        log.info("Equipment profile loaded from %s", path)
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Database loaders
# ---------------------------------------------------------------------------

def load_camera_database(path: Path | None = None) -> list[CameraProfile]:
    """Load the built-in camera database (or a custom one) from JSON.

    Parameters
    ----------
    path : Path, optional
        Path to a ``cameras.json`` file.  Defaults to the bundled database
        inside ``cosmica/resources/``.
    """
    path = path or (_RESOURCES_DIR / "cameras.json")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    cameras = [CameraProfile.from_dict(entry) for entry in raw]
    log.debug("Loaded %d cameras from %s", len(cameras), path)
    return cameras


def load_telescope_database(path: Path | None = None) -> list[TelescopeProfile]:
    """Load the built-in telescope database (or a custom one) from JSON."""
    path = path or (_RESOURCES_DIR / "telescopes.json")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    telescopes = [TelescopeProfile.from_dict(entry) for entry in raw]
    log.debug("Loaded %d telescopes from %s", len(telescopes), path)
    return telescopes


def load_filter_database(path: Path | None = None) -> list[FilterProfile]:
    """Load the built-in filter database (or a custom one) from JSON."""
    path = path or (_RESOURCES_DIR / "filters.json")
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    filters = [FilterProfile.from_dict(entry) for entry in raw]
    log.debug("Loaded %d filters from %s", len(filters), path)
    return filters


# ---------------------------------------------------------------------------
# FITS header sniffing
# ---------------------------------------------------------------------------

# Common FITS keywords across ASCOM, INDI, NINA, SGPro, APT, etc.
_CAMERA_KEYWORDS = (
    "INSTRUME", "CCD-NAME", "CAMERA", "CAMNAME",
)
_FILTER_KEYWORDS = (
    "FILTER", "FILTNAME", "FILT-ID",
)
_BAYER_KEYWORDS = (
    "BAYERPAT", "COLORTYP", "CFA-PAT",
)
_TEMPERATURE_KEYWORDS = (
    "CCD-TEMP", "CCDTEMP", "SET-TEMP", "TEMPERAT",
)
_GAIN_KEYWORDS = (
    "GAIN", "EGAIN", "ISOSPEED", "ISO",
)
_OFFSET_KEYWORDS = (
    "OFFSET", "BLKLEVEL",
)


def detect_from_fits_header(header: dict[str, Any]) -> dict[str, Any]:
    """Extract equipment-related metadata from a FITS header dict.

    Returns a dictionary with keys (all optional, present only if detected):

    * ``camera_name`` -- camera / instrument name
    * ``exposure_s`` -- exposure time in seconds
    * ``gain`` -- gain / ISO value
    * ``offset`` -- offset / black level
    * ``filter`` -- filter name
    * ``temperature_c`` -- sensor temperature in Celsius
    * ``bayer_pattern`` -- Bayer matrix pattern string
    * ``binning`` -- (xbin, ybin) tuple
    * ``focal_length_mm`` -- focal length in mm
    * ``aperture_mm`` -- aperture in mm
    * ``pixel_size_um`` -- pixel size in micrometres
    * ``image_type`` -- frame type string (light, dark, flat, bias)
    * ``object_name`` -- target object name
    * ``ra`` -- right ascension (degrees or sexagesimal string)
    * ``dec`` -- declination (degrees or sexagesimal string)
    """
    info: dict[str, Any] = {}

    # Camera name
    for kw in _CAMERA_KEYWORDS:
        val = header.get(kw)
        if val and str(val).strip():
            info["camera_name"] = str(val).strip()
            break

    # Exposure
    for kw in ("EXPTIME", "EXPOSURE"):
        val = header.get(kw)
        if val is not None:
            try:
                info["exposure_s"] = float(val)
            except (ValueError, TypeError):
                pass
            break

    # Gain / ISO
    for kw in _GAIN_KEYWORDS:
        val = header.get(kw)
        if val is not None:
            try:
                info["gain"] = float(val)
            except (ValueError, TypeError):
                pass
            break

    # Offset
    for kw in _OFFSET_KEYWORDS:
        val = header.get(kw)
        if val is not None:
            try:
                info["offset"] = float(val)
            except (ValueError, TypeError):
                pass
            break

    # Filter
    for kw in _FILTER_KEYWORDS:
        val = header.get(kw)
        if val and str(val).strip():
            info["filter"] = str(val).strip()
            break

    # Temperature
    for kw in _TEMPERATURE_KEYWORDS:
        val = header.get(kw)
        if val is not None:
            try:
                info["temperature_c"] = float(val)
            except (ValueError, TypeError):
                pass
            break

    # Bayer pattern
    for kw in _BAYER_KEYWORDS:
        val = header.get(kw)
        if val and str(val).strip():
            info["bayer_pattern"] = str(val).strip().upper()
            break

    # Binning
    xbin = header.get("XBINNING") or header.get("XBIN")
    ybin = header.get("YBINNING") or header.get("YBIN")
    if xbin is not None and ybin is not None:
        try:
            info["binning"] = (int(xbin), int(ybin))
        except (ValueError, TypeError):
            pass

    # Focal length
    fl = header.get("FOCALLEN") or header.get("FOCAL")
    if fl is not None:
        try:
            info["focal_length_mm"] = float(fl)
        except (ValueError, TypeError):
            pass

    # Aperture
    ap = header.get("APTDIA") or header.get("APTDIAM")
    if ap is not None:
        try:
            info["aperture_mm"] = float(ap)
        except (ValueError, TypeError):
            pass

    # Pixel size
    px = header.get("XPIXSZ") or header.get("PIXSIZE1")
    if px is not None:
        try:
            info["pixel_size_um"] = float(px)
        except (ValueError, TypeError):
            pass

    # Image / frame type
    for kw in ("IMAGETYP", "FRAMETYPE", "FRAME"):
        val = header.get(kw)
        if val and str(val).strip():
            info["image_type"] = str(val).strip().lower()
            break

    # Object name
    obj = header.get("OBJECT") or header.get("OBJNAME")
    if obj and str(obj).strip():
        info["object_name"] = str(obj).strip()

    # Coordinates — normalise sexagesimal strings to decimal degrees
    ra_val = header.get("RA") or header.get("OBJCTRA")
    dec_val = header.get("DEC") or header.get("OBJCTDEC")
    if isinstance(ra_val, str) and isinstance(dec_val, str):
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            c = SkyCoord(ra_val, dec_val, unit=(u.hourangle, u.deg), frame="icrs")
            info["ra"] = c.ra.deg
            info["dec"] = c.dec.deg
        except Exception:
            info["ra"] = ra_val
            info["dec"] = dec_val
    else:
        if ra_val is not None:
            info["ra"] = ra_val
        if dec_val is not None:
            info["dec"] = dec_val

    return info
