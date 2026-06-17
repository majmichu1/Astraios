"""Target Catalog — astronomical object database for Smart Processor.

Provides per-object metadata including coordinates, angular size, surface brightness,
emission line information, and processing hints. Used by the Smart Processor to
identify targets and optimize processing parameters.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Path to the bundled catalog data
_CATALOG_JSON = Path(__file__).resolve().parent.parent / "resources" / "catalog.json"


@dataclass
class TargetInfo:
    """Information about an astronomical target."""

    id: str  # primary designation, e.g. "M42", "NGC 7000"
    names: list[str]  # all common names
    ra_deg: float  # J2000 right ascension in degrees
    dec_deg: float  # J2000 declination in degrees
    angular_size_arcmin: tuple[float, float]  # (major, minor) axis in arcminutes
    object_type: str  # "emission_nebula", "reflection_nebula", "planetary_nebula",
    # "supernova_remnant", "galaxy_spiral", "galaxy_elliptical",
    # "galaxy_irregular", "open_cluster", "globular_cluster",
    # "dark_nebula", "hii_region"
    magnitude: float | None  # visual magnitude (None if not applicable)
    surface_brightness: float | None  # mag/arcsec^2 (None if not known)
    brightness_class: str  # "very_bright", "bright", "moderate", "faint", "very_faint"
    dynamic_range: str  # "low", "moderate", "high", "extreme"
    emission_lines: list[str]  # e.g. ["Ha", "OIII", "SII", "NII", "Hb", "CaII"]
    dominant_emission: str | None  # primary emission line
    constellation: str
    processing_hints: dict[str, Any]  # structured hints for the Smart Processor

    # ---- helpers ---------------------------------------------------------- #

    @property
    def major_axis_arcmin(self) -> float:
        return self.angular_size_arcmin[0]

    @property
    def minor_axis_arcmin(self) -> float:
        return self.angular_size_arcmin[1]

    def covers_fov(self, fov_arcmin: float) -> bool:
        """Return True when the target is larger than the given FOV."""
        return self.major_axis_arcmin > fov_arcmin

    def __repr__(self) -> str:
        names = f" ({', '.join(self.names)})" if self.names else ""
        return f"<TargetInfo {self.id}{names} {self.object_type}>"


class CatalogDB:
    """Local catalog database for target identification.

    The catalog is lazily loaded from the bundled JSON file the first time
    any query method is called.
    """

    def __init__(self) -> None:
        self._targets: list[TargetInfo] = []
        self._by_id: dict[str, TargetInfo] = {}
        self._by_name: dict[str, TargetInfo] = {}  # lowercase name -> target
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Loading                                                            #
    # ------------------------------------------------------------------ #

    def load(self, path: Path | None = None) -> None:
        """Load the catalog from bundled JSON data.

        Parameters
        ----------
        path : Path, optional
            Override path to the JSON file. Defaults to the bundled catalog.
        """
        src = path or _CATALOG_JSON
        if not src.exists():
            log.error("Catalog file not found: %s", src)
            self._loaded = True
            return

        log.info("Loading target catalog from %s", src)
        with open(src, "r", encoding="utf-8") as fh:
            raw: list[dict[str, Any]] = json.load(fh)

        for entry in raw:
            try:
                target = TargetInfo(
                    id=entry["id"],
                    names=entry.get("names", []),
                    ra_deg=float(entry["ra_deg"]),
                    dec_deg=float(entry["dec_deg"]),
                    angular_size_arcmin=tuple(entry["angular_size_arcmin"]),  # type: ignore[arg-type]
                    object_type=entry["object_type"],
                    magnitude=entry.get("magnitude"),
                    surface_brightness=entry.get("surface_brightness"),
                    brightness_class=entry.get("brightness_class", "moderate"),
                    dynamic_range=entry.get("dynamic_range", "moderate"),
                    emission_lines=entry.get("emission_lines", []),
                    dominant_emission=entry.get("dominant_emission"),
                    constellation=entry.get("constellation", ""),
                    processing_hints=entry.get("processing_hints", {}),
                )
                self._targets.append(target)
                self._by_id[target.id.upper()] = target
                for name in target.names:
                    self._by_name[name.lower()] = target
                # Also index by primary id lowered
                self._by_name[target.id.lower()] = target
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Skipping malformed catalog entry: %s — %s", entry.get("id", "?"), exc)

        self._loaded = True
        log.info("Loaded %d targets into catalog", len(self._targets))

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------ #
    #  Lookups                                                            #
    # ------------------------------------------------------------------ #

    def lookup(self, name: str) -> TargetInfo | None:
        """Look up a target by designation or common name.

        The search is case-insensitive. Both primary designations (e.g. ``M 42``,
        ``NGC 7000``) and common names (e.g. ``Orion Nebula``) are accepted.

        Parameters
        ----------
        name : str
            Target designation or common name.

        Returns
        -------
        TargetInfo or None
            The matched target, or *None* if not found.
        """
        self._ensure_loaded()
        key = name.strip().lower()

        # Direct match
        if key in self._by_name:
            return self._by_name[key]

        # Try uppercased ID (handles "m42" -> "M42")
        upper = name.strip().upper()
        if upper in self._by_id:
            return self._by_id[upper]

        # Normalise spacing: "M 42" -> "M42", "NGC 7000" -> "NGC7000"
        compressed = key.replace(" ", "")
        for tid, target in self._by_id.items():
            if tid.replace(" ", "") == compressed.upper():
                return target

        # Substring match on common names (useful for partial names)
        for tname, target in self._by_name.items():
            if key in tname:
                return target

        return None

    def lookup_all(self, name: str) -> list[TargetInfo]:
        """Return all targets matching *name* (may be more than one)."""
        self._ensure_loaded()
        key = name.strip().lower()
        results: list[TargetInfo] = []
        seen: set[str] = set()
        for tname, target in self._by_name.items():
            if key in tname and target.id not in seen:
                results.append(target)
                seen.add(target.id)
        return results

    # ------------------------------------------------------------------ #
    #  Spatial queries                                                    #
    # ------------------------------------------------------------------ #

    def query_region(
        self,
        ra_deg: float,
        dec_deg: float,
        fov_arcmin: float,
    ) -> list[TargetInfo]:
        """Find all targets within a circular field of view.

        Parameters
        ----------
        ra_deg : float
            Centre RA in degrees (J2000).
        dec_deg : float
            Centre Dec in degrees (J2000).
        fov_arcmin : float
            Field of view **radius** in arcminutes.

        Returns
        -------
        list[TargetInfo]
            Targets within the FOV, sorted by angular distance from centre.
        """
        self._ensure_loaded()
        results: list[tuple[float, TargetInfo]] = []
        for target in self._targets:
            sep = self.angular_separation(ra_deg, dec_deg, target.ra_deg, target.dec_deg)
            # Include target if its centre is inside or its extent overlaps
            effective_radius = fov_arcmin + target.major_axis_arcmin / 2.0
            if sep <= effective_radius:
                results.append((sep, target))
        results.sort(key=lambda t: t[0])
        return [t for _, t in results]

    def query_field(
        self,
        ra_deg: float,
        dec_deg: float,
        width_arcmin: float,
        height_arcmin: float,
    ) -> list[TargetInfo]:
        """Find targets in a rectangular field.

        The rectangle is aligned with RA/Dec axes. This is an approximation
        (flat-sky) suitable for typical astrophotography FOVs (< 5 degrees).

        Parameters
        ----------
        ra_deg, dec_deg : float
            Centre of the field.
        width_arcmin, height_arcmin : float
            Field dimensions in arcminutes.

        Returns
        -------
        list[TargetInfo]
            Targets within the field.
        """
        self._ensure_loaded()
        half_w = width_arcmin / 2.0
        half_h = height_arcmin / 2.0
        cos_dec = math.cos(math.radians(dec_deg)) or 1e-10
        # Convert RA width to degrees
        ra_half = (half_w / 60.0) / cos_dec
        dec_half = half_h / 60.0

        results: list[tuple[float, TargetInfo]] = []
        for target in self._targets:
            dra = abs(target.ra_deg - ra_deg)
            if dra > 180:
                dra = 360 - dra
            ddec = abs(target.dec_deg - dec_deg)

            obj_ra_extent = (target.major_axis_arcmin / 60.0 / 2.0) / cos_dec
            obj_dec_extent = target.minor_axis_arcmin / 60.0 / 2.0

            if dra <= (ra_half + obj_ra_extent) and ddec <= (dec_half + obj_dec_extent):
                sep = self.angular_separation(ra_deg, dec_deg, target.ra_deg, target.dec_deg)
                results.append((sep, target))

        results.sort(key=lambda t: t[0])
        return [t for _, t in results]

    # ------------------------------------------------------------------ #
    #  Utility                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def angular_separation(
        ra1: float,
        dec1: float,
        ra2: float,
        dec2: float,
    ) -> float:
        """Compute angular separation in arcminutes between two positions.

        Uses the Vincenty formula (numerically stable variant of the
        haversine for all distances).

        Parameters
        ----------
        ra1, dec1, ra2, dec2 : float
            Positions in degrees (J2000).

        Returns
        -------
        float
            Angular separation in **arcminutes**.
        """
        ra1_r = math.radians(ra1)
        dec1_r = math.radians(dec1)
        ra2_r = math.radians(ra2)
        dec2_r = math.radians(dec2)
        dra = ra2_r - ra1_r

        cos_dec2 = math.cos(dec2_r)
        sin_dec2 = math.sin(dec2_r)
        cos_dec1 = math.cos(dec1_r)
        sin_dec1 = math.sin(dec1_r)
        cos_dra = math.cos(dra)
        sin_dra = math.sin(dra)

        num1 = cos_dec2 * sin_dra
        num2 = cos_dec1 * sin_dec2 - sin_dec1 * cos_dec2 * cos_dra
        numerator = math.sqrt(num1 * num1 + num2 * num2)
        denominator = sin_dec1 * sin_dec2 + cos_dec1 * cos_dec2 * cos_dra

        sep_rad = math.atan2(numerator, denominator)
        return math.degrees(sep_rad) * 60.0  # convert degrees -> arcminutes

    @property
    def targets(self) -> list[TargetInfo]:
        """All loaded targets."""
        self._ensure_loaded()
        return list(self._targets)

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._targets)

    def __iter__(self):
        self._ensure_loaded()
        return iter(self._targets)

    def __repr__(self) -> str:
        n = len(self._targets) if self._loaded else "?"
        return f"<CatalogDB targets={n}>"
