"""What's In My Sky — observation-planning core (visibility, transit, moon/sun).

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright Franklin
Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Given an observer's location, date/timezone, and a set of filters (minimum
altitude, object types, magnitude limit), computes which catalog deep-sky
objects are observable during the night of the given date: their transit
time, peak altitude, and hours above the minimum altitude during the dark
part of the night — plus a sun/moon summary (rise/transit/set, phase,
per-object Moon separation).

Object source: the embedded catalog in ``astraios.core.dso_catalog``
(``all_entries()``) — no external catalog file or network access, matching
SASpro's bundled-catalog mode. SASpro's live SIMBAD/AAVSO/AstroBin lookups,
custom user catalogs, and the horizon-mask editor are out of scope for this
core port (UI-adjacent / network-dependent).

Visibility algorithm (ported from SASpro's ``_compute_alt_curve`` /
``ObjectVisibilityDialog._compute_and_plot`` in ``wims.py``):

1. Build a time grid of ``n`` samples spanning 24 h from local noon on the
   observing date to local noon the next day (so the plotted night is
   contiguous, not wrapped at midnight).
2. Transform every catalog object (as one broadcast ``SkyCoord`` array) plus
   the Sun and Moon to ``AltAz`` at every sample — one vectorized
   ``transform_to`` call for all objects, not a Python loop.
3. "Night" is wherever the Sun's altitude is below the selected twilight
   threshold (civil -6 deg, nautical -12 deg, astronomical -18 deg) —
   SASpro hardcodes astronomical (-18 deg) for its imaging-window calculation;
   this port exposes the threshold as a parameter.
4. Per object: ``max_altitude`` = peak altitude over the grid; ``transit_time``
   = local time of that peak; ``hours_visible`` = count of samples where
   altitude >= ``min_altitude_deg`` AND it is night, times ``24 / n`` hours
   per sample (SASpro: ``imaging_hrs = sum(imaging_mask) * (24.0 / len(times))``).
5. Moon separation per object is evaluated at that object's transit sample
   (SASpro evaluates it at a single query instant; here there is no single
   instant, so the transit — the most relevant moment for that object — is
   used instead).
6. Sun/Moon rise/set/transit come from the sign of their altitude curve
   crossing zero (rise = altitude going negative-to-positive, set = the
   reverse) — same crossing-detection SASpro uses for the object's own
   rise/set in ``ObjectVisibilityDialog``. Circumpolar / never-rises bodies
   report ``None`` for rise and set.
7. Moon phase percentage and waxing/waning classification: SASpro's
   ``calculate_lunar_phase`` — illuminated fraction ``(1 - cos(elongation)) / 2``
   from the Sun-Moon elongation, with waxing/waning determined by comparing
   elongation 6 hours later.

Sigma clipping / normalization are not applicable here — this module never
touches pixel data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


#: SASpro hardcodes astronomical (-18 deg) for its imaging-window score; this port
#: exposes all three standard twilight definitions.
_TWILIGHT_THRESHOLDS_DEG: dict[str, float] = {
    "civil": -6.0,
    "nautical": -12.0,
    "astronomical": -18.0,
}

#: All type codes present in astraios.core.dso_catalog.
ALL_OBJECT_TYPES: tuple[str, ...] = ("G", "N", "OC", "GC", "PN", "SNR", "EN")

_TYPE_LABELS: dict[str, str] = {
    "G": "Galaxy",
    "N": "Nebula",
    "OC": "Open Cluster",
    "GC": "Globular Cluster",
    "PN": "Planetary Nebula",
    "SNR": "Supernova Remnant",
    "EN": "Emission Nebula",
}


@dataclass
class SkyPlanParams:
    """Observer location, night, and filters for a sky-visibility plan.

    Attributes:
        latitude_deg: Observer latitude, degrees (+N).
        longitude_deg: Observer longitude, degrees (+E).
        elevation_m: Observer elevation above sea level, meters.
        date: Local calendar date of the observing night, ``"YYYY-MM-DD"``.
            The planned night runs from local noon on this date to local
            noon the next day (matches SASpro's noon-to-noon convention).
        timezone: IANA timezone name (e.g. ``"America/Denver"``). Falls
            back to UTC (with a note in the result's ``warnings``) if not
            recognized.
        min_altitude_deg: Minimum altitude, degrees, an object must reach
            during the night to count as "visible" / to accumulate
            ``hours_visible``.
        max_magnitude: Faintest magnitude to include, or ``None`` for no
            magnitude limit. Objects with unknown (``None``) catalog
            magnitude are never excluded by this filter (see
            ``dso_catalog._MAGNITUDES``).
        object_types: Type codes to include (subset of ``ALL_OBJECT_TYPES``),
            or ``None`` for all types.
        twilight: Which twilight band defines "night" for the
            ``hours_visible`` calculation — one of ``"civil"``,
            ``"nautical"``, ``"astronomical"``.
        time_resolution_min: Sample spacing across the 24 h grid, minutes.
            SASpro uses a fixed 5-minute grid (288 samples); this port
            derives the sample count from this parameter but keeps 5 as
            the default.
    """

    latitude_deg: float
    longitude_deg: float
    elevation_m: float = 0.0
    date: str = ""
    timezone: str = "UTC"
    min_altitude_deg: float = 20.0
    max_magnitude: float | None = None
    object_types: tuple[str, ...] | None = None
    twilight: str = "astronomical"
    time_resolution_min: float = 5.0


@dataclass
class VisibleObject:
    """A single catalog object's observability summary for the planned night."""

    name: str
    type_code: str
    type_label: str
    ra_deg: float
    dec_deg: float
    magnitude: float | None
    size_arcmin: float
    transit_time_local: str  # "HH:MM", or "n/a" if never above the horizon
    max_altitude_deg: float
    hours_visible: float
    moon_separation_deg: float


@dataclass
class SkyPlanResult:
    """Full result of a sky plan: visible objects plus the night's sun/moon summary."""

    objects: list[VisibleObject] = field(default_factory=list)
    moon_phase_pct: int = 0
    moon_phase_label: str = ""
    moon_rise_local: str | None = None
    moon_transit_local: str | None = None
    moon_set_local: str | None = None
    sun_rise_local: str | None = None
    sun_set_local: str | None = None
    twilight_evening_local: str | None = None
    twilight_morning_local: str | None = None
    local_sidereal_time: str = ""
    n_catalog_total: int = 0
    n_after_filters: int = 0
    warnings: list[str] = field(default_factory=list)
    message: str = "Calculation complete."


def _resolve_timezone(name: str, warnings: list[str]) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        warnings.append(f"Timezone '{name}' not recognized — using UTC instead.")
        return ZoneInfo("UTC")


def _to_local_hhmm(dt_utc, tz: ZoneInfo) -> str:
    return str(dt_utc.astimezone(tz).strftime("%H:%M"))


def _find_crossing(alts: np.ndarray, times, tz: ZoneInfo, rising: bool) -> str | None:
    """Return the local HH:MM of the first sign crossing of *alts* through zero.

    ``rising=True`` looks for a negative-to-positive crossing (rise);
    ``rising=False`` looks for positive-to-negative (set). Returns ``None`` if
    the body never crosses (circumpolar, or never rises) during the grid.
    """
    signs = np.sign(alts)
    crossings = np.where(np.diff(signs))[0]
    for ci in crossings:
        went_up = alts[ci + 1] > alts[ci]
        if went_up == rising:
            # Linear-interpolate the local datetime between the two straddling samples.
            t0, t1 = times[ci].to_datetime(), times[ci + 1].to_datetime()
            frac = (0.0 - alts[ci]) / (alts[ci + 1] - alts[ci])
            dt = t0 + (t1 - t0) * frac
            return _to_local_hhmm(dt.replace(tzinfo=timezone.utc), tz)
    return None


def _moon_phase(elong_deg: float, is_waxing: bool) -> tuple[int, str]:
    """SASpro's ``calculate_lunar_phase`` elongation-bucket classifier (text labels).

    Ported thresholds from ``wims.py``: buckets at 9/18/27/36/45/54/90/108/126/144/162
    degrees of Sun-Moon elongation. Between 54 deg and 90 deg SASpro reports a plain
    "First Quarter" regardless of waxing/waning (an apparent source quirk/simplification
    around exact quarter, preserved here for fidelity).
    """
    phase_pct = int(round((1 - np.cos(np.radians(elong_deg))) / 2 * 100))
    e = elong_deg
    if e < 9:
        label = "New Moon"
    elif e < 54:
        label = "Waxing Crescent" if is_waxing else "Waning Crescent"
    elif e < 90:
        label = "First Quarter"
    elif e < 162:
        label = "Waxing Gibbous" if is_waxing else "Waning Gibbous"
    else:
        label = "Full Moon"
    return phase_pct, label


def _setup_astropy() -> None:
    """Disable IERS auto-download once per process (SASpro's own guard).

    Never touch the network: use whatever IERS (Earth orientation) table ships
    with astropy instead of trying to fetch a fresher one.
    """
    from astropy.utils import iers

    iers.conf.auto_download = False
    iers.conf.auto_max_age = None


@dataclass
class _NightGrid:
    """Shared noon-to-noon time/AltAz grid for a given observer + night."""

    tz: ZoneInfo
    loc: Any  # astropy EarthLocation
    times: Any  # astropy Time array, shape (n,)
    frame: Any  # astropy AltAz frame over `times`
    sun_alts: np.ndarray
    moon_alts: np.ndarray
    hours_per_sample: float
    local_sidereal_time: str
    warnings: list[str]


def _build_night_grid(params: SkyPlanParams) -> _NightGrid:
    """Build the noon-to-noon time/AltAz grid and Sun/Moon altitude tracks.

    Shared by ``plan_sky`` (whole-catalog planning) and
    ``object_altitude_curve`` (single-object plot) so both use exactly the
    same grid, resolution, and Sun/Moon tracks.
    """
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
    from astropy.time import Time

    _setup_astropy()
    warnings: list[str] = []

    tz = _resolve_timezone(params.timezone, warnings)
    try:
        naive_noon = datetime.strptime(params.date, "%Y-%m-%d").replace(hour=12, minute=0)
    except ValueError:
        naive_noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        warnings.append(f"Date '{params.date}' not recognized — using today instead.")
    local_noon = naive_noon.replace(tzinfo=tz)
    t_start = Time(local_noon)

    loc = EarthLocation(
        lat=params.latitude_deg * u.deg,
        lon=params.longitude_deg * u.deg,
        height=params.elevation_m * u.m,
    )

    n = max(48, int(round(24.0 * 60.0 / max(params.time_resolution_min, 1.0))))
    hours_per_sample = 24.0 / n
    times = t_start + np.linspace(0, 24, n) * u.hour
    frame = AltAz(obstime=times, location=loc)

    lst = t_start.sidereal_time("apparent", params.longitude_deg * u.deg)
    local_sidereal_time = lst.to_string(unit=u.hour, precision=3)

    sun_alts = get_sun(times).transform_to(frame).alt.deg
    moon_alts = get_body("moon", times, loc).transform_to(frame).alt.deg

    return _NightGrid(
        tz=tz, loc=loc, times=times, frame=frame,
        sun_alts=sun_alts, moon_alts=moon_alts,
        hours_per_sample=hours_per_sample,
        local_sidereal_time=local_sidereal_time,
        warnings=warnings,
    )


def plan_sky(
    params: SkyPlanParams,
    progress: ProgressCallback | None = None,
) -> SkyPlanResult:
    """Compute tonight's observable objects and the sun/moon summary.

    Pure function, CPU/astropy only — no GPU, no network. Uses astropy's
    builtin JPL/ERFA ephemeris for the Sun and Moon and the embedded catalog
    in ``astraios.core.dso_catalog`` for objects.

    Args:
        params: Observer location, night, and filters.
        progress: Optional ``(fraction, message)`` callback.

    Returns:
        A ``SkyPlanResult`` with the filtered/ranked object list and the
        night's sun/moon summary.
    """
    progress = progress or _noop_progress

    # Deferred imports: keep astropy off the module import path for callers that
    # never actually plan a night (mirrors the lazy-import style used elsewhere
    # in astraios.core for optional/heavy dependencies).
    import astropy.units as u
    from astropy.coordinates import SkyCoord, get_body, get_sun

    from astraios.core.dso_catalog import all_entries

    progress(0.0, "Setting up observer and time grid...")
    grid = _build_night_grid(params)
    tz, loc, times, frame = grid.tz, grid.loc, grid.times, grid.frame
    sun_alts, moon_alts = grid.sun_alts, grid.moon_alts
    hours_per_sample = grid.hours_per_sample
    local_sidereal_time = grid.local_sidereal_time
    warnings = grid.warnings
    n = len(sun_alts)

    progress(0.1, "Computing Sun and Moon positions...")
    moon = get_body("moon", times, loc)
    moon_altaz = moon.transform_to(frame)

    twilight_threshold = _TWILIGHT_THRESHOLDS_DEG.get(
        params.twilight, _TWILIGHT_THRESHOLDS_DEG["astronomical"]
    )
    night_mask = sun_alts < twilight_threshold

    sun_rise = _find_crossing(sun_alts, times, tz, rising=True)
    sun_set = _find_crossing(sun_alts, times, tz, rising=False)
    twilight_evening = _find_crossing(sun_alts - twilight_threshold, times, tz, rising=False)
    twilight_morning = _find_crossing(sun_alts - twilight_threshold, times, tz, rising=True)

    moon_rise = _find_crossing(moon_alts, times, tz, rising=True)
    moon_set = _find_crossing(moon_alts, times, tz, rising=False)
    moon_transit_idx = int(np.argmax(moon_alts))
    moon_transit = _to_local_hhmm(
        times[moon_transit_idx].to_datetime().replace(tzinfo=timezone.utc), tz
    )

    # Moon phase at local midnight (the middle of the noon-to-noon grid).
    mid_idx = n // 2
    t_mid = times[mid_idx]
    moon_mid = get_body("moon", t_mid, loc)
    sun_mid = get_sun(t_mid)
    elong = float(moon_mid.separation(sun_mid).deg)
    future = t_mid + 6 * u.hour
    is_waxing = bool(
        get_body("moon", future, loc).separation(get_sun(future)).deg > elong
    )
    moon_phase_pct, moon_phase_label = _moon_phase(elong, is_waxing)

    progress(0.2, "Filtering catalog...")
    entries = all_entries()
    n_total = len(entries)

    if params.object_types:
        wanted = set(params.object_types)
        entries = [e for e in entries if e.type_code in wanted]
    if params.max_magnitude is not None:
        entries = [
            e for e in entries if e.magnitude is None or e.magnitude <= params.max_magnitude
        ]

    if not entries:
        progress(1.0, "No catalog objects matched the filters.")
        return SkyPlanResult(
            objects=[],
            moon_phase_pct=moon_phase_pct,
            moon_phase_label=moon_phase_label,
            moon_rise_local=moon_rise,
            moon_transit_local=moon_transit,
            moon_set_local=moon_set,
            sun_rise_local=sun_rise,
            sun_set_local=sun_set,
            twilight_evening_local=twilight_evening,
            twilight_morning_local=twilight_morning,
            local_sidereal_time=local_sidereal_time,
            n_catalog_total=n_total,
            n_after_filters=0,
            warnings=warnings,
            message="No catalog objects matched the filters.",
        )

    progress(0.3, f"Computing altitude curves for {len(entries)} objects...")
    ra_arr = np.array([e.ra_deg for e in entries])
    dec_arr = np.array([e.dec_deg for e in entries])
    sky = SkyCoord(ra=ra_arr * u.deg, dec=dec_arr * u.deg, frame="icrs")

    # One broadcast transform for every object against every time sample:
    # shape (N, 1) x (M,) -> (N, M). This is the vectorization SASpro's per-object
    # Python-loop score/curve helpers (`_score_targets_batch`, `_compute_alt_curve`)
    # do not do — here it is a single astropy call for the whole catalog.
    altaz = sky[:, None].transform_to(frame)
    obj_alts = altaz.alt.deg  # (N, M)

    progress(0.7, "Ranking observable objects...")
    max_alt = obj_alts.max(axis=1)
    transit_idx = obj_alts.argmax(axis=1)

    visible_mask = (obj_alts >= params.min_altitude_deg) & night_mask[None, :]
    hours_visible = visible_mask.sum(axis=1) * hours_per_sample

    # Moon separation evaluated at each object's own transit sample — one broadcast
    # separation() call between the per-object AltAz array and the Moon's AltAz
    # track already computed above, indexed at each object's transit_idx.
    moon_seps_all = altaz.separation(moon_altaz[None, :]).deg  # (N, M)

    results: list[VisibleObject] = []
    for i, entry in enumerate(entries):
        if hours_visible[i] <= 0:
            continue
        ti = int(transit_idx[i])
        transit_dt = times[ti].to_datetime().replace(tzinfo=timezone.utc)
        transit_local = _to_local_hhmm(transit_dt, tz)
        moon_sep = float(moon_seps_all[i, ti])
        results.append(
            VisibleObject(
                name=entry.name,
                type_code=entry.type_code,
                type_label=_TYPE_LABELS.get(entry.type_code, entry.type_code),
                ra_deg=entry.ra_deg,
                dec_deg=entry.dec_deg,
                magnitude=entry.magnitude,
                size_arcmin=entry.size_arcmin,
                transit_time_local=transit_local,
                max_altitude_deg=round(float(max_alt[i]), 1),
                hours_visible=round(float(hours_visible[i]), 2),
                moon_separation_deg=round(moon_sep, 1),
            )
        )

    # Default rank: best (longest) observing window first, peak altitude as tiebreak —
    # the natural "what should I image tonight" order. The UI table remains fully
    # sortable by any column (name/type/mag/transit/max-alt/hours-visible).
    results.sort(key=lambda r: (-r.hours_visible, -r.max_altitude_deg))

    progress(1.0, "Done.")
    return SkyPlanResult(
        objects=results,
        moon_phase_pct=moon_phase_pct,
        moon_phase_label=moon_phase_label,
        moon_rise_local=moon_rise,
        moon_transit_local=moon_transit,
        moon_set_local=moon_set,
        sun_rise_local=sun_rise,
        sun_set_local=sun_set,
        twilight_evening_local=twilight_evening,
        twilight_morning_local=twilight_morning,
        local_sidereal_time=local_sidereal_time,
        n_catalog_total=n_total,
        n_after_filters=len(entries),
        warnings=warnings,
        message=f"{len(results)} object(s) observable tonight.",
    )


@dataclass
class AltitudeCurve:
    """A single object's altitude track across the observing night, for plotting.

    ``hours`` runs from 12 to 36 (noon on the observing date through noon the
    next day), matching SASpro's ``_compute_alt_curve`` plotting convention so
    a night is one contiguous curve instead of wrapping at midnight.
    """

    hours: np.ndarray
    object_alt_deg: np.ndarray
    sun_alt_deg: np.ndarray
    moon_alt_deg: np.ndarray
    twilight_threshold_deg: float


def object_altitude_curve(
    ra_deg: float,
    dec_deg: float,
    params: SkyPlanParams,
) -> AltitudeCurve:
    """Compute one object's altitude curve for the night described by *params*.

    Cheap single-object companion to ``plan_sky`` — ported from SASpro's
    ``_compute_alt_curve``, ``ObjectVisibilityDialog._compute_and_plot`` — for
    drawing an altitude-vs-time plot for one selected object without
    recomputing the whole catalog.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    grid = _build_night_grid(params)
    obj_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    obj_alts = obj_coord.transform_to(grid.frame).alt.deg

    n = len(grid.sun_alts)
    hours = 12.0 + np.linspace(0, 24, n)
    twilight_threshold = _TWILIGHT_THRESHOLDS_DEG.get(
        params.twilight, _TWILIGHT_THRESHOLDS_DEG["astronomical"]
    )
    return AltitudeCurve(
        hours=hours,
        object_alt_deg=obj_alts,
        sun_alt_deg=grid.sun_alts,
        moon_alt_deg=grid.moon_alts,
        twilight_threshold_deg=twilight_threshold,
    )
