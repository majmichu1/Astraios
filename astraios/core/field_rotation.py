"""Alt/Az field-rotation calculator.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SASpro's ``altaz_field_rotation.py`` computes the sky-field rotation rate an
Alt/Az-mounted telescope experiences while tracking a target (the reason
alt-az mounts need a de-rotator, or short subs, for long exposures near the
zenith). The core formula (``field_rotation_rate_arcsec_per_sec`` below) is
ported byte-for-byte from that file::

    R = 15.04 * cos(lat) * cos(az) / cos(alt)      [arcsec/sec]

``15.04`` arcsec/sec is Earth's sidereal angular rate
(360 deg * 3600 arcsec/deg / 86164.09 s), and the ``cos(lat)`` term is why
the rate goes to zero for an observer at the geographic pole (an alt-az
mount there behaves like an equatorial mount — its azimuth axis is parallel
to the sky's rotation axis) while the ``1/cos(alt)`` term is why it diverges
at the zenith (the azimuth axis has to sweep infinitely fast to hold a
target exactly overhead).

This module adds (not present verbatim in the source file, since SASpro
computes this from a WIMS ephemeris curve rather than a single-shot API) a
:func:`compute_field_rotation` entry point that:

* Accepts a target as RA/Dec *or* a direct Alt/Az, an observer location, and
  an observation time, and uses astropy to fill in whichever half is
  missing (RA/Dec <-> Alt/Az are inverse ``SkyCoord`` transforms of one
  another for a fixed observer/time).
* Reports the SASpro rate formula's value in both native (arcsec/sec) and
  requested (deg/min) units, plus the total rotation accumulated over a
  given exposure length.
* Reports the parallactic angle at the target's position — the standard
  spherical-astronomy quantity (Meeus, *Astronomical Algorithms*, ch. 14)
  ``q = atan2(sin(H), tan(lat)*cos(dec) - sin(dec)*cos(H))`` where ``H`` is
  the target's hour angle. This is the angle field rotation *rotates
  about*: it is exactly zero at meridian transit and antisymmetric around
  it, which is what the accompanying test asserts.

This is a pure-CPU astropy calculator (coordinate transforms + trig) — no
GPU work, no benchmark, per the astraios convention that ``device_manager``
is reserved for image-tensor workloads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

#: Earth's sidereal rotation rate, arcsec/sec (360 deg * 3600 / 86164.0905 s).
#: Matches the literal constant used in SASpro's field_rotation_rate_arcsec_per_sec.
SIDEREAL_RATE_ARCSEC_PER_SEC = 15.04

# Below this |cos(alt)|, the rate formula's zenith singularity is considered
# reached and +inf is returned — identical guard to the SASpro source.
_ZENITH_EPS = 1e-9


@dataclass
class FieldRotationParams:
    """Inputs for an Alt/Az field-rotation calculation.

    Provide either (``ra_deg``, ``dec_deg``) or (``alt_deg``, ``az_deg``) —
    whichever half is missing is filled in via an astropy coordinate
    transform for the given observer/time. If both are given, the Alt/Az
    pair is used for the rate formula and the RA/Dec pair for the
    parallactic-angle hour angle (so callers who already have both from a
    plate solve + ephemeris need not worry about them drifting apart).

    Attributes:
        lat_deg: Observer latitude, degrees, north positive.
        lon_deg: Observer longitude, degrees, east positive.
        time: Observation time — an :class:`~astropy.time.Time`, ISO string,
            or :class:`~datetime.datetime` (naive datetimes are treated as UTC,
            matching ``astropy.time.Time``'s own default).
        height_m: Observer elevation above the ellipsoid, meters.
        ra_deg: Target right ascension, degrees (ICRS). Optional if alt/az given.
        dec_deg: Target declination, degrees (ICRS). Optional if alt/az given.
        alt_deg: Target altitude, degrees. Optional if ra/dec given.
        az_deg: Target azimuth, degrees (0=N, 90=E, 180=S, 270=W — the same
            convention as astropy's ``AltAz`` frame and the SASpro dialog).
        exposure_s: Sub-exposure length in seconds, used to report the total
            rotation accumulated over one sub.
    """

    lat_deg: float
    lon_deg: float
    time: Time | str | datetime
    height_m: float = 0.0
    ra_deg: float | None = None
    dec_deg: float | None = None
    alt_deg: float | None = None
    az_deg: float | None = None
    exposure_s: float = 0.0


@dataclass
class FieldRotationResult:
    """Result of :func:`compute_field_rotation`."""

    alt_deg: float
    az_deg: float
    ra_deg: float
    dec_deg: float
    hour_angle_deg: float
    parallactic_angle_deg: float
    rate_arcsec_per_sec: float
    rate_arcsec_per_min: float
    rate_deg_per_min: float
    exposure_s: float
    total_rotation_arcsec: float
    total_rotation_deg: float


def field_rotation_rate_arcsec_per_sec(lat_deg: float, az_deg: float, alt_deg: float) -> float:
    """Field-rotation rate in arcsec/sec for an Alt/Az mount.

    Ported verbatim from SASpro's ``altaz_field_rotation.py``::

        R = 15.04 * cos(lat) * cos(az) / cos(alt)

    Returns the signed rate; returns ``+inf`` at the zenith singularity
    (``|cos(alt)|`` below a tiny epsilon), matching source behavior exactly.
    """
    cos_alt = math.cos(math.radians(alt_deg))
    if abs(cos_alt) < _ZENITH_EPS:
        return math.inf
    return (
        SIDEREAL_RATE_ARCSEC_PER_SEC
        * math.cos(math.radians(lat_deg))
        * math.cos(math.radians(az_deg))
        / cos_alt
    )


def parallactic_angle_deg(lat_deg: float, dec_deg: float, hour_angle_deg: float) -> float:
    """Parallactic angle, degrees (Meeus, *Astronomical Algorithms*, eq. 14.1).

    ``q = atan2(sin(H), tan(lat)*cos(dec) - sin(dec)*cos(H))``

    Zero at meridian transit (``H = 0``) and antisymmetric in ``H`` (an
    hour angle and its mirror image around transit give equal-magnitude,
    opposite-sign angles) — the sign-correctness property this module's
    test suite checks.
    """
    lat = math.radians(lat_deg)
    dec = math.radians(dec_deg)
    ha = math.radians(hour_angle_deg)
    return math.degrees(
        math.atan2(math.sin(ha), math.tan(lat) * math.cos(dec) - math.sin(dec) * math.cos(ha))
    )


def _wrap_deg_signed(value_deg: float) -> float:
    """Wrap an angle in degrees to [-180, 180)."""
    return (value_deg + 180.0) % 360.0 - 180.0


def compute_field_rotation(params: FieldRotationParams) -> FieldRotationResult:
    """Compute Alt/Az field-rotation rate, parallactic angle, and total rotation.

    Raises:
        ValueError: if neither (ra_deg, dec_deg) nor (alt_deg, az_deg) is
            fully supplied.
    """
    have_radec = params.ra_deg is not None and params.dec_deg is not None
    have_altaz = params.alt_deg is not None and params.az_deg is not None
    if not have_radec and not have_altaz:
        raise ValueError(
            "FieldRotationParams needs either (ra_deg, dec_deg) or (alt_deg, az_deg)"
        )

    time = params.time if isinstance(params.time, Time) else Time(params.time)
    location = EarthLocation(
        lat=params.lat_deg * u.deg,
        lon=params.lon_deg * u.deg,
        height=params.height_m * u.m,
    )
    altaz_frame = AltAz(obstime=time, location=location)

    if have_radec:
        ra_deg = float(params.ra_deg)
        dec_deg = float(params.dec_deg)
        if have_altaz:
            alt_deg = float(params.alt_deg)
            az_deg = float(params.az_deg)
        else:
            icrs = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
            altaz = icrs.transform_to(altaz_frame)
            alt_deg = float(altaz.alt.deg)
            az_deg = float(altaz.az.deg)
    else:
        alt_deg = float(params.alt_deg)
        az_deg = float(params.az_deg)
        altaz = SkyCoord(alt=alt_deg * u.deg, az=az_deg * u.deg, frame=altaz_frame)
        icrs = altaz.transform_to("icrs")
        ra_deg = float(icrs.ra.deg)
        dec_deg = float(icrs.dec.deg)

    lst = time.sidereal_time("apparent", longitude=location.lon)
    hour_angle_deg = float(_wrap_deg_signed(float(lst.deg) - ra_deg))

    q_deg = parallactic_angle_deg(params.lat_deg, dec_deg, hour_angle_deg)

    rate_asec_sec = field_rotation_rate_arcsec_per_sec(params.lat_deg, az_deg, alt_deg)
    rate_asec_min = rate_asec_sec * 60.0
    rate_deg_min = rate_asec_min / 3600.0

    exposure_s = float(params.exposure_s)
    total_rotation_arcsec = rate_asec_sec * exposure_s
    total_rotation_deg = total_rotation_arcsec / 3600.0

    return FieldRotationResult(
        alt_deg=alt_deg,
        az_deg=az_deg,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        hour_angle_deg=hour_angle_deg,
        parallactic_angle_deg=q_deg,
        rate_arcsec_per_sec=rate_asec_sec,
        rate_arcsec_per_min=rate_asec_min,
        rate_deg_per_min=rate_deg_min,
        exposure_s=exposure_s,
        total_rotation_arcsec=total_rotation_arcsec,
        total_rotation_deg=total_rotation_deg,
    )
