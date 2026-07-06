"""Tests for the Alt/Az field-rotation calculator (ported from SASpro)."""

from __future__ import annotations

import math

import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astraios.core.field_rotation import (
    FieldRotationParams,
    compute_field_rotation,
    field_rotation_rate_arcsec_per_sec,
    parallactic_angle_deg,
)

# ── Rate formula: R = 15.04 * cos(lat) * cos(az) / cos(alt) ────────────────


def test_rate_matches_hand_computed_value():
    """lat=0, az=0, alt=45 -> 15.04 / cos(45deg), a known numeric value."""
    rate = field_rotation_rate_arcsec_per_sec(lat_deg=0.0, az_deg=0.0, alt_deg=45.0)
    expected = 15.04 / math.cos(math.radians(45.0))
    assert rate == pytest.approx(expected, rel=1e-9)
    assert rate == pytest.approx(21.269772, rel=1e-5)


def test_rate_zero_due_east_and_west():
    """cos(az)=0 at az=90 (E) and az=270 (W) -> rate is zero regardless of alt/lat."""
    for az in (90.0, 270.0):
        for alt in (10.0, 45.0, 80.0):
            rate = field_rotation_rate_arcsec_per_sec(lat_deg=35.0, az_deg=az, alt_deg=alt)
            assert abs(rate) < 1e-9


def test_rate_diverges_at_zenith():
    """cos(alt)->0 as alt->90 -> the source's zenith singularity guard returns +inf."""
    rate = field_rotation_rate_arcsec_per_sec(lat_deg=40.0, az_deg=180.0, alt_deg=90.0)
    assert rate == math.inf


def test_rate_increases_monotonically_toward_zenith():
    """Holding lat/az fixed (due south, cos(az) != 0), rate magnitude grows as alt -> 90."""
    rates = [
        abs(field_rotation_rate_arcsec_per_sec(lat_deg=40.0, az_deg=180.0, alt_deg=alt))
        for alt in (10.0, 45.0, 60.0, 80.0, 89.0, 89.9)
    ]
    assert rates == sorted(rates)
    assert rates[-1] > rates[0] * 10


def test_rate_near_zero_for_observer_near_geographic_pole():
    """cos(lat)->0 as |lat|->90: at the pole an alt-az mount behaves like an
    equatorial one (its azimuth axis is parallel to the sky's rotation axis),
    so field rotation vanishes there regardless of where the target points."""
    rate_pole = field_rotation_rate_arcsec_per_sec(lat_deg=89.9, az_deg=180.0, alt_deg=45.0)
    rate_equator = field_rotation_rate_arcsec_per_sec(lat_deg=0.0, az_deg=180.0, alt_deg=45.0)
    assert abs(rate_pole) < abs(rate_equator) * 0.01


# ── Parallactic angle: q = atan2(sin H, tan(lat)*cos(dec) - sin(dec)*cos(H)) ──


def test_parallactic_angle_zero_at_meridian_transit():
    assert parallactic_angle_deg(lat_deg=40.0, dec_deg=10.0, hour_angle_deg=0.0) == pytest.approx(
        0.0, abs=1e-9
    )


def test_parallactic_angle_antisymmetric_about_transit():
    """Mirroring the hour angle around transit (H -> -H) must flip the sign
    of the parallactic angle but keep its magnitude — this is the
    "correct sign" property required of the formula."""
    for h in (5.0, 15.0, 45.0, 90.0, 150.0):
        q_pos = parallactic_angle_deg(lat_deg=35.0, dec_deg=20.0, hour_angle_deg=h)
        q_neg = parallactic_angle_deg(lat_deg=35.0, dec_deg=20.0, hour_angle_deg=-h)
        assert q_neg == pytest.approx(-q_pos, abs=1e-9)


# ── End-to-end compute_field_rotation ───────────────────────────────────────

_LAT, _LON = 30.4, -90.2  # Baton Rouge area, matches SASpro's own example
_TIME = "2026-01-15T06:00:00"  # UTC


def test_compute_field_rotation_requires_target():
    params = FieldRotationParams(lat_deg=_LAT, lon_deg=_LON, time=_TIME)
    with pytest.raises(ValueError):
        compute_field_rotation(params)


def test_compute_field_rotation_radec_self_consistent():
    params = FieldRotationParams(
        lat_deg=_LAT, lon_deg=_LON, time=_TIME,
        ra_deg=83.822, dec_deg=-5.391,  # M42 (Orion Nebula)
        exposure_s=60.0,
    )
    result = compute_field_rotation(params)

    # Rate is exactly the formula applied to the alt/az this run derived.
    expected_rate = field_rotation_rate_arcsec_per_sec(_LAT, result.az_deg, result.alt_deg)
    assert result.rate_arcsec_per_sec == pytest.approx(expected_rate)

    # Unit conversions are internally consistent.
    assert result.rate_deg_per_min == pytest.approx(result.rate_arcsec_per_sec * 60.0 / 3600.0)
    assert result.total_rotation_arcsec == pytest.approx(
        result.rate_arcsec_per_sec * result.exposure_s
    )
    assert result.total_rotation_deg == pytest.approx(result.total_rotation_arcsec / 3600.0)


def test_compute_field_rotation_altaz_input_round_trips_to_radec():
    """Feeding alt/az directly should recover a consistent ra/dec (astropy
    round trip) that reproduces the same alt/az when transformed forward again."""
    params = FieldRotationParams(
        lat_deg=_LAT, lon_deg=_LON, time=_TIME, alt_deg=55.0, az_deg=120.0,
    )
    result = compute_field_rotation(params)

    location = EarthLocation(lat=_LAT * u.deg, lon=_LON * u.deg)
    from astropy.coordinates import AltAz, SkyCoord

    time = Time(_TIME)
    icrs = SkyCoord(ra=result.ra_deg * u.deg, dec=result.dec_deg * u.deg, frame="icrs")
    altaz = icrs.transform_to(AltAz(obstime=time, location=location))
    assert altaz.alt.deg == pytest.approx(55.0, abs=1e-6)
    assert altaz.az.deg == pytest.approx(120.0, abs=1e-6)


def test_compute_field_rotation_parallactic_angle_zero_at_lst_equals_ra():
    """Choosing ra_deg == local sidereal time puts the target exactly on the
    meridian (hour angle 0), so the parallactic angle must vanish."""
    time = Time(_TIME)
    location = EarthLocation(lat=_LAT * u.deg, lon=_LON * u.deg)
    lst = time.sidereal_time("apparent", longitude=location.lon)

    params = FieldRotationParams(
        lat_deg=_LAT, lon_deg=_LON, time=_TIME,
        ra_deg=lst.deg, dec_deg=_LAT - 20.0,
    )
    result = compute_field_rotation(params)
    assert result.hour_angle_deg == pytest.approx(0.0, abs=1e-6)
    assert result.parallactic_angle_deg == pytest.approx(0.0, abs=1e-6)


def test_compute_field_rotation_deterministic():
    params = FieldRotationParams(
        lat_deg=_LAT, lon_deg=_LON, time=_TIME,
        ra_deg=83.822, dec_deg=-5.391, exposure_s=30.0,
    )
    r1 = compute_field_rotation(params)
    r2 = compute_field_rotation(params)
    assert r1 == r2


def test_compute_field_rotation_accepts_astropy_time_object():
    params = FieldRotationParams(
        lat_deg=_LAT, lon_deg=_LON, time=Time(_TIME),
        ra_deg=83.822, dec_deg=-5.391,
    )
    result = compute_field_rotation(params)
    assert math.isfinite(result.rate_arcsec_per_sec) or result.rate_arcsec_per_sec == math.inf
