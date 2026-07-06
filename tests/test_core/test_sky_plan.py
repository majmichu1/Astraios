"""Tests for the What's In My Sky observation planner (ported from SASpro)."""

from __future__ import annotations

import pytest

from astraios.core.dso_catalog import all_entries
from astraios.core.sky_plan import SkyPlanParams, plan_sky

pytestmark = pytest.mark.filterwarnings(
    "ignore:transforming other coordinates.*:UserWarning"
)


def _names(result):
    return {o.name for o in result.objects}


def test_no_network_guard_engaged():
    """plan_sky must disable astropy's IERS auto-download (SASpro's own guard)."""
    from astropy.utils import iers

    plan_sky(SkyPlanParams(latitude_deg=0.0, longitude_deg=0.0, date="2026-01-01"))
    assert iers.conf.auto_download is False


def test_zenith_transit_altitude_matches_culmination_formula():
    """An object transits at altitude 90 - |lat - dec| (standard culmination formula).

    M78 sits almost exactly on the celestial equator (dec ~ 0.079 deg). Observing
    from a site at the same latitude, it should culminate almost at the zenith.
    """
    m78 = next(e for e in all_entries() if e.name == "M78")
    lat = m78.dec_deg  # observer directly under the object's declination circle
    params = SkyPlanParams(
        latitude_deg=lat, longitude_deg=0.0, date="2026-03-20",
        timezone="UTC", min_altitude_deg=0.0,
    )
    result = plan_sky(params)
    obj = next(o for o in result.objects if o.name == "M78")
    # 5-minute grid resolution means the true peak can be missed by a fraction of a
    # degree; a couple of degrees of tolerance comfortably covers that.
    assert obj.max_altitude_deg == pytest.approx(90.0, abs=2.0)


def test_culmination_altitude_generic_lat_dec():
    """Same culmination formula at a non-zenith latitude (M31, dec ~41.269 deg)."""
    m31 = next(e for e in all_entries() if e.name == "M31")
    lat = 40.0
    expected = 90.0 - abs(lat - m31.dec_deg)
    params = SkyPlanParams(
        latitude_deg=lat, longitude_deg=-105.0, date="2026-10-15",
        timezone="UTC", min_altitude_deg=0.0,
    )
    result = plan_sky(params)
    obj = next(o for o in result.objects if o.name == "M31")
    assert obj.max_altitude_deg == pytest.approx(expected, abs=2.0)


def test_far_south_object_not_visible_from_northern_site():
    """NGC 3372 (Eta Carinae Nebula, dec ~ -59.9 deg) never rises from lat 60 N."""
    params = SkyPlanParams(
        latitude_deg=60.0, longitude_deg=0.0, date="2026-06-01",
        timezone="UTC", min_altitude_deg=0.0,
    )
    result = plan_sky(params)
    assert "NGC 3372" not in _names(result)


def test_moon_phase_spans_new_and_full_within_a_lunar_month():
    """Scanning a 30-day window must hit both a near-new and a near-full phase.

    A synodic month is ~29.53 days, so any 30-consecutive-day window contains one
    of each extreme — this is a physical invariant, not tied to any specific
    calendar date, so it doesn't depend on knowing today's real moon phase.
    """
    from datetime import date, timedelta

    start = date(2026, 1, 1)
    phases = []
    for i in range(30):
        d = (start + timedelta(days=i)).isoformat()
        params = SkyPlanParams(latitude_deg=0.0, longitude_deg=0.0, date=d, timezone="UTC")
        result = plan_sky(params)
        phases.append((result.moon_phase_pct, result.moon_phase_label))

    pcts = [p for p, _ in phases]
    assert min(pcts) <= 5
    assert max(pcts) >= 95
    labels_near_new = {lbl for p, lbl in phases if p <= 5}
    labels_near_full = {lbl for p, lbl in phases if p >= 95}
    assert "New Moon" in labels_near_new
    assert "Full Moon" in labels_near_full


def test_magnitude_filter_excludes_faint_objects():
    params_unfiltered = SkyPlanParams(
        latitude_deg=35.0, longitude_deg=-110.0, date="2026-07-06",
        timezone="UTC", min_altitude_deg=0.0, max_magnitude=None,
    )
    unfiltered = plan_sky(params_unfiltered)
    faint_bright_names = {
        o.name for o in unfiltered.objects
        if o.magnitude is not None and o.magnitude > 9.0
    }
    assert faint_bright_names, "test setup: expected at least one faint object in range"

    params_filtered = SkyPlanParams(
        latitude_deg=35.0, longitude_deg=-110.0, date="2026-07-06",
        timezone="UTC", min_altitude_deg=0.0, max_magnitude=9.0,
    )
    filtered = plan_sky(params_filtered)
    assert not (faint_bright_names & _names(filtered))
    for obj in filtered.objects:
        assert obj.magnitude is None or obj.magnitude <= 9.0


def test_object_type_filter():
    params = SkyPlanParams(
        latitude_deg=35.0, longitude_deg=-110.0, date="2026-07-06",
        timezone="UTC", min_altitude_deg=0.0, object_types=("GC",),
    )
    result = plan_sky(params)
    assert result.objects  # sanity: globular clusters exist above the horizon
    assert all(o.type_code == "GC" for o in result.objects)


def test_altitude_filter_reduces_candidate_count():
    base = dict(
        latitude_deg=35.0, longitude_deg=-110.0, date="2026-07-06", timezone="UTC",
    )
    lenient = plan_sky(SkyPlanParams(min_altitude_deg=5.0, **base))
    strict = plan_sky(SkyPlanParams(min_altitude_deg=80.0, **base))
    assert len(strict.objects) < len(lenient.objects)
    for obj in strict.objects:
        assert obj.max_altitude_deg >= 80.0


def test_deterministic_repeat_call():
    params = SkyPlanParams(
        latitude_deg=45.0, longitude_deg=-93.0, date="2026-09-15",
        timezone="America/Chicago", min_altitude_deg=20.0, max_magnitude=8.0,
    )
    r1 = plan_sky(params)
    r2 = plan_sky(params)
    assert [o.name for o in r1.objects] == [o.name for o in r2.objects]
    assert [o.transit_time_local for o in r1.objects] == [o.transit_time_local for o in r2.objects]
    assert [o.max_altitude_deg for o in r1.objects] == [o.max_altitude_deg for o in r2.objects]


def test_unknown_timezone_falls_back_to_utc_with_warning():
    params = SkyPlanParams(
        latitude_deg=10.0, longitude_deg=10.0, date="2026-05-01",
        timezone="Not/A_Real_Zone",
    )
    result = plan_sky(params)
    assert result.warnings
    assert any("Not/A_Real_Zone" in w for w in result.warnings)


def test_hours_visible_and_transit_time_are_populated():
    params = SkyPlanParams(
        latitude_deg=35.0, longitude_deg=-110.0, date="2026-07-06",
        timezone="America/Phoenix", min_altitude_deg=30.0, max_magnitude=9.0,
    )
    result = plan_sky(params)
    assert result.objects
    for obj in result.objects:
        assert obj.hours_visible > 0.0
        assert obj.transit_time_local != "n/a"
        assert 0 <= obj.max_altitude_deg <= 90.0
        assert 0 <= obj.moon_separation_deg <= 180.0
    # Default ranking: best (longest) window first.
    hours = [o.hours_visible for o in result.objects]
    assert hours == sorted(hours, reverse=True)


def test_progress_callback_reaches_completion():
    calls = []
    params = SkyPlanParams(latitude_deg=0.0, longitude_deg=0.0, date="2026-01-01")
    plan_sky(params, progress=lambda f, m: calls.append((f, m)))
    assert calls
    assert calls[-1][0] == 1.0
