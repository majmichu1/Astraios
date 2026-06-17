"""Tests for the SIMBAD fallback lookup (HTTP mocked — no network)."""

import json
from contextlib import contextmanager

from astraios.core import simbad_lookup
from astraios.core.simbad_lookup import _recipe_for_otype, lookup_simbad


@contextmanager
def _fake_simbad(rows, monkeypatch, raises=None):
    """Patch urlopen to return a canned SIMBAD TAP JSON payload (or raise)."""
    def fake_urlopen(req, timeout=15.0):
        if raises is not None:
            raise raises
        body = json.dumps({"data": rows}).encode()

        class _Resp:
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
            def read(self_): return body
        return _Resp()

    monkeypatch.setattr(simbad_lookup.urllib.request, "urlopen", fake_urlopen)
    yield


def test_resolves_galaxy(monkeypatch):
    # main_id, ra, dec, otype, major, minor, pa
    rows = [["M  51", 202.4696, 47.1952, "Sy2", 11.2, 6.9, 10.0]]
    with _fake_simbad(rows, monkeypatch):
        t = lookup_simbad("M51")
    assert t is not None
    assert abs(t.ra_deg - 202.4696) < 1e-3
    assert t.object_type == "galaxy_spiral"
    assert t.angular_size_arcmin == (11.2, 6.9)
    assert t.processing_hints.get("position_angle_deg") == 10.0
    assert t.processing_hints.get("bg_sensitive") is True


def test_planetary_nebula_recipe():
    ot, br, dr, hints = _recipe_for_otype("PN")
    assert ot == "planetary_nebula"
    assert hints.get("deconv_aggressive") is True


def test_supernova_remnant_is_bg_sensitive():
    ot, br, dr, hints = _recipe_for_otype("SNR")
    assert ot == "supernova_remnant"
    assert hints.get("bg_sensitive") is True
    assert hints.get("stretch") == "aggressive"


def test_no_match_returns_none(monkeypatch):
    with _fake_simbad([], monkeypatch):
        assert lookup_simbad("NotARealObject") is None


def test_network_error_returns_none(monkeypatch):
    with _fake_simbad(None, monkeypatch, raises=OSError("no network")):
        assert lookup_simbad("M51") is None


def test_empty_name_returns_none():
    assert lookup_simbad("") is None
    assert lookup_simbad("   ") is None


def test_missing_size_gets_default(monkeypatch):
    rows = [["NGC 1234", 50.0, 20.0, "G", None, None, None]]
    with _fake_simbad(rows, monkeypatch):
        t = lookup_simbad("NGC 1234")
    assert t is not None
    assert t.angular_size_arcmin[0] > 0  # defaulted, not zero
