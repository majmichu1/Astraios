"""Tests for the AstroBin acquisition-details CSV exporter
(ported from SASpro pro/astrobin_exporter.py)."""

from __future__ import annotations

import csv
import time

import numpy as np
import pytest
from astropy.io import fits

from astraios.core.astrobin_export import (
    BASE_FIELDNAMES,
    AstroBinExportParams,
    export_astrobin_csv,
    read_frame_headers,
)


@pytest.fixture(autouse=True)
def _force_utc_local_time(monkeypatch):
    """Night-date grouping converts to the host's local timezone (matching
    SASpro's behavior). Pin it to UTC so the "which night" assertions below
    are deterministic regardless of where the test suite runs."""
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    yield
    time.tzset()


def _write_fits(path, header: dict) -> None:
    data = np.zeros((8, 8), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data)
    for k, v in header.items():
        hdu.header[k] = v
    hdu.writeto(str(path), overwrite=True)


@pytest.fixture
def light_frames(tmp_path):
    """Two nights x Ha filter (3 subs @ 300s night 1, 2 subs @ 300s night 2)
    plus one OIII sub, all same night-1 date. Exercises grouping by
    (night, filter, exposure)."""
    paths = []

    # Night 1, Ha, 300s x3
    for i in range(3):
        p = tmp_path / f"ha_n1_{i}.fits"
        _write_fits(p, {
            "OBJECT": "M31", "FILTER": "Ha", "EXPTIME": 300.0, "GAIN": 100,
            "CCD-TEMP": -10.0, "FOCTEMP": 5.0, "XBINNING": 1,
            "DATE-OBS": "2026-01-15T22:00:00",
        })
        paths.append(p)

    # Night 1, OIII, 300s x1
    p = tmp_path / "oiii_n1_0.fits"
    _write_fits(p, {
        "OBJECT": "M31", "FILTER": "OIII", "EXPTIME": 300.0, "GAIN": 100,
        "CCD-TEMP": -10.0, "FOCTEMP": 6.0, "XBINNING": 1,
        "DATE-OBS": "2026-01-15T23:30:00",
    })
    paths.append(p)

    # Night 2 (a distinct, unambiguous separate observing night), Ha, 300s x2
    for i in range(2):
        p = tmp_path / f"ha_n2_{i}.fits"
        _write_fits(p, {
            "OBJECT": "M31", "FILTER": "Ha", "EXPTIME": 300.0, "GAIN": 100,
            "CCD-TEMP": -10.0, "FOCTEMP": 4.0, "XBINNING": 1,
            "DATE-OBS": "2026-01-20T22:00:00",
        })
        paths.append(p)

    return paths


class TestReadFrameHeaders:
    def test_reads_expected_fields(self, light_frames):
        records = read_frame_headers(light_frames)
        assert len(records) == len(light_frames)
        rec = next(r for r in records if r["NAME"] == "ha_n1_0.fits")
        assert rec["FILTER"] == "Ha"
        assert rec["EXPOSURE"] == 300.0
        assert rec["GAIN"] == "100"
        assert rec["BINNING"] == "1"
        assert rec["CCD_TEMP"] == -10.0

    def test_skips_unreadable_files(self, tmp_path, light_frames):
        bogus = tmp_path / "not_a_fits.fits"
        bogus.write_text("not fits data")
        records = read_frame_headers([*light_frames, bogus])
        assert len(records) == len(light_frames)  # bogus file silently skipped

    def test_missing_headers_default_gracefully(self, tmp_path):
        p = tmp_path / "bare.fits"
        _write_fits(p, {})  # no FILTER/EXPTIME/GAIN/etc at all
        records = read_frame_headers([p])
        assert len(records) == 1
        rec = records[0]
        assert rec["OBJECT"] == "Unknown"
        assert rec["FILTER"] == "Unknown"
        assert rec["EXPOSURE"] == 0.0
        assert rec["GAIN"] == "0"
        assert rec["BINNING"] == "0"
        assert rec["CCD_TEMP"] == 0.0
        assert rec["DATE"] == "0"


class TestExportCsv:
    def test_exact_column_set(self, light_frames, tmp_path):
        out = export_astrobin_csv(light_frames, tmp_path / "out.csv")
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        # gain present on all rows -> iso column dropped entirely
        expected = [c for c in BASE_FIELDNAMES if c != "iso"]
        assert header == expected

    def test_groups_and_counts_correctly(self, light_frames, tmp_path):
        out = export_astrobin_csv(light_frames, tmp_path / "out.csv")
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # 3 groups: (night1, Ha, 300), (night1, OIII, 300), (night2, Ha, 300)
        assert len(rows) == 3

        by_filter_date = {(r["date"], r["filter"]): r for r in rows}
        ha_n1 = by_filter_date[("2026-01-15", "Ha")]
        assert int(ha_n1["number"]) == 3
        assert float(ha_n1["duration"]) == 300.0
        assert ha_n1["gain"] == "100"
        assert int(ha_n1["sensorCooling"]) == -10
        # FOCTEMP averaged across the 3 Ha-night1 subs (5,5,5 -> 5)
        assert int(ha_n1["temperature"]) == 5

        oiii_n1 = by_filter_date[("2026-01-15", "OIII")]
        assert int(oiii_n1["number"]) == 1
        assert int(oiii_n1["temperature"]) == 6

        ha_n2 = by_filter_date[("2026-01-20", "Ha")]
        assert int(ha_n2["number"]) == 2
        assert int(ha_n2["temperature"]) == 4

    def test_filter_map_applied(self, light_frames, tmp_path):
        params = AstroBinExportParams(filter_map={"Ha": "4408", "OIII": "4413"})
        out = export_astrobin_csv(light_frames, tmp_path / "out.csv", params)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        filters = {r["filter"] for r in rows}
        assert filters == {"4408", "4413"}

    def test_global_fallback_used_when_header_missing(self, tmp_path):
        p = tmp_path / "no_bortle.fits"
        _write_fits(p, {
            "FILTER": "L", "EXPTIME": 60.0, "DATE-OBS": "2026-02-01T20:00:00",
        })
        params = AstroBinExportParams(bortle=4, darks=20, flats=30)
        out = export_astrobin_csv([p], tmp_path / "out.csv", params)
        # gain not present anywhere -> iso column retained
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["bortle"] == "4"
        assert rows[0]["darks"] == "20"
        assert rows[0]["flats"] == "30"

    def test_iso_kept_when_no_gain_present(self, tmp_path):
        p = tmp_path / "iso_only.fits"
        _write_fits(p, {
            "FILTER": "L", "EXPTIME": 60.0, "ISO": 800,
            "DATE-OBS": "2026-02-01T20:00:00",
        })
        out = export_astrobin_csv([p], tmp_path / "out.csv")
        with open(out, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert "iso" in header

    def test_empty_frame_list_produces_header_only_csv(self, tmp_path):
        out = export_astrobin_csv([], tmp_path / "out.csv")
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # header row only
        assert rows[0] == BASE_FIELDNAMES
