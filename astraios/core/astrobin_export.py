"""AstroBin acquisition-details CSV exporter.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Scans a set of light-frame FITS headers, groups them the way AstroBin's
"acquisition details" CSV importer expects (observing night x filter x
exposure length), and writes the CSV. Column set, field names, grouping key,
and per-row precedence rules (header value wins, global fallback used only
when the header is missing/zero) are copied verbatim from SASpro's
``pro/astrobin_exporter.py`` (``AstrobinExportTab._recompute`` /
``_rows_to_csv_str``); only the Qt UI/QSettings/XISF plumbing was stripped out
in favor of a pure function operating on FITS headers.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from astropy.io import fits

# CSV column order AstroBin's acquisition-details importer expects.
BASE_FIELDNAMES = [
    "date", "filter", "number", "duration", "gain", "iso", "binning",
    "sensorCooling", "fNumber", "darks", "flats", "flatDarks", "bias",
    "bortle", "meanSqm", "meanFwhm", "temperature",
]


@dataclass
class AstroBinExportParams:
    """Global fallback values, used only when a frame's FITS header is
    missing the corresponding keyword (or the header value is "0"/"0.0").

    ``filter_map`` maps a local filter name (e.g. "Ha") to AstroBin's numeric
    filter-equipment-database ID (e.g. "4408"); frames whose filter name has
    no mapping are grouped/emitted under the raw filter name instead (and
    will show up red in SASpro's preview table since AstroBin requires a
    numeric ID — same behavior here, just without the color).
    """

    fnumber: str = "0"
    darks: str = ""
    flats: str = ""
    flat_darks: str = ""
    bias: str = ""
    bortle: str = ""
    mean_sqm: str = ""
    mean_fwhm: str = ""
    noon_to_noon: bool = True
    filter_map: dict[str, str] = field(default_factory=dict)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _derive_binning(header: dict) -> str:
    for key in ("XBINNING", "XBIN", "CCDXBIN"):
        if key in header:
            try:
                return str(int(float(header[key])))
            except (TypeError, ValueError):
                return str(header[key])
    return "0"


def _to_date_only(date_obs: str) -> str:
    if not date_obs or date_obs == "0":
        return "0"
    return date_obs.split("T")[0].strip()


def _parse_date_obs(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s or s == "0":
        return None
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _night_date_str(date_obs: str, noon_to_noon: bool) -> str:
    """Group a timestamp into an observing "night", local time.

    With ``noon_to_noon`` (the default), 12 hours are subtracted before
    taking the date so a night that crosses local midnight is not split
    into two separate nights.
    """
    dt = _parse_date_obs(date_obs)
    if not dt:
        return date_obs.split("T")[0] if date_obs else "0"
    local_tz = datetime.now().astimezone().tzinfo
    ldt = dt.astimezone(local_tz)
    if noon_to_noon:
        ldt = ldt - timedelta(hours=12)
    return ldt.date().isoformat()


def _has_gain(v: Any) -> bool:
    s = str(v).strip()
    if not s:
        return False
    try:
        return float(s) > 0.0
    except ValueError:
        return False


def _fallback(header_val: Any, global_val: Any) -> str:
    hv = str(header_val).strip() if header_val not in (None, "") else ""
    gv = str(global_val).strip() if global_val not in (None, "") else ""
    if hv in ("", "0", "0.0") and gv != "":
        return gv
    return hv or "0"


def read_frame_headers(frame_paths: list[str | Path]) -> list[dict]:
    """Read the FITS header fields needed for AstroBin export from each path.

    Files that fail to open (corrupt, missing, not FITS) are skipped rather
    than raising, matching SASpro's "load what we can" behavior; the caller
    can compare the length of the result against ``len(frame_paths)`` to see
    how many were skipped.
    """
    records: list[dict] = []
    for fp in frame_paths:
        path = Path(fp)
        try:
            header = fits.getheader(str(path), ext=0)
        except Exception:
            continue

        exposure = header.get("EXPOSURE", header.get("EXPTIME", 0.0))
        date_obs = str(header.get("DATE-OBS", ""))
        records.append({
            "PATH": path,
            "NAME": path.name,
            "OBJECT": str(header.get("OBJECT", "Unknown")),
            "FILTER": str(header.get("FILTER", "Unknown")),
            "EXPOSURE": _safe_float(exposure),
            "GAIN": str(header.get("GAIN", "0")),
            "ISO": str(header.get("ISO", "0")),
            "BINNING": _derive_binning(header),
            "CCD_TEMP": _safe_float(header.get("CCD-TEMP", 0.0)),
            "FOCTEMP": _safe_float(header.get("FOCTEMP", 0.0)),
            "DARK": str(header.get("DARK", "0")),
            "FLAT": str(header.get("FLAT", "0")),
            "FLATDARK": str(header.get("FLATDARK", "0")),
            "BIAS": str(header.get("BIAS", "0")),
            "BORTLE": str(header.get("BORTLE", "0")),
            "MEAN_SQM": str(header.get("MEAN_SQM", "0")),
            "MEAN_FWHM": str(header.get("MEAN_FWHM", "0")),
            "DATE": _to_date_only(date_obs),
            "DATEOBS": date_obs,
        })
    return records


def aggregate_astrobin_rows(
    records: list[dict],
    params: AstroBinExportParams | None = None,
) -> list[dict]:
    """Group frame records by (observing night, filter, exposure) and
    produce one AstroBin acquisition-details row per group.

    ``records`` is the list produced by :func:`read_frame_headers` (or an
    equivalent list of dicts with the same keys).
    """
    if params is None:
        params = AstroBinExportParams()

    agg: dict[tuple, dict] = defaultdict(lambda: {
        "date": "0", "filter": "0", "number": 0, "duration": 0, "gain": "0",
        "iso": "0", "binning": "0", "sensorCooling": 0, "fNumber": "0",
        "darks": "0", "flats": "0", "flatDarks": "0", "bias": "0",
        "bortle": "0", "meanSqm": "0", "meanFwhm": "0",
        "temperature_sum": 0.0, "temp_count": 0,
    })

    for rec in records:
        date = _night_date_str(rec.get("DATEOBS", ""), params.noon_to_noon)
        filt_name = rec["FILTER"] or "0"
        filt_id = params.filter_map.get(filt_name, filt_name)
        exposure = rec["EXPOSURE"] or 0.0
        key = (date, str(filt_id), float(exposure))

        item = agg[key]
        item["date"] = date
        item["filter"] = str(filt_id)
        item["duration"] = exposure
        item["gain"] = rec["GAIN"]
        item["iso"] = rec["ISO"]
        item["binning"] = rec["BINNING"]
        item["sensorCooling"] = int(round(rec["CCD_TEMP"])) if rec["CCD_TEMP"] else 0
        item["fNumber"] = str(params.fnumber).strip() or "0"

        item["darks"] = _fallback(rec["DARK"], params.darks)
        item["flats"] = _fallback(rec["FLAT"], params.flats)
        item["flatDarks"] = _fallback(rec["FLATDARK"], params.flat_darks)
        item["bias"] = _fallback(rec["BIAS"], params.bias)
        item["bortle"] = _fallback(rec["BORTLE"], params.bortle)
        item["meanSqm"] = _fallback(rec["MEAN_SQM"], params.mean_sqm)
        item["meanFwhm"] = _fallback(rec["MEAN_FWHM"], params.mean_fwhm)

        if rec["FOCTEMP"]:
            item["temperature_sum"] += float(rec["FOCTEMP"])
            item["temp_count"] += 1
        item["number"] += 1

    out = []
    for (_date, _fid, _exp), v in agg.items():
        temp = int(round(v["temperature_sum"] / v["temp_count"])) if v["temp_count"] > 0 else 0
        row = {
            "date": v["date"], "filter": v["filter"], "number": v["number"],
            "duration": v["duration"], "gain": v["gain"], "iso": v["iso"],
            "binning": v["binning"], "sensorCooling": v["sensorCooling"],
            "fNumber": v["fNumber"], "darks": v["darks"], "flats": v["flats"],
            "flatDarks": v["flatDarks"], "bias": v["bias"], "bortle": v["bortle"],
            "meanSqm": v["meanSqm"], "meanFwhm": v["meanFwhm"], "temperature": temp,
        }
        if _has_gain(row["gain"]):
            row["iso"] = ""  # gain present => blank ISO (AstroBin wants one or the other)
        out.append(row)

    out.sort(key=lambda r: (r["date"], r["filter"], float(r["duration"])))
    return out


def rows_to_csv_str(rows: list[dict]) -> str:
    """Serialize aggregated rows to CSV text (AstroBin acquisition format).

    The ``iso`` column is dropped entirely (not just blanked) if any row in
    the export has a nonzero gain — matching SASpro, which treats gain and
    ISO as mutually exclusive for the whole export rather than per row.
    """
    drop_iso = any(_has_gain(r.get("gain", "")) for r in (rows or []))
    fieldnames = [f for f in BASE_FIELDNAMES if f != "iso"] if drop_iso else BASE_FIELDNAMES
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows or [])
    return buf.getvalue()


def export_astrobin_csv(
    frame_paths: list[str | Path],
    output_path: str | Path,
    params: AstroBinExportParams | None = None,
) -> Path:
    """Scan ``frame_paths``' FITS headers and write an AstroBin acquisition
    CSV to ``output_path``. Returns the path written.
    """
    records = read_frame_headers(frame_paths)
    rows = aggregate_astrobin_rows(records, params)
    csv_text = rows_to_csv_str(rows)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(csv_text, encoding="utf-8")
    return out
