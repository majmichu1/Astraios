"""SIMBAD fallback for target identification.

When a user-typed object name isn't in the bundled 139-object catalog, query
SIMBAD over HTTP (no heavy ``astroquery`` dependency — just ``urllib``) to
resolve its coordinates, object type, and angular size, then synthesise a
:class:`~astraios.core.catalog.TargetInfo` with a processing recipe derived from
the object type. Best-effort and fully optional: any failure (offline, not
found, parse error) returns ``None`` and the caller falls back to whatever it
did before.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from astraios.core.catalog import TargetInfo

log = logging.getLogger(__name__)

__all__ = ["lookup_simbad"]

# SIMBAD TAP sync endpoint (Strasbourg). ADQL over the basic + ident tables.
_TAP_URL = "https://simbad.u-strasbg.fr/simbad/sim-tap/sync"

# Map SIMBAD otype (or its prefix) → (our object_type, recipe-hint builder key).
# SIMBAD types are hierarchical short codes; we match the most specific first,
# then fall back to coarse prefixes.
_GALAXY_PREFIXES = ("G", "AGN", "Sy", "QSO", "Bla", "LIN", "SBG", "rG", "IG", "GiP", "GiG", "GiC")


def _recipe_for_otype(otype: str) -> tuple[str, str, str, dict]:
    """Return (object_type, brightness_class, dynamic_range, processing_hints).

    A compact translation of SIMBAD object types into the same recipe vocabulary
    the bundled catalog uses, so downstream planning is identical whether the
    target came from the local catalog or SIMBAD.
    """
    o = otype.strip()

    if o in ("PN", "PNe"):  # planetary nebula — small, bright core, fine detail
        return ("planetary_nebula", "bright", "high",
                {"stretch": "moderate", "deconv_aggressive": True, "ha_dominant": True})
    if o in ("SNR", "SNRemnant"):  # supernova remnant — faint filaments
        return ("supernova_remnant", "faint", "high",
                {"stretch": "aggressive", "bg_sensitive": True, "ha_dominant": True})
    if o in ("HII", "HII_G"):  # HII region — emission, Ha-dominant
        return ("hii_region", "moderate", "high",
                {"stretch": "moderate", "ha_dominant": True})
    if o in ("RNe", " refN", "RfN"):  # reflection nebula — blue, broadband
        return ("reflection_nebula", "faint", "moderate",
                {"stretch": "gentle", "reflection_nebulosity": True})
    if o in ("DNe", "DkN", "MoC"):  # dark nebula
        return ("dark_nebula", "faint", "moderate",
                {"stretch": "gentle", "bg_sensitive": True})
    if o in ("EmN", "EmO", "Cld", "GNe", "ISM", "Neb"):  # generic nebulosity
        return ("emission_nebula", "moderate", "high",
                {"stretch": "moderate", "ha_dominant": True, "bg_sensitive": True})
    if o in ("GlC", "GlCl"):  # globular cluster — dense bright core
        return ("globular_cluster", "bright", "high",
                {"stretch": "moderate", "hdr_merge_recommended": True})
    if o in ("OpC", "Cl*", "As*"):  # open cluster / association — stars only
        return ("open_cluster", "bright", "moderate",
                {"stretch": "gentle"})
    if any(o.startswith(p) for p in _GALAXY_PREFIXES) or o == "Galaxy":
        return ("galaxy_spiral", "moderate", "high",
                {"stretch": "moderate", "bg_sensitive": True})

    # Unknown / generic deep-sky object — safe neutral recipe.
    return ("unknown", "moderate", "moderate", {"stretch": "moderate"})


def lookup_simbad(name: str, timeout: float = 15.0) -> TargetInfo | None:
    """Resolve *name* via SIMBAD and return a TargetInfo, or None on any failure.

    Parameters
    ----------
    name : str
        Object designation or common name (e.g. ``"NGC 6888"``, ``"Crescent"``).
    timeout : float
        Network timeout in seconds.
    """
    name = (name or "").strip()
    if not name:
        return None

    # ADQL: resolve the identifier, return position, type and angular size.
    adql = (
        "SELECT TOP 1 b.main_id, b.ra, b.dec, b.otype_txt, "
        "b.galdim_majaxis, b.galdim_minaxis, b.galdim_angle "
        "FROM basic AS b JOIN ident AS i ON b.oid = i.oidref "
        f"WHERE i.id = '{name.replace(chr(39), '')}'"
    )
    params = urllib.parse.urlencode({
        "request": "doQuery", "lang": "ADQL", "format": "json", "query": adql,
    }).encode()

    try:
        req = urllib.request.Request(_TAP_URL, data=params)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        log.info("SIMBAD lookup failed for %r: %s", name, exc)
        return None

    rows = payload.get("data") or []
    if not rows:
        log.info("SIMBAD: no match for %r", name)
        return None

    row = rows[0]
    try:
        main_id = str(row[0]).strip()
        ra = float(row[1])
        dec = float(row[2])
        otype = str(row[3] or "").strip()
        major = float(row[4]) if row[4] is not None else 0.0  # arcmin
        minor = float(row[5]) if row[5] is not None else major
        pa = float(row[6]) if row[6] is not None else 0.0
    except (TypeError, ValueError, IndexError) as exc:
        log.info("SIMBAD: malformed row for %r: %s", name, exc)
        return None

    if major <= 0:
        major = 5.0  # sane default extent when SIMBAD has no size
    if minor <= 0:
        minor = major

    object_type, brightness_class, dynamic_range, hints = _recipe_for_otype(otype)
    if pa:
        hints["position_angle_deg"] = pa

    log.info("SIMBAD resolved %r → %s (%s, %.1f'×%.1f')",
             name, main_id, otype, major, minor)
    return TargetInfo(
        id=main_id or name,
        names=[name] if name.lower() != main_id.lower() else [],
        ra_deg=ra,
        dec_deg=dec,
        angular_size_arcmin=(major, minor),
        object_type=object_type,
        magnitude=None,
        surface_brightness=None,
        brightness_class=brightness_class,
        dynamic_range=dynamic_range,
        emission_lines=[],
        dominant_emission=None,
        constellation="",
        processing_hints=hints,
    )
