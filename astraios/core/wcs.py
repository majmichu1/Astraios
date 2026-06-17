"""WCS dict utilities — normalise between two incompatible key formats.

There are two WCS dict formats in the codebase. They are NOT interchangeable:

    Source                              Keys
    ------                              ----
    plate_solve._parse_wcs_header       ra_center, dec_center
    star_catalog._parse_wcs_fits        ra, dec
    color_calibration._make_pixel_to_sky  expects ra, dec

Always call `normalise_wcs_dict()` at the boundary between a plate-solve
producer (``ra_center``) and a PCC/overlay consumer (``ra``, ``dec``).
"""

from __future__ import annotations

from typing import Any


def normalise_wcs_dict(wcs: dict[str, Any]) -> dict[str, Any]:
    """Return a WCS dict that has both ``ra, dec`` and ``ra_center, dec_center`` keys.

    If the input uses ``ra_center`` / ``dec_center``, aliases are added as
    ``ra`` / ``dec``.  If it uses ``ra`` / ``dec``, the reverse aliases are
    added.  Mixed or missing keys are logged but not raised.
    """
    norm = dict(wcs)

    if "ra" not in norm and "ra_center" in norm:
        norm["ra"] = norm["ra_center"]
        norm["dec"] = norm["dec_center"]
    elif "ra_center" not in norm and "ra" in norm:
        norm["ra_center"] = norm["ra"]
        norm["dec_center"] = norm["dec"]

    return norm
