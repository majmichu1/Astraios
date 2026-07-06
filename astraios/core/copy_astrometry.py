"""Copy Astrometry — transfer a WCS/SIP solution between FITS headers.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SASpro's ``pro/copyastro.py`` (``CopyAstrometryDialog``) lets a user pick a
plate-solved "source" view and copy its astrometric solution onto a "target"
view that lacks one (or has a stale one) -- e.g. after stacking discards a
per-frame solution, or to seed a mosaic tile with a neighbor's rough
placement. The actual key set and merge logic lives in SASpro's
``gui/mixins/header_mixin.py`` (``HeaderMixin._WCS_KEY_SET`` /
``_extract_wcs_dict``) and ``gui/main_window.py``
(``_apply_wcs_dict_to_doc``); this module ports that key set and the
"replace target's WCS block, keep everything else" merge behavior as plain
header-dict functions, without the Qt dialog / document-manager plumbing.

GPU/CPU decision: pure dict/string bookkeeping over a few dozen header
cards -- no GPU involvement is applicable.
"""

from __future__ import annotations

import re

Header = dict  # a FITS header behaves like a dict for our purposes (astropy
# ``fits.Header`` supports ``.items()``/``.keys()``/``__getitem__`` too).

# Exact keys SASpro always treats as WCS (HeaderMixin._WCS_KEY_SET).
# NAXIS1/NAXIS2 are included because SASpro keeps them "as useful context for
# UIs/solvers" even though copying them can overwrite the target's own image
# dimensions -- ported faithfully; see module docstring / caller beware.
_WCS_KEY_SET: set[str] = {
    "WCSAXES", "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2",
    "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    "CDELT1", "CDELT2",
    "LONPOLE", "LATPOLE",
    "RADESYS", "RADECSYS", "EQUINOX", "EPOCH",
    "NAXIS1", "NAXIS2",
}

# SASpro also matches these prefix/regex families when scanning a document's
# mirrored image_meta["WCS"] dict (HeaderMixin._extract_wcs_dict): CROTA*,
# PV<i>_<j> (SCAMP/TPV distortion terms), and SIP polynomial coefficients
# A_/B_/AP_/BP_ (+ their *_ORDER keys).
_CROTA_RE = re.compile(r"^CROTA\d+$")
_PV_RE = re.compile(r"^PV\d+_\d+$")
_SIP_RE = re.compile(r"^(A|B|AP|BP)_\d+_\d+$")
_SIP_ORDER_KEYS = {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}


def _is_wcs_key(key: str) -> bool:
    """Return True if ``key`` is one SASpro copies as part of a WCS solution."""
    k = str(key).upper()
    if k in _WCS_KEY_SET:
        return True
    if k in _SIP_ORDER_KEYS:
        return True
    if _CROTA_RE.match(k) or _PV_RE.match(k) or _SIP_RE.match(k):
        return True
    return False


def _header_items(header) -> list[tuple[str, object]]:
    """Return ``(key, value)`` pairs for a dict or an astropy ``fits.Header``."""
    if header is None:
        return []
    if hasattr(header, "items"):
        return list(header.items())
    return list(dict(header).items())


def extract_wcs_dict(header) -> dict:
    """Extract the WCS/SIP key subset of ``header`` (SASpro's ``_extract_wcs_dict``).

    Args:
        header: A FITS-header-like mapping (``dict`` or ``astropy.io.fits.Header``).

    Returns:
        A flat ``dict`` of only the WCS/SIP cards present in ``header``, with
        keys upper-cased (FITS keyword convention).
    """
    return {str(k).upper(): v for k, v in _header_items(header) if _is_wcs_key(k)}


def wcs_keywords_present(header) -> bool:
    """Return True if ``header`` carries a usable astrometric solution.

    Mirrors SASpro's minimal criterion for "this document has WCS": both
    ``CRVAL1`` and ``CRVAL2`` (the solved pointing) must be present.
    """
    keys = {str(k).upper() for k, _ in _header_items(header)}
    return "CRVAL1" in keys and "CRVAL2" in keys


def copy_astrometry(source_header: dict, target_header: dict) -> dict:
    """Copy the WCS/SIP astrometric solution from ``source_header`` onto ``target_header``.

    Any WCS keys already present on the target are dropped and replaced by
    the source's solution (a "copy", not a merge, of the WCS block); every
    non-WCS key on the target (acquisition metadata, comments, etc.) is
    preserved untouched. This matches SASpro's ``_apply_wcs_dict_to_doc``,
    which rebuilds the document's WCS header block wholesale from the
    donor's solution while keeping the target's own acquisition header
    (``fits_header``) separate.

    Args:
        source_header: Header of the plate-solved donor image.
        target_header: Header to receive the solution.

    Returns:
        A new header dict: ``target_header``'s non-WCS keys plus
        ``source_header``'s WCS/SIP keys, with ``HasAstrometricSolution``
        set to ``True`` (SASpro sets this flag after a successful copy).

    Raises:
        ValueError: If ``source_header`` has no WCS solution to copy.
    """
    if not wcs_keywords_present(source_header):
        raise ValueError("source_header has no WCS solution (missing CRVAL1/CRVAL2).")

    wcs = extract_wcs_dict(source_header)

    result = {k: v for k, v in _header_items(target_header) if not _is_wcs_key(k)}
    result.update(wcs)
    result["HasAstrometricSolution"] = True
    return result
