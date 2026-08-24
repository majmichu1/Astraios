"""Recognise images that came out of a smart telescope, already stacked.

Seestar and Dwarf owners are the largest group of new astrophotographers,
and they arrive with a file that is already calibrated, debayered, aligned
and stacked. Handed that file, most software still presents them with a
calibration and stacking pipeline they do not need and cannot interpret,
which is a large part of why they bounce.

Detecting the device lets Astraios say something useful instead: this is
already stacked, here is what it came from, and here is the one button that
takes it to a finished picture.

Matching is by substring across several identity keywords rather than an
exact string compare, because vendors and firmware revisions do not agree on
where the device name goes (TELESCOP on one, INSTRUME or ORIGIN on another)
nor on its exact formatting. Optical specs are only filled in where they
have been verified; ``None`` means "not claimed" and callers must not invent
a value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Header keywords that, across vendors, may carry the device identity.
_IDENTITY_KEYS = (
    "TELESCOP",
    "INSTRUME",
    "ORIGIN",
    "CREATOR",
    "PROGRAM",
    "SWCREATE",
    "OBSERVER",
    "MODEL",
    "CAMERA",
)

# Header keywords that indicate the frame is a stack rather than a sub.
_STACK_COUNT_KEYS = ("STACKCNT", "NCOMBINE", "NIMAGES", "SNAPSHOT", "LIVETIME")


@dataclass(frozen=True)
class SmartTelescope:
    """A recognised smart telescope and what is known about it."""

    key: str
    name: str
    vendor: str
    aliases: tuple[str, ...]
    focal_length_mm: float | None = None
    aperture_mm: float | None = None
    pixel_size_um: float | None = None
    already_stacked: bool = True

    @property
    def focal_ratio(self) -> float | None:
        if self.focal_length_mm and self.aperture_mm:
            return round(self.focal_length_mm / self.aperture_mm, 1)
        return None

    def pixel_scale_arcsec(self, binning: int = 1) -> float | None:
        """Image scale in arcsec/pixel, or None if the optics are unknown."""
        if not (self.focal_length_mm and self.pixel_size_um):
            return None
        return round(
            206.265 * self.pixel_size_um * max(binning, 1) / self.focal_length_mm, 2
        )


# Only specs that have been verified against vendor documentation are filled
# in. Where a figure is not confirmed it stays None rather than being guessed:
# a wrong focal length would silently poison plate-solve hints.
KNOWN_TELESCOPES: tuple[SmartTelescope, ...] = (
    SmartTelescope(
        key="seestar_s50",
        name="ZWO Seestar S50",
        vendor="ZWO",
        aliases=("seestar s50", "seestar_s50", "seestars50", "s50"),
        focal_length_mm=250.0,
        aperture_mm=50.0,
        pixel_size_um=2.9,
    ),
    SmartTelescope(
        key="seestar_s30",
        name="ZWO Seestar S30",
        vendor="ZWO",
        aliases=("seestar s30", "seestar_s30", "seestars30", "s30"),
        focal_length_mm=150.0,
        aperture_mm=30.0,
        pixel_size_um=None,
    ),
    SmartTelescope(
        key="seestar",
        name="ZWO Seestar",
        vendor="ZWO",
        aliases=("seestar",),
    ),
    SmartTelescope(
        key="dwarf3",
        name="DwarfLab Dwarf 3",
        vendor="DwarfLab",
        aliases=("dwarf 3", "dwarf3", "dwarf_3", "dwarflab 3"),
    ),
    SmartTelescope(
        key="dwarf2",
        name="DwarfLab Dwarf II",
        vendor="DwarfLab",
        aliases=("dwarf ii", "dwarf2", "dwarf_2", "dwarf ii"),
    ),
    SmartTelescope(
        key="dwarf",
        name="DwarfLab Dwarf",
        vendor="DwarfLab",
        aliases=("dwarflab", "dwarf"),
    ),
    SmartTelescope(
        key="vespera",
        name="Vaonis Vespera",
        vendor="Vaonis",
        aliases=("vespera", "vaonis"),
    ),
    SmartTelescope(
        key="stellina",
        name="Vaonis Stellina",
        vendor="Vaonis",
        aliases=("stellina",),
    ),
    SmartTelescope(
        key="unistellar",
        name="Unistellar eVscope",
        vendor="Unistellar",
        aliases=("unistellar", "evscope", "equinox"),
    ),
)


def _header_text(header: dict | None) -> str:
    """All identity-bearing header values, lowercased into one haystack."""
    if not header:
        return ""
    parts = []
    for key in _IDENTITY_KEYS:
        value = header.get(key)
        if value is None:
            value = header.get(key.lower())
        if isinstance(value, str):
            parts.append(value)
    return " ".join(parts).lower()


def identify(header: dict | None) -> SmartTelescope | None:
    """Return the smart telescope this frame came from, or None.

    More specific entries are listed before their generic fallbacks (S50
    before the bare "seestar"), and the first match wins, so a fully
    identified device is preferred over the family.
    """
    haystack = _header_text(header)
    if not haystack:
        return None
    for scope in KNOWN_TELESCOPES:
        for alias in scope.aliases:
            if alias in haystack:
                return scope
    return None


def stack_count(header: dict | None) -> int | None:
    """Number of sub-frames in the stack, if the header records it."""
    if not header:
        return None
    for key in _STACK_COUNT_KEYS:
        value = header.get(key, header.get(key.lower()))
        if value is None:
            continue
        try:
            count = int(float(value))
        except (TypeError, ValueError):
            continue
        if count > 1:
            return count
    return None


def looks_prestacked(header: dict | None, data: np.ndarray | None = None) -> bool:
    """True when the frame appears to be an integration, not a single sub.

    Three independent signals, any of which is enough: a recognised smart
    telescope (they only ever hand you stacks), an explicit stack count in
    the header, or a total exposure far longer than any single sub-frame.
    """
    if identify(header) is not None:
        return True
    if stack_count(header) is not None:
        return True
    if header:
        exposure = header.get("EXPTIME", header.get("EXPOSURE"))
        try:
            if exposure is not None and float(exposure) > 600.0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def describe(header: dict | None, data: np.ndarray | None = None) -> str | None:
    """A one-line, plain-language summary for the log, or None if unknown.

    Deliberately ends by naming the next action: the whole point of
    recognising the device is to remove the "what now?" moment.
    """
    scope = identify(header)
    count = stack_count(header)
    if scope is None and count is None:
        return None

    if scope is not None:
        bits = [f"{scope.name} image detected"]
        scale = scope.pixel_scale_arcsec()
        details = []
        if scope.focal_length_mm:
            ratio = scope.focal_ratio
            details.append(
                f"{scope.focal_length_mm:.0f}mm"
                + (f" f/{ratio:g}" if ratio else "")
            )
        if scale:
            details.append(f"{scale} arcsec/pixel")
        if details:
            bits.append(f"({', '.join(details)})")
    else:
        bits = ["Stacked image detected"]

    if count:
        bits.append(f"from {count} sub-frames")

    message = " ".join(bits) + "."
    if scope is not None and scope.already_stacked:
        message += (
            " It is already calibrated and stacked, so skip straight to"
            " processing: Tools > Guided Processing (Ctrl+G)."
        )
    else:
        message += " Tools > Guided Processing (Ctrl+G) will take it from here."
    return message


__all__ = [
    "SmartTelescope",
    "KNOWN_TELESCOPES",
    "identify",
    "stack_count",
    "looks_prestacked",
    "describe",
]
