"""Photoshop .acv curves file exporter.

PROVENANCE NOTE (read before assuming this is a SASpro port): the task that
produced this module pointed at Seti Astro Suite Pro's
``pro/acv_exporter.py`` as the "ground truth" source for a Photoshop .acv
curves exporter. That file does NOT do this. Despite the module name, its
"ACV" stands for "Astro Catalogue Viewer" — it is an unrelated feature that
exports a rendered image (jpg/png/tif) into per-catalog folders (Messier/
NGC/IC/Caldwell) chosen by the object name's prefix. It contains no curve
control points, no binary struct packing, and nothing related to Photoshop's
Curves dialog. A repo-wide search of the SASpro source tree for
``struct.pack``/``.acv`` turned up no Photoshop-curves writer anywhere.

There is therefore nothing to port for this feature. What follows is an
independent implementation of Adobe's own publicly documented "Curves file
format" (.acv), as specified in Adobe's Photoshop File Formats
Specification (the same format written by Photoshop's Curves dialog "Save…"
button and read back by "Load…", and by third-party tools like GIMP's
curves-import). It is not a derivative of any SASpro code; no GPL
attribution header is attached because none of this was ported.

Binary layout (big-endian / network byte order throughout):

    Version     uint16   = 1
    Count       uint16   number of curves in the file
    For each curve:
        PointCount  uint16   2..19 (Photoshop's own hard limit)
        For each point (PointCount of them):
            Output  uint16   0-255  (Y / vertical / output value)
            Input   uint16   0-255  (X / horizontal / input value)

Points within each curve must be written in ascending Input (X) order;
Photoshop does not require this to open the file but well-formed .acv files
always are, so this exporter sorts before writing.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

ACV_VERSION = 1
MIN_POINTS_PER_CURVE = 2
MAX_POINTS_PER_CURVE = 19

# (channel_name, points) where points is a list of (x, y) in [0, 1] — the
# same normalized representation as astraios.core.curves.CurvePoints.points.
CurveSpec = tuple[str, list[tuple[float, float]]]


@dataclass
class ACVExportParams:
    """Reserved for future export options. Currently no knobs are needed —
    the .acv format has no metadata fields beyond version/count/points."""


def _to_acv_value(v: float) -> int:
    """Clamp a normalized [0, 1] coordinate to Photoshop's 8-bit 0-255 range."""
    v = max(0.0, min(1.0, float(v)))
    return int(round(v * 255.0))


def export_acv(
    curves: list[CurveSpec],
    output_path: str | Path,
    params: ACVExportParams | None = None,
) -> Path:
    """Write curve control points to a Photoshop-compatible .acv file.

    Parameters
    ----------
    curves:
        Ordered list of ``(channel_name, points)``. ``points`` are ``(x, y)``
        control points normalized to ``[0, 1]``. Channel names are not part
        of the .acv binary format (Photoshop identifies channels purely by
        curve order) — they exist here only for caller readability. For an
        RGB document, Photoshop's own convention is to order curves as
        Composite, Red, Green, Blue; pass curves in that order if the file
        needs to open cleanly against a real Photoshop RGB document.
    output_path:
        Destination ``.acv`` file path. Parent directories are created if
        needed.
    params:
        Currently unused (reserved).

    Returns
    -------
    Path
        ``output_path`` as a :class:`~pathlib.Path`.

    Raises
    ------
    ValueError
        If ``curves`` is empty, or any curve has fewer than 2 or more than
        19 points (the range the .acv format supports).
    """
    if not curves:
        raise ValueError("export_acv requires at least one curve")

    chunks = [struct.pack(">HH", ACV_VERSION, len(curves))]
    for name, points in curves:
        n = len(points)
        if n < MIN_POINTS_PER_CURVE or n > MAX_POINTS_PER_CURVE:
            raise ValueError(
                f"Curve '{name}' has {n} point(s); .acv requires between "
                f"{MIN_POINTS_PER_CURVE} and {MAX_POINTS_PER_CURVE}."
            )
        ordered = sorted(points, key=lambda p: p[0])
        chunks.append(struct.pack(">H", n))
        for x, y in ordered:
            chunks.append(struct.pack(">HH", _to_acv_value(y), _to_acv_value(x)))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"".join(chunks))
    return out
