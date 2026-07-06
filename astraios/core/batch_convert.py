"""Batch Convert — convert a set of image files from one format to another.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Deviations from the source tool (documented, not silent):

* The source scanned an input *folder* (optionally recursively) using glob
  patterns and additionally accepted common camera RAW formats (CR2, NEF,
  ARW, DNG, ORF, RW2, PEF) via a legacy loader. This port instead takes an
  explicit list of file paths (matching this codebase's other batch dialogs,
  e.g. :mod:`astraios.core.batch`) and only accepts the formats
  :func:`astraios.core.image_io.load_image` already understands: FITS
  (``.fit``/``.fits``/``.fts``), XISF, TIFF, PNG, and JPEG. RAW inputs are not
  supported because ``image_io`` has no RAW decoder — do not add one here;
  route any RAW ingestion need through a dedicated loader instead.
* The source offered per-format bit-depth choices including "16-bit",
  "32-bit unsigned integer" and "32-bit floating point" for TIFF/XISF, and
  preserved the *source* file's bit depth automatically under "Auto".
  ``image_io.save_image`` only ever writes FITS/XISF as 32-bit float, JPEG as
  8-bit, and TIFF/PNG as 8-bit or 16-bit integer — and ``image_io.load_image``
  normalizes every input to float32 without recording the original integer
  bit depth. "Auto" here therefore picks a sensible per-format default
  instead of mirroring the source file's depth.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from astraios.core.image_io import load_image, save_image

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


# Output formats this tool can write, in the same order SASpro's combo box
# presented them. Values are the exact suffix passed to image_io.save_image.
OUTPUT_FORMATS: tuple[str, ...] = (
    ".fits", ".fit", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".xisf",
)

# Input formats accepted (mirrors astraios.core.image_io.load_image exactly;
# see the module docstring for why RAW formats from the source are omitted).
INPUT_SUFFIXES: tuple[str, ...] = (
    ".fit", ".fits", ".fts", ".xisf", ".tif", ".tiff", ".png", ".jpg", ".jpeg",
)

# Bit-depth options actually supported by image_io.save_image, per output
# format. FITS/XISF are hard-coded to 32-bit float; JPEG is hard-coded 8-bit.
ALLOWED_BIT_DEPTHS: dict[str, tuple[int, ...]] = {
    ".fits": (32,),
    ".fit": (32,),
    ".xisf": (32,),
    ".tif": (8, 16),
    ".tiff": (8, 16),
    ".png": (8, 16),
    ".jpg": (8,),
    ".jpeg": (8,),
}


@dataclass
class BatchConvertParams:
    """Options for a batch conversion run.

    Attributes:
        output_format: One of :data:`OUTPUT_FORMATS` (leading dot, e.g. ``".tiff"``).
        bit_depth: ``"auto"`` or an explicit int from
            ``ALLOWED_BIT_DEPTHS[output_format]``. "Auto" picks 8-bit for
            PNG/JPEG, 16-bit for TIFF, and 32-bit (fixed) for FITS/XISF.
        jpeg_quality: JPEG quality 1-100 (only used when output_format is
            ``.jpg``/``.jpeg``).
        skip_existing: Skip files whose destination already exists.
        overwrite: Overwrite an existing destination file (FITS only; other
            writers always overwrite). Ignored when skip_existing is set.
    """

    output_format: str = ".tiff"
    bit_depth: int | str = "auto"
    jpeg_quality: int = 95
    skip_existing: bool = False
    overwrite: bool = True


def _resolve_bit_depth(requested: int | str, fmt: str) -> int:
    """Resolve "auto" (or a bogus explicit value) to a concrete bit depth."""
    allowed = ALLOWED_BIT_DEPTHS.get(fmt, (16,))
    if isinstance(requested, str) and requested.strip().lower() == "auto":
        if fmt in (".png", ".jpg", ".jpeg"):
            return 8
        if fmt in (".tif", ".tiff"):
            return 16
        return allowed[0]  # fits/xisf: always 32
    try:
        depth = int(requested)
    except (TypeError, ValueError):
        return allowed[0]
    return depth if depth in allowed else allowed[0]


def batch_convert(
    paths: Sequence[str | Path],
    output_dir: str | Path,
    params: BatchConvertParams,
    progress: ProgressCallback | None = None,
) -> list[Path]:
    """Convert a set of image files into ``params.output_format``.

    Unreadable/unwritable files are skipped (logged as a warning) rather than
    aborting the whole batch, matching the source tool's behavior.

    Args:
        paths: Source image file paths.
        output_dir: Directory to write converted files into (created if needed).
        params: Conversion options, see :class:`BatchConvertParams`.
        progress: Optional ``(fraction, message)`` callback.

    Returns:
        The list of output paths that were written successfully.
    """
    if progress is None:
        progress = _noop_progress

    fmt = params.output_format
    if not fmt.startswith("."):
        fmt = f".{fmt}"
    fmt = fmt.lower()
    if fmt not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {params.output_format!r}")

    bit_depth = _resolve_bit_depth(params.bit_depth, fmt)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_paths = [Path(p) for p in paths]
    n = len(src_paths)
    outputs: list[Path] = []

    for i, src in enumerate(src_paths):
        frac = i / n if n else 0.0
        dst = out_dir / f"{src.stem}{fmt}"

        if params.skip_existing and dst.exists():
            progress((i + 1) / n if n else 1.0, f"Skipping (exists): {dst.name}")
            continue

        progress(frac, f"Loading {src.name}")
        try:
            image = load_image(src)
        except Exception as exc:
            log.warning("Skipping (load failed): %s (%s)", src.name, exc)
            progress((i + 1) / n if n else 1.0, f"Skipping (load failed): {src.name}")
            continue

        progress(frac, f"Saving {dst.name}")
        try:
            save_image(
                image,
                dst,
                overwrite=params.overwrite,
                bit_depth=bit_depth,
                jpeg_quality=params.jpeg_quality,
            )
        except Exception as exc:
            log.warning("Failed to save %s: %s", dst.name, exc)
            progress((i + 1) / n if n else 1.0, f"ERROR: {src.name} -> {exc}")
            continue

        outputs.append(dst)
        progress((i + 1) / n if n else 1.0, f"Saved {dst.name}")

    progress(1.0, f"Batch convert complete: {len(outputs)}/{n} converted")
    return outputs
