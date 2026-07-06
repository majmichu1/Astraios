"""Batch Rename — rename a set of files from a token template built out of
FITS header values.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

Template syntax (identical to the source tool)::

    LIGHT_{FILTER}_{EXPOSURE:.0f}s_{DATE-OBS:%Y%m%d}_{#03}.{ext}

Tokens
------
``{KEYWORD}``
    Any FITS header keyword (looked up uppercased), e.g. ``{OBJECT}``,
    ``{FILTER}``. Missing keyword -> empty string (matches source: SASpro's
    ``hdr.get(key_up, "")`` fallback).
``{KEYWORD:fmt}``
    A keyword with a format spec:

    * Numeric: passed through Python's ``format(float(value), fmt)``,
      e.g. ``{EXPOSURE:.1f}``.
    * ``DATE-OBS``/``DATE`` with a ``fmt``: parsed as an ISO datetime and
      rendered with ``datetime.strftime(fmt)``, e.g. ``{DATE-OBS:%Y%m%d}``.
    * ``TIME-OBS``/``UTSTART``/``UTC-START`` with a ``fmt``: parsed as
      ``HH:MM:SS`` or ``HH:MM`` and rendered with ``time.strftime(fmt)``.
    * Any of the above that fails to parse falls back to ``str(value)``.
``{#}`` / ``{#03}``
    Sequential counter (``i + index_start``, 0-based ``i``); the optional
    digits after ``#`` zero-pad the width (``{#03}`` -> ``001``, ``002``, ...).
``{ext}``
    The source file's extension, without the dot.
Filters
    Any token body may carry ``|``-separated filters applied to the
    rendered text, e.g. ``{OBJECT|upper}`` or ``{OBJECT|re:(\\w+)|lower}``:

    * ``re:PATTERN`` — regex search; returns group 1 if the pattern has a
      capturing group, else group 0; empty string if no match.
    * ``lower`` / ``upper`` — case conversion.
    * ``slice:a:b`` — Python slice ``text[a:b]`` (``a``/``b`` may be omitted).
    * ``strip`` — ``str.strip()``.

Options (:class:`BatchRenameParams`)
    ``lowercase`` lowercases the whole rendered name; ``slugify`` replaces
    spaces with ``_`` then strips anything outside ``[A-Za-z0-9._-]``;
    ``keep_ext`` appends the source extension when the template doesn't
    already reference ``{ext}``; ``index_start`` offsets the ``{#}`` counter;
    ``output_dir`` moves renamed files into a folder instead of renaming
    in place.

Deviation from the source: header extraction reuses
:func:`astraios.core.image_io.load_xisf` for XISF files (so we don't
reimplement its XML header parsing), but reads FITS headers directly via
``astropy.io.fits.getheader`` rather than ``image_io.load_fits`` — the
latter decodes and normalizes the full pixel array, which would be pure
overhead for a rename tool that only ever looks at header keywords.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from astropy.io import fits

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

# FITS-family suffixes whose header we read directly via astropy (matches
# the source tool's folder-scan extension set).
_FITS_SUFFIXES = (".fit", ".fits", ".fts", ".fz")

_TOKEN_RE = re.compile(r"\{((?:[^{}]|\{[^{}]*\})+)\}")


def _noop_progress(fraction: float, message: str) -> None:
    pass


@dataclass
class BatchRenameParams:
    """Options for a batch rename run.

    Attributes:
        lowercase: Lowercase the entire rendered filename.
        slugify: Replace spaces with ``_``, then strip characters outside
            ``[A-Za-z0-9._-]``.
        keep_ext: Append the source file's extension if the template does
            not already reference ``{ext}``.
        index_start: Value added to the 0-based file index for ``{#}``.
        output_dir: Move renamed files here instead of renaming in place
            (``None`` = rename in place, next to each source file).
    """

    lowercase: bool = False
    slugify: bool = True
    keep_ext: bool = True
    index_start: int = 1
    output_dir: str | Path | None = None


def _read_header(path: Path) -> dict[str, Any]:
    """Read header keywords for a file, or {} if the format has none."""
    suffix = path.suffix.lower()
    if suffix in _FITS_SUFFIXES:
        try:
            return dict(fits.getheader(str(path)))
        except Exception as exc:
            log.warning("Could not read FITS header from %s: %s", path.name, exc)
            return {}
    if suffix == ".xisf":
        try:
            from astraios.core.image_io import load_xisf
            return dict(load_xisf(path).header)
        except Exception as exc:
            log.warning("Could not read XISF header from %s: %s", path.name, exc)
            return {}
    return {}


def _slugify(s: str) -> str:
    s = s.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "", s)


def _split_top_level_pipes(s: str) -> list[str]:
    """Split on ``|`` that isn't inside brackets/escaped (filters chain)."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    esc = False
    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "|" and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts


def _apply_filters(text: str, filters: list[str]) -> str:
    out = str(text)
    for f in filters:
        f = f.strip()
        if f.startswith("re:"):
            pattern = f[3:]
            m = re.search(pattern, out)
            if not m:
                out = ""
            else:
                out = m.group(1) if m.lastindex else m.group(0)
        elif f == "lower":
            out = out.lower()
        elif f == "upper":
            out = out.upper()
        elif f.startswith("slice:"):
            try:
                _, a, b = f.split(":", 2)
                a_i = int(a) if a else None
                b_i = int(b) if b else None
                out = out[a_i:b_i]
            except Exception:
                pass
        elif f == "strip":
            out = out.strip()
    return out


def render_pattern(
    pattern: str, header: dict[str, Any], index: int, index_start: int, file_path: str | Path
) -> str:
    """Render one filename (without applying lowercase/slugify/keep_ext)."""
    file_path = str(file_path)

    def repl(m: re.Match) -> str:
        body = m.group(1)
        parts = _split_top_level_pipes(body)
        key_fmt = parts[0]
        filters = parts[1:] if len(parts) > 1 else []

        if key_fmt.startswith("#"):
            w = key_fmt[1:]
            try:
                pad = int(w) if w else 0
            except ValueError:
                pad = 0
            num = index + index_start
            return f"{num:0{pad}d}" if pad else str(num)

        if key_fmt.lower() == "ext":
            import os
            return os.path.splitext(file_path)[1].lstrip(".")

        if ":" in key_fmt:
            key, fmt = key_fmt.split(":", 1)
        else:
            key, fmt = key_fmt, ""
        key_up = key.upper()
        val = header.get(key_up, "")
        if val is None:
            val = ""

        if fmt and key_up in ("DATE-OBS", "DATE"):
            s = str(val).strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                out = dt.strftime(fmt)
            except Exception:
                out = str(val)
            return _apply_filters(out, filters)

        if fmt and key_up in ("TIME-OBS", "UTSTART", "UTC-START"):
            s = str(val).strip()
            try:
                if s.count(":") == 2:
                    tt = datetime.strptime(s, "%H:%M:%S").time()
                else:
                    tt = datetime.strptime(s, "%H:%M").time()
                out = tt.strftime(fmt)
            except Exception:
                try:
                    tt = datetime.fromisoformat(f"1970-01-01T{s}").time()
                    out = tt.strftime(fmt)
                except Exception:
                    out = str(val)
            return _apply_filters(out, filters)

        if fmt:
            try:
                out = format(float(val), fmt)
                return _apply_filters(out, filters)
            except Exception:
                pass

        return _apply_filters(str(val), filters)

    return _TOKEN_RE.sub(repl, pattern)


def plan_renames(
    paths: Sequence[str | Path],
    template: str,
    params: BatchRenameParams,
    progress: ProgressCallback | None = None,
) -> list[tuple[Path, Path]]:
    """Compute (source, destination) pairs for a template, without touching disk."""
    if progress is None:
        progress = _noop_progress

    src_paths = [Path(p) for p in paths]
    n = len(src_paths)
    out_dir = Path(params.output_dir) if params.output_dir else None

    planned: list[tuple[Path, Path]] = []
    for i, src in enumerate(src_paths):
        progress(i / n if n else 0.0, f"Rendering {src.name}")
        header = _read_header(src)
        base = render_pattern(template, header, i, params.index_start, src)

        if params.keep_ext and "{ext}" not in template:
            ext = src.suffix
            if ext:
                base = f"{base}{ext}"

        if params.lowercase:
            base = base.lower()
        if params.slugify:
            base = _slugify(base)

        folder = out_dir if out_dir is not None else src.parent
        planned.append((src, folder / base))

    progress(1.0, "Preview complete")
    return planned


def find_collisions(planned: list[tuple[Path, Path]]) -> dict[Path, list[Path]]:
    """Return {destination: [sources]} for destinations produced by >1 source."""
    by_dest: dict[Path, list[Path]] = defaultdict(list)
    for src, dst in planned:
        by_dest[dst].append(src)
    return {dst: srcs for dst, srcs in by_dest.items() if len(srcs) > 1}


def batch_rename(
    paths: Sequence[str | Path],
    template: str,
    params: BatchRenameParams,
    progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> list[tuple[Path, Path]]:
    """Rename (or preview renaming) files according to a token template.

    Args:
        paths: Source file paths.
        template: Token template, see the module docstring for syntax.
        params: Rename options, see :class:`BatchRenameParams`.
        progress: Optional ``(fraction, message)`` callback.
        dry_run: If True, only compute and return the planned renames —
            nothing is moved on disk (safe even if the plan has collisions).

    Returns:
        ``dry_run=True``: every planned ``(source, destination)`` pair.
        ``dry_run=False``: the pairs actually renamed (in order).

    Raises:
        ValueError: If ``dry_run=False`` and two or more sources would
            collide on the same destination name — nothing is renamed in
            that case, mirroring the source tool's collision guard.
    """
    if progress is None:
        progress = _noop_progress

    planned = plan_renames(paths, template, params, progress=progress)

    if dry_run:
        progress(1.0, "Preview complete (dry run — no files changed)")
        return planned

    collisions = find_collisions(planned)
    if collisions:
        detail = "; ".join(
            f"{dst.name} <- {[s.name for s in srcs]}" for dst, srcs in collisions.items()
        )
        raise ValueError(f"Rename aborted: name collisions detected ({detail})")

    n = len(planned)
    performed: list[tuple[Path, Path]] = []
    for i, (src, dst) in enumerate(planned):
        progress(i / n if n else 0.0, f"Renaming {src.name} -> {dst.name}")
        if src == dst:
            performed.append((src, dst))
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            performed.append((src, dst))
        except OSError as exc:
            log.error("Failed to rename %s -> %s: %s", src, dst, exc)
            progress((i + 1) / n if n else 1.0, f"ERROR: {src.name} -> {exc}")
            continue

    progress(1.0, f"Rename complete: {len(performed)}/{n} renamed")
    return performed
