#!/usr/bin/env python3
"""Verify the release checksum sidecars before they are uploaded.

Usage: ``check_release_checksums.py <checksums dir> <artifacts dir>``

Every ``<asset>.sha256`` must name an asset that exists among the build
artifacts, be formatted as ``<64 hex>  <asset name>`` (what ``sha256sum -c``
and the in-app updater parse), and match the file's real digest. Exits
non-zero on the first problem so a bad sidecar never reaches a release.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


def parse_sidecar(text: str) -> tuple[str, str]:
    """Return (hash, asset name) from a sidecar's content, or raise ValueError."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one line, got {len(lines)}")
    m = LINE.match(lines[0])
    if not m:
        raise ValueError(f"malformed line: {lines[0]!r}")
    return m.group(1), m.group(2)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(checksums_dir: Path, artifacts_dir: Path) -> list[str]:
    """Return a list of problems (empty when every sidecar is right)."""
    problems: list[str] = []
    assets = {p.name: p for p in artifacts_dir.rglob("*") if p.is_file()}
    sidecars = sorted(checksums_dir.glob("*.sha256"))
    if not sidecars:
        return ["no .sha256 sidecars were produced"]
    for sc in sidecars:
        expected_asset = sc.name[: -len(".sha256")]
        try:
            digest, named = parse_sidecar(sc.read_text())
        except ValueError as exc:
            problems.append(f"{sc.name}: {exc}")
            continue
        if named != expected_asset:
            problems.append(f"{sc.name}: names {named!r}, sidecar is for {expected_asset!r}")
        asset = assets.get(expected_asset)
        if asset is None:
            problems.append(f"{sc.name}: asset {expected_asset!r} not among the artifacts")
            continue
        if sha256_of(asset) != digest:
            problems.append(f"{sc.name}: digest does not match {expected_asset}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    problems = check(Path(argv[1]), Path(argv[2]))
    for p in problems:
        print("ERROR:", p)
    if not problems:
        print("checksum sidecars OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
