"""The release workflow publishes one ``<asset>.sha256`` per asset; the
updater and the installers must agree on the format and never mistake a
sidecar for the asset itself."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from astraios.updater.auto_updater import _CHECKSUM_SUFFIXES, AutoUpdater
from scripts.check_release_checksums import check, parse_sidecar

REPO = Path(__file__).resolve().parents[2]
ASSETS = [
    "Astraios-Setup-0.1.25-alpha.exe",
    "astraios-0.1.25-py3-none-any.whl",
    "Astraios-0.1.25-alpha-x86_64.AppImage",
    "install-astraios.sh",
    "install-astraios-macos.sh",
]


def _make_release(tmp_path: Path) -> tuple[Path, Path]:
    art = tmp_path / "artifacts"
    chk = tmp_path / "checksums"
    chk.mkdir()
    for i, name in enumerate(ASSETS):
        d = art / f"job{i}"
        d.mkdir(parents=True)
        (d / name).write_bytes(name.encode() * 100)
        digest = hashlib.sha256((d / name).read_bytes()).hexdigest()
        (chk / f"{name}.sha256").write_text(f"{digest}  {name}\n")
    return chk, art


def test_workflow_format_is_what_the_updater_parses(tmp_path):
    chk, art = _make_release(tmp_path)
    assert check(chk, art) == []
    for sc in chk.glob("*.sha256"):
        digest, name = parse_sidecar(sc.read_text())
        assert sc.name == f"{name}.sha256"
        # the updater's parser accepts the same "<hash>  <name>" line
        (tmp_path / "dl").mkdir(exist_ok=True)
        f = tmp_path / "dl" / name
        f.write_bytes(name.encode() * 100)
        assert AutoUpdater()._verify_sha256(f, sc.read_text().split()[0], require=True)


def test_checker_rejects_bad_sidecars(tmp_path):
    chk, art = _make_release(tmp_path)
    (chk / "install-astraios.sh.sha256").write_text("deadbeef  install-astraios.sh\n")
    (chk / "ghost.exe.sha256").write_text("0" * 64 + "  ghost.exe\n")
    problems = check(chk, art)
    assert any("install-astraios.sh.sha256" in p and "malformed" in p for p in problems)
    assert any("ghost.exe" in p and "not among" in p for p in problems)
    (chk / "ghost.exe.sha256").unlink()
    (chk / "install-astraios.sh.sha256").write_text("0" * 64 + "  install-astraios.sh\n")
    assert any("does not match" in p for p in check(chk, art))


def test_checker_refuses_an_empty_release(tmp_path):
    (tmp_path / "c").mkdir()
    (tmp_path / "a").mkdir()
    assert check(tmp_path / "c", tmp_path / "a") == ["no .sha256 sidecars were produced"]


@pytest.mark.parametrize("name", [f"{a}.sha256" for a in ASSETS])
def test_sidecars_are_never_picked_as_assets(name):
    assert name.lower().endswith(_CHECKSUM_SUFFIXES)


def test_installer_wheel_lookup_skips_sidecars():
    """The Linux and macOS installers grep the release JSON for the wheel URL;
    the pattern must not match the wheel's .sha256 sidecar."""
    api = (
        '"browser_download_url": "https://x/astraios-0.1.25-py3-none-any.whl.sha256"\n'
        '"browser_download_url": "https://x/astraios-0.1.25-py3-none-any.whl"\n'
    )
    scripts = ("packaging/linux/install-astraios.sh", "packaging/macos/install-astraios-macos.sh")
    for script in scripts:
        text = (REPO / script).read_text()
        m = re.search(r"grep -o '([^']+)'", text)
        assert m, script
        pattern = m.group(1)
        hits = re.findall(pattern, api)
        assert hits and all(h.endswith('.whl"') for h in hits), (script, hits)
