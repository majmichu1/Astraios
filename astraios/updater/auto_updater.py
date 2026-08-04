"""Auto-Updater — checks GitHub Releases for new versions."""

from __future__ import annotations

import logging
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests
from packaging.version import Version

import astraios

log = logging.getLogger(__name__)

GITHUB_REPO = "majmichu1/astraios"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


# Sidecar files that must never be picked as the installer itself.
_CHECKSUM_SUFFIXES = (".sha256", ".sha512", ".sha1", ".sig", ".asc", ".sum")


@dataclass
class UpdateInfo:
    available: bool
    current_version: str
    latest_version: str = ""
    download_url: str = ""
    release_notes: str = ""
    asset_name: str = ""
    sha256: str = ""


class AutoUpdater:
    """Checks for and applies application updates via GitHub Releases."""

    def __init__(self, repo: str = GITHUB_REPO):
        self._repo = repo
        self._api_url = f"https://api.github.com/repos/{repo}/releases/latest"

    def check_for_updates(self) -> UpdateInfo:
        """Check GitHub for a newer release."""
        current = astraios.__version__
        info = UpdateInfo(available=False, current_version=current)

        try:
            resp = requests.get(
                self._api_url,
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=10,
            )
            if resp.status_code != 200:
                log.debug("Update check HTTP %d", resp.status_code)
                return info

            data = resp.json()
            tag = data.get("tag_name", "").lstrip("v")
            if not tag:
                return info

            try:
                latest = Version(tag)
                current_v = Version(current)
            except Exception:
                return info

            if latest <= current_v:
                return info

            info.available = True
            info.latest_version = str(latest)
            info.release_notes = data.get("body", "")

            # Find the right asset for this platform. Substring matching
            # used to pick the ".exe.sha256" sidecar as the "installer";
            # sidecars are excluded and the checksum sidecar is collected
            # for integrity verification instead.
            system = platform.system().lower()
            asset_patterns = {
                "windows": [".exe", ".msi", "-win"],
                "linux": [".appimage", ".deb", "-linux"],
                "darwin": [".dmg", "-macos"],
            }
            patterns = asset_patterns.get(system, [])

            assets = data.get("assets", [])
            checksums: dict[str, str] = {}
            for asset in assets:
                name = asset.get("name", "")
                if name.lower().endswith(".sha256"):
                    checksums[name[:-7].lower()] = asset.get("browser_download_url", "")

            for asset in assets:
                name = asset.get("name", "")
                lname = name.lower()
                if lname.endswith(_CHECKSUM_SUFFIXES):
                    continue
                for pattern in patterns:
                    if pattern in lname:
                        info.download_url = asset.get("browser_download_url", "")
                        info.asset_name = name
                        info.sha256 = self._fetch_sha256(checksums.get(lname))
                        break
                if info.download_url:
                    break

            log.info("Update available: %s -> %s", current, info.latest_version)
            return info

        except requests.RequestException as e:
            log.debug("Update check failed: %s", e)
            return info

    def _fetch_sha256(self, sidecar_url: str | None) -> str:
        """Download and parse a ``.sha256`` sidecar (``<hash>  <name>`` or
        a bare hash). Returns "" when unavailable or unparseable."""
        if not sidecar_url:
            return ""
        try:
            resp = requests.get(sidecar_url, timeout=10)
            resp.raise_for_status()
            first = resp.text.strip().splitlines()[0].strip()
            candidate = first.split()[0] if first else ""
            if len(candidate) == 64 and all(c in "0123456789abcdefABCDEF" for c in candidate):
                return candidate.lower()
        except Exception as e:
            log.debug("Checksum sidecar fetch failed: %s", e)
        return ""

    def _verify_sha256(self, path: Path, expected: str | None, require: bool = True) -> bool:
        """Verify the SHA-256 of a downloaded file.

        Fail-closed: without an expected hash the file is REJECTED when
        ``require`` is true — an installer binary must not be accepted on
        TLS alone.
        """
        if not expected:
            if require:
                log.error("No SHA-256 available — refusing unverified download")
            else:
                log.warning("No SHA-256 available — skipping verification")
            return not require
        import hashlib
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual == expected.lower():
            log.info("SHA-256 integrity check passed")
            return True
        log.error(
            "SHA-256 mismatch! Expected: %s, got: %s. File may be corrupted or tampered with.",
            expected, actual,
        )
        Path(path).unlink(missing_ok=True)
        return False

    def download_update(
        self,
        url: str,
        progress_callback=None,
        sha256: str | None = None,
        asset_name: str | None = None,
        require_hash: bool = True,
    ) -> Path | None:
        """Download the update installer to a temp directory.

        The download is rejected unless its SHA-256 matches ``sha256``
        (``require_hash=False`` only for non-executable payloads).
        """
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            tmp_dir = Path(tempfile.mkdtemp())
            # Derive the filename from the release asset, not the URL path
            # (empty for trailing slashes, and attacker-influenceable).
            name = asset_name or url.rstrip("/").split("/")[-1] or "update.bin"
            tmp = tmp_dir / Path(name).name
            try:
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total > 0:
                            progress_callback(
                                downloaded / total,
                                f"Downloading update... {downloaded // 1024}KB",
                            )

                if not self._verify_sha256(tmp, sha256, require=require_hash):
                    import shutil
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return None

                log.info("Update downloaded: %s", tmp)
                return tmp
            except Exception:
                # Clean up temp dir on failure
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        except Exception as e:
            log.error("Download failed: %s", e)
            return None
