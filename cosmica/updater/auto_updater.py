"""Auto-Updater — checks GitHub Releases for new versions."""

from __future__ import annotations

import logging
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests
from packaging.version import Version

import cosmica

log = logging.getLogger(__name__)

GITHUB_REPO = "majmichu1/cosmica"
GITHUB_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


@dataclass
class UpdateInfo:
    available: bool
    current_version: str
    latest_version: str = ""
    download_url: str = ""
    release_notes: str = ""
    asset_name: str = ""


class AutoUpdater:
    """Checks for and applies application updates via GitHub Releases."""

    def __init__(self, repo: str = GITHUB_REPO):
        self._repo = repo
        self._api_url = f"https://api.github.com/repos/{repo}/releases/latest"

    def check_for_updates(self) -> UpdateInfo:
        """Check GitHub for a newer release."""
        current = cosmica.__version__
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

            # Find the right asset for this platform
            system = platform.system().lower()
            asset_patterns = {
                "windows": [".exe", ".msi", "-win"],
                "linux": [".AppImage", ".deb", "-linux"],
                "darwin": [".dmg", "-macos"],
            }
            patterns = asset_patterns.get(system, [])

            for asset in data.get("assets", []):
                name = asset.get("name", "")
                for pattern in patterns:
                    if pattern in name.lower():
                        info.download_url = asset.get("browser_download_url", "")
                        info.asset_name = name
                        break
                if info.download_url:
                    break

            log.info("Update available: %s -> %s", current, info.latest_version)
            return info

        except requests.RequestException as e:
            log.debug("Update check failed: %s", e)
            return info

    def _verify_sha256(self, path: Path, expected: str | None) -> bool:
        """Verify SHA-256 hash of a downloaded file. Returns True if match or no hash provided."""
        if expected is None:
            log.warning("No SHA-256 hash available for integrity check — skipping verification")
            return True
        import hashlib
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual == expected:
            log.info("SHA-256 integrity check passed")
            return True
        log.error(
            "SHA-256 mismatch! Expected: %s, got: %s. File may be corrupted or tampered with.",
            expected, actual,
        )
        Path(path).unlink(missing_ok=True)
        return False

    def download_update(self, url: str, progress_callback=None, sha256: str | None = None) -> Path | None:
        """Download the update installer to a temp directory.

        If sha256 is provided, verifies the file integrity after download.
        """
        try:
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            downloaded = 0

            tmp_dir = Path(tempfile.mkdtemp())
            tmp = tmp_dir / url.split("/")[-1]
            try:
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total > 0:
                            progress_callback(downloaded / total, f"Downloading update... {downloaded // 1024}KB")

                if not self._verify_sha256(tmp, sha256):
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
