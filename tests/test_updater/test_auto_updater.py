"""Tests for the auto-updater hardening."""

from astraios.updater.auto_updater import AutoUpdater


class TestVerifySha256:
    def test_missing_hash_fails_closed(self, tmp_path):
        """An installer binary must not be accepted without a checksum."""
        f = tmp_path / "installer.bin"
        f.write_bytes(b"payload")
        assert AutoUpdater()._verify_sha256(f, None, require=True) is False

    def test_missing_hash_ok_when_not_required(self, tmp_path):
        f = tmp_path / "payload.bin"
        f.write_bytes(b"payload")
        assert AutoUpdater()._verify_sha256(f, None, require=False) is True

    def test_matching_hash_passes(self, tmp_path):
        import hashlib

        f = tmp_path / "installer.bin"
        f.write_bytes(b"payload")
        digest = hashlib.sha256(b"payload").hexdigest().upper()  # case-insensitive
        assert AutoUpdater()._verify_sha256(f, digest) is True

    def test_mismatch_deletes_and_fails(self, tmp_path):
        f = tmp_path / "installer.bin"
        f.write_bytes(b"payload")
        assert AutoUpdater()._verify_sha256(f, "0" * 64) is False
        assert not f.exists()


class TestAssetSelection:
    def test_checksum_sidecar_never_selected(self, monkeypatch):
        """Regression: substring matching picked 'Setup.exe.sha256' as the
        installer because it contains '.exe'."""

        class _Resp:
            status_code = 200

            def json(self):
                return {
                    "tag_name": "v99.0.0",
                    "body": "notes",
                    "assets": [
                        {"name": "Astraios-Setup.exe.sha256",
                         "browser_download_url": "https://x/sha"},
                        {"name": "Astraios-Setup.exe",
                         "browser_download_url": "https://x/exe"},
                    ],
                }

        monkeypatch.setattr(
            "astraios.updater.auto_updater.requests.get",
            lambda *a, **k: _Resp(),
        )
        monkeypatch.setattr("astraios.updater.auto_updater.platform.system", lambda: "Windows")
        info = AutoUpdater().check_for_updates()
        assert info.asset_name == "Astraios-Setup.exe"
        assert info.download_url == "https://x/exe"
