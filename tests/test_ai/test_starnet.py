"""Tests for StarNet subprocess integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from astraios.ai.inference.starnet import StarNetResult, find_starnet_binary, run_starnet


def _mono_image(h=64, w=64, value=0.5):
    return np.full((h, w), value, dtype=np.float32)


def _color_image(c=3, h=64, w=64, value=0.5):
    return np.full((c, h, w), value, dtype=np.float32)


class TestStarNetResult:
    def test_dataclass_fields(self):
        img = _mono_image()
        result = StarNetResult(starless=img)
        assert result.success is True
        assert result.message == ""
        assert result.stars_only is None
        np.testing.assert_array_equal(result.starless, img)

    def test_failure_result(self):
        img = _mono_image()
        result = StarNetResult(
            starless=img,
            success=False,
            message="StarNet not found",
        )
        assert result.success is False
        assert "not found" in result.message

    def test_with_stars_only(self):
        starless = _mono_image(value=0.3)
        stars = _mono_image(value=0.2)
        result = StarNetResult(starless=starless, stars_only=stars, success=True)
        assert result.stars_only is not None
        assert result.stars_only.shape == starless.shape


class TestFindStarnetBinary:
    @patch("shutil.which", return_value=None)
    def test_returns_none_when_not_installed(self, mock_which):
        """Should return None when no StarNet binary is found anywhere."""
        result = find_starnet_binary()
        assert result is None

    @patch("shutil.which")
    def test_finds_starnet_in_path(self, mock_which):
        """Should find starnet++ in PATH."""
        def side_effect(name):
            if name == "starnet++":
                return "/usr/local/bin/starnet++"
            return None
        mock_which.side_effect = side_effect
        result = find_starnet_binary()
        assert result == Path("/usr/local/bin/starnet++")

    @patch("shutil.which", return_value=None)
    def test_checks_common_install_paths(self, mock_which, tmp_path):
        """Should check common installation locations."""
        # When shutil.which returns None and no common paths exist,
        # should return None
        result = find_starnet_binary()
        assert result is None


class TestRunStarnet:
    def test_returns_failure_when_binary_not_found(self):
        """Should return failure result when StarNet binary is not found."""
        data = _mono_image()
        result = run_starnet(data, starnet_path="/nonexistent/path/starnet++")
        assert result.success is False
        assert "not found" in result.message.lower() or "not found" in result.message
        np.testing.assert_array_equal(result.starless, data)

    def test_returns_failure_with_none_path(self):
        """When auto-detection fails, should return failure."""
        data = _mono_image()
        with patch("astraios.ai.inference.starnet.find_starnet_binary", return_value=None):
            result = run_starnet(data)
        assert result.success is False

    @patch("astraios.ai.inference.starnet.find_starnet_binary")
    @patch("astraios.ai.inference.starnet._write_starnet_tiff")
    @patch("astraios.ai.inference.starnet._read_starnet_tiff")
    @patch("subprocess.run")
    def test_successful_run(self, mock_subprocess, mock_read, mock_write, mock_find, tmp_path):
        """Simulate a successful StarNet run with mocked subprocess + TIFF I/O."""
        binary = tmp_path / "StarNetv2CLI"
        binary.touch()
        binary.chmod(0o755)
        mock_find.return_value = binary

        data = _mono_image(value=0.6)
        starless = _mono_image(value=0.4)

        mock_subprocess.return_value = MagicMock(returncode=0, stderr="", stdout="")
        # _read_starnet_tiff returns the starless array directly.
        mock_read.return_value = starless

        # The output file must "exist" for the temp dir.
        original_exists = Path.exists
        def patched_exists(self):
            if "starless" in str(self):
                return True
            return original_exists(self)

        with patch.object(Path, "exists", patched_exists):
            result = run_starnet(data, starnet_path=binary)

        assert result.success is True
        np.testing.assert_array_equal(result.starless, starless)

    @patch("astraios.ai.inference.starnet.find_starnet_binary")
    @patch("astraios.ai.inference.starnet._write_starnet_tiff")
    @patch("subprocess.run")
    def test_subprocess_failure(self, mock_subprocess, mock_write, mock_find, tmp_path):
        """StarNet returning non-zero exit code should be reported as failure."""
        binary = tmp_path / "StarNetv2CLI"
        binary.touch()
        binary.chmod(0o755)
        mock_find.return_value = binary

        data = _mono_image()
        mock_subprocess.return_value = MagicMock(returncode=1, stderr="Segfault", stdout="")

        result = run_starnet(data, starnet_path=binary)
        assert result.success is False
        assert "code 1" in result.message or "Segfault" in result.message

    def test_input_image_preserved_on_failure(self):
        """On failure, starless should be a copy of the input."""
        data = _mono_image(value=0.7)
        result = run_starnet(data, starnet_path="/nonexistent/starnet++")
        assert result.success is False
        np.testing.assert_array_equal(result.starless, data)
        # Verify it is a copy, not the same object
        assert result.starless is not data
