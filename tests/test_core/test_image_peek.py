"""Tests for the Image Peeker core (astraios/core/image_peek.py).

Uses small synthetic FITS frames (flat sky + a handful of Gaussian "stars")
of differing brightness so ``peek_frames`` output can be checked against
known-injected values without depending on real sub data.
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from astraios.core.image_peek import FramePeek, ImagePeekParams, peek_frames

_POSITIONS = [(20, 20), (40, 30), (30, 45), (45, 15)]
_THUMB = 48  # small thumbnail size to keep the test fast and downscaling obvious


def _make_starfield(h, w, bg, sigma=1.8, peak=0.35, noise=0.004, seed=0):
    """Flat sky background of level `bg` plus a few 2D-Gaussian "stars"."""
    rng = np.random.RandomState(seed)
    img = np.full((h, w), bg, dtype=np.float64)
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in _POSITIONS:
        img += peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    img += rng.normal(0, noise, (h, w))
    return np.clip(img, 0, 1).astype(np.float32)


def _write_fits(path, data):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.header["IMAGETYP"] = "Light"
    # CREATOR=Astraios avoids the min-max re-stretch on reload (see
    # image_io._normalize_fits_tile) so the deliberately distinct brightness
    # levels survive the round trip untouched.
    hdu.header["CREATOR"] = "Astraios"
    hdu.header["EXPTIME"] = 30.0
    hdu.header["FILTER"] = "L"
    hdu.writeto(str(path), overwrite=True)


@pytest.fixture(scope="module")
def frame_set(tmp_path_factory):
    """Three mono frames of increasing brightness + one color frame."""
    tmp = tmp_path_factory.mktemp("image_peek_frames")

    paths = {}
    for name, bg in (("dim", 0.05), ("mid", 0.20), ("bright", 0.40)):
        data = _make_starfield(90, 90, bg=bg, seed=hash(name) % 1000)
        p = tmp / f"{name}.fits"
        _write_fits(p, data)
        paths[name] = str(p)

    # Color frame: (3, H, W), each channel a slightly different starfield
    color = np.stack(
        [_make_starfield(80, 80, bg=0.15, seed=s) for s in (1, 2, 3)],
        axis=0,
    )
    color_path = tmp / "color.fits"
    _write_fits(color_path, color)
    paths["color"] = str(color_path)

    # Unreadable "frame": garbage bytes with a .fits extension
    bad_path = tmp / "corrupt.fits"
    bad_path.write_bytes(b"not a real fits file")
    paths["bad"] = str(bad_path)

    return paths


class TestPeekFrames:
    def test_one_result_per_readable_file(self, frame_set):
        ordered = [frame_set["dim"], frame_set["mid"], frame_set["bright"], frame_set["bad"]]
        results = peek_frames(ordered, params=ImagePeekParams(thumbnail_size=_THUMB))

        # The corrupt file is skipped gracefully, not raised.
        assert len(results) == 3
        assert all(isinstance(r, FramePeek) for r in results)
        returned_paths = {r.path for r in results}
        assert frame_set["bad"] not in returned_paths
        assert returned_paths == {frame_set["dim"], frame_set["mid"], frame_set["bright"]}

    def test_thumbnail_is_downscaled(self, frame_set):
        results = peek_frames([frame_set["bright"]], params=ImagePeekParams(thumbnail_size=_THUMB))
        assert len(results) == 1
        thumb = results[0].thumbnail
        assert thumb.ndim == 3
        assert thumb.shape[2] == 3
        assert max(thumb.shape[0], thumb.shape[1]) <= _THUMB
        # Full-resolution dimensions are still reported accurately.
        assert results[0].width == 90
        assert results[0].height == 90
        assert thumb.dtype == np.uint8

    def test_median_ordering_matches_injected_brightness(self, frame_set):
        results = peek_frames(
            [frame_set["bright"], frame_set["dim"], frame_set["mid"]],
            params=ImagePeekParams(thumbnail_size=_THUMB),
        )
        by_path = {r.path: r for r in results}
        dim_median = by_path[frame_set["dim"]].median
        mid_median = by_path[frame_set["mid"]].median
        bright_median = by_path[frame_set["bright"]].median
        assert dim_median < mid_median < bright_median
        # min/max sanity
        for r in results:
            assert r.min_val <= r.median <= r.max_val

    def test_mono_frame_stats(self, frame_set):
        results = peek_frames([frame_set["mid"]], params=ImagePeekParams(thumbnail_size=_THUMB))
        r = results[0]
        assert r.is_color is False
        assert r.n_channels == 1
        assert r.exposure == 30.0
        assert r.filter_name == "L"

    def test_color_frame_handled(self, frame_set):
        results = peek_frames([frame_set["color"]], params=ImagePeekParams(thumbnail_size=_THUMB))
        assert len(results) == 1
        r = results[0]
        assert r.is_color is True
        assert r.n_channels == 3
        assert r.width == 80
        assert r.height == 80
        assert r.thumbnail.shape[2] == 3

    def test_star_measurement_populates_fwhm_and_star_count(self, frame_set):
        results = peek_frames(
            [frame_set["bright"]],
            params=ImagePeekParams(thumbnail_size=_THUMB, measure_stars=True, max_stars=10),
        )
        r = results[0]
        assert r.fwhm is not None
        assert r.fwhm > 0
        assert r.eccentricity is not None
        assert r.n_stars is not None

    def test_measure_stars_false_skips_psf(self, frame_set):
        results = peek_frames(
            [frame_set["bright"]],
            params=ImagePeekParams(thumbnail_size=_THUMB, measure_stars=False),
        )
        r = results[0]
        assert r.fwhm is None
        assert r.eccentricity is None
        assert r.n_stars is None

    def test_progress_callback_reaches_completion(self, frame_set):
        calls = []
        peek_frames(
            [frame_set["dim"], frame_set["mid"]],
            params=ImagePeekParams(thumbnail_size=_THUMB),
            progress=lambda frac, msg: calls.append((frac, msg)),
        )
        assert calls
        assert calls[-1][0] == pytest.approx(1.0)

    def test_empty_input_returns_empty_list(self):
        assert peek_frames([]) == []
