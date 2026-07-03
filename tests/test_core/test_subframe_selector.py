"""Tests for subframe scoring/rejection (astraios/core/subframe_selector.py).

`score_subframes` drives a `ProcessPoolExecutor`, so each call has real
multiprocessing (forkserver) startup overhead — tests are grouped to keep the
number of `score_subframes` calls small while still exercising ranking,
rejection, and `filter_by_metric` behavior.
"""

import numpy as np
import pytest
from astropy.io import fits

from astraios.core.subframe_selector import (
    SubframeScore,
    SubframeSelectorParams,
    filter_by_metric,
    score_subframes,
)

_POSITIONS = [(30, 30), (90, 40), (60, 90), (100, 100), (40, 110)]


def _make_starfield(h, w, sigma, peak=0.6, bg=0.05, noise=0.002, seed=0):
    """Synthetic light frame: a flat sky background plus 2D-Gaussian "stars"
    at fixed positions. `sigma` controls star sharpness (smaller = sharper).
    """
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
    # CREATOR=Astraios is required so load_fits() treats the data as already
    # normalized and does NOT min-max stretch it on reload (see
    # image_io._normalize_fits_tile) — without this the deliberate sharp/blurred
    # FWHM difference would be scrambled by per-file contrast stretching.
    hdu.header["CREATOR"] = "Astraios"
    hdu.writeto(str(path), overwrite=True)


@pytest.fixture(scope="module")
def sharp_and_blurred(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("subframe_sharp_blur")
    sharp = _make_starfield(140, 140, sigma=1.5, seed=0)
    blurred = _make_starfield(140, 140, sigma=4.0, seed=0)
    sharp_path = tmp / "sharp.fits"
    blurred_path = tmp / "blurred.fits"
    _write_fits(sharp_path, sharp)
    _write_fits(blurred_path, blurred)
    return str(sharp_path), str(blurred_path)


@pytest.fixture(scope="module")
def good_and_outlier(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("subframe_good_outlier")
    paths = []
    for i in range(5):
        img = _make_starfield(140, 140, sigma=1.5, seed=10 + i)
        p = tmp / f"good_{i}.fits"
        _write_fits(p, img)
        paths.append(str(p))
    outlier = _make_starfield(140, 140, sigma=4.0, seed=99)
    outlier_path = tmp / "outlier.fits"
    _write_fits(outlier_path, outlier)
    paths.append(str(outlier_path))
    return paths  # last entry is the deliberately blurred outlier


class TestScoreSubframesRanking:
    def test_sharp_frame_ranks_above_blurred_frame(self, sharp_and_blurred):
        sharp_path, blurred_path = sharp_and_blurred
        results = score_subframes([sharp_path, blurred_path])
        by_path = {r.file_path: r for r in results}

        sharp_r = by_path[sharp_path]
        blurred_r = by_path[blurred_path]

        assert sharp_r.fwhm < blurred_r.fwhm
        assert sharp_r.quality_score > blurred_r.quality_score

    def test_blurred_outlier_gets_lowest_quality_and_highest_fwhm(self, good_and_outlier):
        results = score_subframes(good_and_outlier)
        outlier_result = results[-1]
        good_results = results[:-1]

        assert outlier_result.fwhm > max(g.fwhm for g in good_results)
        assert outlier_result.quality_score < min(g.quality_score for g in good_results)

    def test_progress_and_frame_callbacks_are_invoked(self, sharp_and_blurred):
        sharp_path, blurred_path = sharp_and_blurred
        progress_calls = []
        frame_calls = []
        score_subframes(
            [sharp_path, blurred_path],
            progress=lambda frac, msg: progress_calls.append(frac),
            frame_callback=lambda idx, metrics: frame_calls.append(idx),
        )
        assert progress_calls[-1] == 1.0
        assert sorted(frame_calls) == [0, 1]


class TestSigmaClipRejection:
    def test_aggressive_sigma_rejects_the_outlier(self, good_and_outlier):
        results = score_subframes(
            good_and_outlier, SubframeSelectorParams(rejection_sigma=0.5)
        )
        assert results[-1].accepted is False

    def test_lenient_sigma_accepts_everything(self, good_and_outlier):
        results = score_subframes(
            good_and_outlier, SubframeSelectorParams(rejection_sigma=10.0)
        )
        assert all(r.accepted for r in results)


class TestScoreSubframesEdgeCases:
    def test_empty_input_returns_empty_list(self):
        assert score_subframes([]) == []


class TestFilterByMetric:
    @staticmethod
    def _sample_scores():
        return [
            SubframeScore(
                file_path="a", fwhm=2.0, eccentricity=0.1, snr=10,
                background=0.1, n_stars=5, quality_score=0.5, accepted=True,
            ),
            SubframeScore(
                file_path="b", fwhm=4.0, eccentricity=0.2, snr=20,
                background=0.1, n_stars=5, quality_score=0.8, accepted=True,
            ),
            SubframeScore(
                file_path="c", fwhm=1.0, eccentricity=0.05, snr=5,
                background=0.1, n_stars=5, quality_score=0.2, accepted=True,
            ),
        ]

    def test_top_n_by_fwhm_keeps_the_sharpest_frames(self):
        result = filter_by_metric(self._sample_scores(), metric="fwhm", mode="top_n", top_n=2)
        accepted = {s.file_path for s in result if s.accepted}
        assert accepted == {"a", "c"}  # lowest two FWHM values

    def test_top_percent_by_snr_keeps_the_highest_snr(self):
        result = filter_by_metric(
            self._sample_scores(), metric="snr", mode="top_percent", top_percent=50
        )
        accepted = {s.file_path for s in result if s.accepted}
        assert accepted == {"b"}  # highest SNR, top 50% of 3 -> keep 1

    def test_does_not_mutate_the_input_list(self):
        original = self._sample_scores()
        original_flags = [s.accepted for s in original]
        filter_by_metric(original, metric="fwhm", mode="top_n", top_n=1)
        assert [s.accepted for s in original] == original_flags

    def test_empty_input_returns_empty(self):
        assert filter_by_metric([], metric="fwhm") == []
