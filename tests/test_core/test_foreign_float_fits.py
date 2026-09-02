"""Foreign float FITS (no CREATOR key) get a nominal scale, not a per-file stretch.

Siril's registered frames are [0, 1] floats whose clamped Lanczos still
overshoots to ~1.07 on saturated stars. Min-max normalizing each file by its
own range gave each frame a different gain, which is what made Siril-registered
subs stack wrongly in Astraios and made a Siril frame look 10% quieter than
ours in a like-for-like comparison.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from astraios.core.image_io import load_image
from astraios.core.stacking import _fits_float_range, _load_fits_tile


def _write(path, data):
    fits.PrimaryHDU(data.astype(np.float32)).writeto(str(path), overwrite=True)
    return path


def test_unit_range_with_overshoot_is_left_alone(tmp_path):
    d = np.full((20, 30), 0.2, np.float32)
    d[5, 5] = 1.075  # Lanczos overshoot on a saturated star
    img = load_image(_write(tmp_path / "siril.fits", d))
    assert img.data[0, 0] == np.float32(0.2)
    assert img.data[5, 5] == np.float32(1.075)


def test_frames_with_different_maxima_share_one_gain(tmp_path):
    a = np.full((20, 30), 0.2, np.float32)
    b = a.copy()
    b[3, 3] = 1.07
    ia = load_image(_write(tmp_path / "a.fits", a)).data
    ib = load_image(_write(tmp_path / "b.fits", b)).data
    assert ia[0, 0] == ib[0, 0] == np.float32(0.2)


def test_adu_floats_scale_by_16_bit_full_well(tmp_path):
    d = np.full((20, 30), 20000.0, np.float32)
    d[2, 2] = 40000.0
    img = load_image(_write(tmp_path / "adu.fits", d)).data
    assert abs(img[0, 0] - 20000 / 65535) < 1e-6
    assert abs(img[2, 2] - 40000 / 65535) < 1e-6


def test_exotic_range_falls_back_to_min_max(tmp_path):
    d = np.linspace(0, 1e6, 600, dtype=np.float32).reshape(20, 30)
    img = load_image(_write(tmp_path / "big.fits", d)).data
    assert img.min() == 0.0 and abs(img.max() - 1.0) < 1e-6


def test_tiled_loader_agrees_with_full_loader(tmp_path):
    rng = np.random.default_rng(0)
    d = (0.2 + rng.normal(0, 0.01, (40, 30))).astype(np.float32)
    d[10, 10] = 1.06
    p = _write(tmp_path / "siril.fits", d)
    full = load_image(p).data
    rng_ = _fits_float_range(p)
    for y0, y1 in ((0, 13), (13, 26), (26, 40)):
        np.testing.assert_array_equal(_load_fits_tile(p, y0, y1, rng_), full[y0:y1])
