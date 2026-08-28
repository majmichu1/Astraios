"""FITS row order: Siril writes ``ROWORDER = 'BOTTOM-UP'`` and until 2026-08 we
ignored the key, so every Siril sub, master or stack opened upside down and a
Siril flat would have been applied to the wrong corners. Our own files carry
``ROWORDER = 'TOP-DOWN'`` so Siril shows them right side up."""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from astraios.core.image_io import ImageData, fits_bottom_up, load_image, save_image
from astraios.core.stacking import _load_fits_tile, _write_aligned_fits


def _write(path, data, roworder=None):
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.header["CREATOR"] = "Astraios"  # float data taken as-is
    if roworder:
        hdu.header["ROWORDER"] = roworder
    hdu.writeto(str(path), overwrite=True)
    return path


def _mono():
    d = np.full((12, 9), 0.2, np.float32)
    d[2, 4] = 0.9  # third row from the top when the file is top-down
    return d


def test_bottom_up_mono_is_flipped_on_load(tmp_path):
    p = _write(tmp_path / "siril.fits", _mono(), "BOTTOM-UP")
    img = load_image(p)
    assert img.data[12 - 1 - 2, 4] == np.float32(0.9)
    assert img.data[2, 4] == np.float32(0.2)
    assert img.header["ROWORDER"] == "TOP-DOWN"


def test_bottom_up_color_is_flipped_on_load(tmp_path):
    c = np.stack([_mono(), _mono() * 0.5, _mono() * 0.25])
    p = _write(tmp_path / "siril_rgb.fits", c, "BOTTOM-UP")
    img = load_image(p)
    assert img.data.shape == (3, 12, 9)
    assert img.data[0, 9, 4] == np.float32(0.9)
    assert img.data[2, 9, 4] == np.float32(0.9 * 0.25)


def test_top_down_and_unmarked_files_are_unchanged(tmp_path):
    for ro in ("TOP-DOWN", None):
        p = _write(tmp_path / f"cam_{ro}.fits", _mono(), ro)
        assert load_image(p).data[2, 4] == np.float32(0.9)


def test_saved_files_declare_top_down_and_round_trip(tmp_path):
    p = tmp_path / "ours.fits"
    save_image(ImageData(data=_mono()), p)
    assert fits.getheader(str(p))["ROWORDER"] == "TOP-DOWN"
    assert load_image(p).data[2, 4] == np.float32(0.9)
    # a Siril file loaded and saved again is top-down on disk, not flipped twice
    src = _write(tmp_path / "siril.fits", _mono(), "BOTTOM-UP")
    save_image(load_image(src), tmp_path / "resaved.fits")
    assert load_image(tmp_path / "resaved.fits").data[9, 4] == np.float32(0.9)


def test_tiled_loader_matches_the_full_loader_for_bottom_up(tmp_path):
    p = _write(tmp_path / "siril.fits", _mono(), "BOTTOM-UP")
    full = load_image(p).data
    for y0, y1 in ((0, 5), (5, 12), (3, 4)):
        np.testing.assert_array_equal(_load_fits_tile(p, y0, y1), full[y0:y1])
    c = np.stack([_mono(), _mono() * 0.5, _mono() * 0.25])
    pc = _write(tmp_path / "siril_rgb.fits", c, "BOTTOM-UP")
    fullc = load_image(pc).data
    np.testing.assert_array_equal(_load_fits_tile(pc, 8, 11), fullc[:, 8:11])


def test_aligned_writer_declares_top_down(tmp_path):
    out = tmp_path / "aligned.fits"
    _write_aligned_fits(ImageData(data=_mono(), header={"ROWORDER": "BOTTOM-UP"}), out)
    hdr = fits.getheader(str(out))
    assert hdr["ROWORDER"] == "TOP-DOWN" and hdr["CREATOR"] == "Astraios"
    assert load_image(out).data[2, 4] == np.float32(0.9)


def test_fits_bottom_up_helper():
    assert fits_bottom_up({"ROWORDER": "bottom-up "})
    assert not fits_bottom_up({"ROWORDER": "TOP-DOWN"})
    assert not fits_bottom_up({})
