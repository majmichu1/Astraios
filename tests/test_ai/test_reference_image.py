"""Tests for the reference-image prior (hips2fits mocked — no network)."""

import io

import numpy as np

from cosmica.ai import reference_image
from cosmica.ai.reference_image import fetch_reference_image, reference_object_mask


def _fake_fits_bytes(arr):
    from astropy.io import fits

    buf = io.BytesIO()
    fits.HDUList([fits.PrimaryHDU(arr.astype(np.float32))]).writeto(buf)
    return buf.getvalue()


def _patch_fetch(monkeypatch, arr=None, raises=None):
    body = None if arr is None else _fake_fits_bytes(arr)

    def fake_urlopen(url, timeout=20.0):
        if raises is not None:
            raise raises

        class _Resp:
            def __enter__(self_): return self_
            def __exit__(self_, *a): return False
            def read(self_): return body
        return _Resp()

    monkeypatch.setattr(reference_image.urllib.request, "urlopen", fake_urlopen)


def test_fetch_normalises(monkeypatch):
    arr = np.linspace(1000, 20000, 64 * 64).reshape(64, 64)
    _patch_fetch(monkeypatch, arr)
    out = fetch_reference_image(202.0, 47.0, 0.3, 64, 64)
    assert out is not None
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_fetch_network_error_returns_none(monkeypatch):
    _patch_fetch(monkeypatch, raises=OSError("offline"))
    assert fetch_reference_image(202.0, 47.0, 0.3, 64, 64) is None


def test_fetch_bad_args_returns_none():
    assert fetch_reference_image(202.0, 47.0, 0.0, 64, 64) is None
    assert fetch_reference_image(202.0, 47.0, 0.3, 0, 64) is None


def test_object_mask_from_reference(monkeypatch):
    # Reference with a bright blob in the centre → mask covers the centre.
    yy, xx = np.mgrid[0:80, 0:80]
    arr = 1000 + 8000 * np.exp(-(((xx - 40) ** 2 + (yy - 40) ** 2) / (2 * 12 ** 2)))
    _patch_fetch(monkeypatch, arr)
    m = reference_object_mask(202.0, 47.0, 0.3, 400, 320)
    assert m is not None
    assert m.shape == (320, 400)
    assert m.dtype == np.float32
    assert m[160, 200] > 0.5     # centre is object
    assert m[5, 5] < 0.2          # corner is sky


def test_object_mask_blank_returns_none(monkeypatch):
    arr = np.full((64, 64), 5000.0) + np.random.default_rng(0).normal(0, 1, (64, 64))
    _patch_fetch(monkeypatch, arr)
    # No structure stands out → None, so callers don't get a bogus full mask.
    assert reference_object_mask(202.0, 47.0, 0.3, 200, 200) is None
