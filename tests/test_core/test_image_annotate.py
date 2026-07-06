"""Tests for the image annotate ("What's In My Image") core, ported from
Seti Astro Suite Pro."""

from __future__ import annotations

import pytest
from astropy.io import fits

from astraios.core.image_annotate import (
    AnnotateParams,
    IdentifiedObject,
    identify_objects,
    split_for_finder_chart,
)

SIZE = 300
# M42 (Orion Nebula) — present in astraios.core.dso_catalog verbatim at this
# RA/Dec, so a field centered here is guaranteed to contain it.
CENTER_RA = 83.822
CENTER_DEC = -5.391
SCALE_DEG = 0.0008  # deg/px, tangent-plane

# Far outside any field built from CENTER_RA/DEC above (M31, ~30 deg away in
# dec) — used to confirm out-of-frame objects are excluded.
FAR_RA = 10.685
FAR_DEC = 41.269


def _make_header(size=SIZE, ra=CENTER_RA, dec=CENTER_DEC, scale=SCALE_DEG):
    hdr = fits.Header()
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    # FITS CRPIX is 1-indexed; +1 here makes the *0-indexed* array/WCS pixel
    # (as returned by world_to_pixel) land exactly on size/2.
    hdr["CRPIX1"] = size / 2.0 + 1.0
    hdr["CRPIX2"] = size / 2.0 + 1.0
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec
    hdr["CDELT1"] = -scale
    hdr["CDELT2"] = scale
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    return hdr


class TestIdentifyObjectsDSO:
    def test_finds_m42_in_field_center(self):
        results = identify_objects((SIZE, SIZE), _make_header())
        names = [o.name for o in results]
        assert "M42" in names

        m42 = next(o for o in results if o.name == "M42")
        assert m42.catalog == "Messier"
        assert m42.type == "EN"
        assert m42.magnitude is None
        assert m42.size == pytest.approx(85.0)
        # Field is centered on M42 -> pixel position should land near center.
        assert m42.x == pytest.approx(SIZE / 2.0, abs=1.0)
        assert m42.y == pytest.approx(SIZE / 2.0, abs=1.0)
        assert 0.0 <= m42.x < SIZE
        assert 0.0 <= m42.y < SIZE

    def test_excludes_out_of_field_objects(self):
        results = identify_objects((SIZE, SIZE), _make_header())
        names = [o.name for o in results]
        # M31/NGC 224 sit ~30 deg away in declination from this M42 field.
        assert "M31" not in names
        assert "NGC 224" not in names

    def test_supports_chw_image_shape(self):
        results = identify_objects((3, SIZE, SIZE), _make_header())
        assert any(o.name == "M42" for o in results)

    def test_empty_field_returns_no_results(self):
        # A tiny field, far from any embedded catalog entry.
        header = _make_header(ra=200.0, dec=70.0, scale=0.0001)
        results = identify_objects((SIZE, SIZE), header)
        assert results == []

    def test_deterministic_ordering(self):
        r1 = identify_objects((SIZE, SIZE), _make_header())
        r2 = identify_objects((SIZE, SIZE), _make_header())
        assert [o.name for o in r1] == [o.name for o in r2]
        names = [o.name for o in r1]
        assert names == sorted(names)

    def test_include_dso_false_returns_no_dso(self):
        params = AnnotateParams(include_dso=False)
        results = identify_objects((SIZE, SIZE), _make_header(), params=params)
        assert results == []

    def test_max_dso_caps_result_count(self):
        # A very wide field (many catalog entries) capped to 1 result.
        header = _make_header(scale=0.05)
        params = AnnotateParams(max_dso=1)
        results = identify_objects((SIZE, SIZE), header, params=params)
        assert len(results) <= 1

    def test_progress_callback_invoked(self):
        calls = []
        identify_objects(
            (SIZE, SIZE), _make_header(),
            progress=lambda f, m: calls.append((f, m)),
        )
        assert calls
        assert calls[0][0] == 0.0
        assert calls[-1][0] == 1.0

    def test_broken_progress_callback_does_not_raise(self):
        def bad_progress(_f, _m):
            raise RuntimeError("boom")

        results = identify_objects((SIZE, SIZE), _make_header(), progress=bad_progress)
        assert any(o.name == "M42" for o in results)


class TestEdgeHandling:
    def test_zero_size_image_returns_empty(self):
        assert identify_objects((0, 0), _make_header(size=0)) == []

    def test_unsupported_wcs_header_type_returns_empty_not_raise(self):
        assert identify_objects((SIZE, SIZE), 12345) == []

    def test_unsupported_image_shape_raises(self):
        with pytest.raises(ValueError):
            identify_objects((SIZE,), _make_header())

    def test_degenerate_pixel_scale_returns_empty(self):
        # CDELT of exactly 0 makes the pixel scale non-finite/invalid.
        header = _make_header(scale=0.0)
        assert identify_objects((SIZE, SIZE), header) == []


class TestBrightStarsOptIn:
    def test_disabled_by_default_no_network_or_disk_access(self):
        # include_bright_stars defaults to False -- gaia_catalog must never
        # be touched unless explicitly opted in.
        results = identify_objects((SIZE, SIZE), _make_header())
        assert all(o.catalog != "Gaia DR3" for o in results)

    def test_missing_local_catalog_is_silently_skipped(self, tmp_path):
        # An empty directory has no installed Gaia catalog files -> the
        # bright-star pass should degrade gracefully (no exception), leaving
        # only the DSO results.
        params = AnnotateParams(
            include_bright_stars=True, gaia_catalog_dir=str(tmp_path)
        )
        results = identify_objects((SIZE, SIZE), _make_header(), params=params)
        assert all(o.catalog != "Gaia DR3" for o in results)
        assert any(o.name == "M42" for o in results)


class TestSplitForFinderChart:
    def test_splits_stars_and_dso_by_type(self):
        objects = [
            IdentifiedObject(
                name="M42", catalog="Messier", type="EN", ra=CENTER_RA, dec=CENTER_DEC,
                x=150.0, y=150.0, magnitude=None, size=85.0,
            ),
            IdentifiedObject(
                name="Gaia DR3 123", catalog="Gaia DR3", type="Star",
                ra=CENTER_RA, dec=CENTER_DEC, x=100.0, y=100.0, magnitude=4.2, size=None,
            ),
        ]
        stars, dsos = split_for_finder_chart(objects)
        assert len(stars) == 1 and len(dsos) == 1
        assert stars[0]["name"] == "Gaia DR3 123"
        assert stars[0]["mag"] == pytest.approx(4.2)
        assert dsos[0]["name"] == "M42"
        assert dsos[0]["size_arcmin"] == pytest.approx(85.0)

    def test_empty_input(self):
        assert split_for_finder_chart([]) == ([], [])
