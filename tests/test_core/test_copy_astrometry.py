"""Tests for WCS/astrometry-solution copying (ported from SASpro copyastro.py)."""

from astraios.core.copy_astrometry import (
    copy_astrometry,
    extract_wcs_dict,
    wcs_keywords_present,
)


def _solved_header(**overrides):
    hdr = {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": 4096,
        "NAXIS2": 4096,
        "OBJECT": "M42",
        "EXPTIME": 300.0,
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2048.0,
        "CRPIX2": 2048.0,
        "CRVAL1": 83.822083,
        "CRVAL2": -5.391111,
        "CD1_1": -0.0002,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.0002,
        "RADESYS": "ICRS",
        "EQUINOX": 2000.0,
    }
    hdr.update(overrides)
    return hdr


def _unsolved_header(**overrides):
    hdr = {
        "SIMPLE": True,
        "BITPIX": -32,
        "NAXIS": 2,
        "NAXIS1": 1024,
        "NAXIS2": 1024,
        "OBJECT": "M42-target",
        "EXPTIME": 60.0,
        "TELESCOP": "MyScope",
    }
    hdr.update(overrides)
    return hdr


class TestWcsKeywordsPresent:
    def test_true_when_solved(self):
        assert wcs_keywords_present(_solved_header())

    def test_false_when_unsolved(self):
        assert not wcs_keywords_present(_unsolved_header())

    def test_false_when_only_partial_crval(self):
        hdr = _unsolved_header(CRVAL1=10.0)  # missing CRVAL2
        assert not wcs_keywords_present(hdr)


class TestExtractWcsDict:
    def test_extracts_only_wcs_keys(self):
        src = _solved_header()
        w = extract_wcs_dict(src)
        assert w["CTYPE1"] == "RA---TAN"
        assert w["CRVAL1"] == src["CRVAL1"]
        assert "OBJECT" not in w
        assert "EXPTIME" not in w

    def test_sip_keys_extracted_when_present(self):
        src = _solved_header(
            **{
                "CTYPE1": "RA---TAN-SIP",
                "CTYPE2": "DEC--TAN-SIP",
                "A_ORDER": 2,
                "B_ORDER": 2,
                "A_0_2": 1.2e-6,
                "A_1_1": -3.4e-7,
                "B_2_0": 5.6e-7,
            }
        )
        w = extract_wcs_dict(src)
        assert w["A_ORDER"] == 2
        assert w["B_ORDER"] == 2
        assert w["A_0_2"] == 1.2e-6
        assert w["A_1_1"] == -3.4e-7
        assert w["B_2_0"] == 5.6e-7

    def test_crota_and_pv_keys_extracted(self):
        src = _solved_header(CROTA2=45.0, PV1_1=1.0, PV1_2=0.0)
        w = extract_wcs_dict(src)
        assert w["CROTA2"] == 45.0
        assert w["PV1_1"] == 1.0
        assert w["PV1_2"] == 0.0

    def test_no_wcs_keys_returns_empty(self):
        # NAXIS1/NAXIS2 are part of SASpro's WCS key set (solver context), so
        # exclude them here to test the "truly no WCS content" case.
        hdr = {k: v for k, v in _unsolved_header().items() if k not in ("NAXIS1", "NAXIS2")}
        assert extract_wcs_dict(hdr) == {}


class TestCopyAstrometry:
    def test_wcs_keys_copied_from_source(self):
        source = _solved_header()
        target = _unsolved_header()
        out = copy_astrometry(source, target)
        for key in ("CTYPE1", "CTYPE2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2",
                    "CD1_1", "CD1_2", "CD2_1", "CD2_2", "RADESYS", "EQUINOX", "WCSAXES"):
            assert out[key] == source[key]

    def test_non_wcs_target_keys_preserved(self):
        source = _solved_header()
        target = _unsolved_header()
        out = copy_astrometry(source, target)
        assert out["OBJECT"] == "M42-target"
        assert out["EXPTIME"] == 60.0
        assert out["TELESCOP"] == "MyScope"

    def test_marks_solution_flag(self):
        out = copy_astrometry(_solved_header(), _unsolved_header())
        assert out["HasAstrometricSolution"] is True

    def test_target_stale_wcs_is_replaced_not_merged(self):
        source = _solved_header(CRVAL1=10.0, CRVAL2=20.0)
        target = _solved_header(CRVAL1=999.0, CRVAL2=999.0, OBJECT="stale-target")
        out = copy_astrometry(source, target)
        assert out["CRVAL1"] == 10.0
        assert out["CRVAL2"] == 20.0
        assert out["OBJECT"] == "stale-target"

    def test_sip_keys_copied_when_present(self):
        source = _solved_header(
            **{"CTYPE1": "RA---TAN-SIP", "CTYPE2": "DEC--TAN-SIP", "A_ORDER": 2, "A_0_2": 1e-6}
        )
        out = copy_astrometry(source, _unsolved_header())
        assert out["A_ORDER"] == 2
        assert out["A_0_2"] == 1e-6
        assert out["CTYPE1"] == "RA---TAN-SIP"

    def test_raises_when_source_has_no_wcs(self):
        import pytest

        with pytest.raises(ValueError):
            copy_astrometry(_unsolved_header(), _unsolved_header())

    def test_wcs_keywords_present_on_result(self):
        out = copy_astrometry(_solved_header(), _unsolved_header())
        assert wcs_keywords_present(out)
