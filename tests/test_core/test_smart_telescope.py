"""Tests for smart-telescope detection.

The point of this module is to stop handing a Seestar owner a calibration
and stacking pipeline they do not need, so the tests care about two things:
that real-world header variations are recognised, and that ordinary
telescope data is NOT misidentified.
"""

import numpy as np

from astraios.core.smart_telescope import (
    KNOWN_TELESCOPES,
    describe,
    identify,
    looks_prestacked,
    stack_count,
)


class TestIdentification:
    def test_seestar_s50_in_telescop(self):
        assert identify({"TELESCOP": "Seestar S50"}).key == "seestar_s50"

    def test_matching_is_case_and_format_insensitive(self):
        """Vendors and firmware revisions do not agree on formatting."""
        for value in ("Seestar S50", "SEESTAR S50", "seestar_s50", "ZWO SeeStar S50 v2"):
            assert identify({"TELESCOP": value}).key == "seestar_s50"

    def test_identity_can_live_in_any_of_several_keywords(self):
        for key in ("TELESCOP", "INSTRUME", "ORIGIN", "CREATOR", "PROGRAM"):
            assert identify({key: "Seestar S50"}).key == "seestar_s50", key

    def test_specific_model_wins_over_the_family(self):
        """A bare 'seestar' entry exists as a fallback; it must not shadow S50."""
        assert identify({"TELESCOP": "Seestar S50"}).key == "seestar_s50"
        assert identify({"TELESCOP": "Seestar"}).key == "seestar"

    def test_other_vendors(self):
        assert identify({"TELESCOP": "DWARF 3"}).vendor == "DwarfLab"
        assert identify({"INSTRUME": "Dwarf II"}).vendor == "DwarfLab"
        assert identify({"TELESCOP": "Vaonis Vespera II"}).vendor == "Vaonis"
        assert identify({"TELESCOP": "Unistellar eVscope 2"}).vendor == "Unistellar"

    def test_ordinary_equipment_is_not_misidentified(self):
        for value in ("Takahashi FSQ-106", "GSO RC8", "William Optics RedCat 51", ""):
            assert identify({"TELESCOP": value}) is None, value

    def test_missing_or_empty_header(self):
        assert identify(None) is None
        assert identify({}) is None

    def test_non_string_values_do_not_crash(self):
        assert identify({"TELESCOP": 1234, "INSTRUME": None}) is None


class TestSpecs:
    def test_seestar_s50_specs_are_the_published_ones(self):
        s50 = identify({"TELESCOP": "Seestar S50"})
        assert s50.focal_length_mm == 250.0
        assert s50.aperture_mm == 50.0
        assert s50.focal_ratio == 5.0

    def test_seestar_s50_pixel_scale_matches_published_field_of_view(self):
        """Published FOV is 1.29 deg across 1920 px, i.e. ~2.42 arcsec/px."""
        s50 = identify({"TELESCOP": "Seestar S50"})
        assert abs(s50.pixel_scale_arcsec() - 2.42) < 0.1

    def test_unverified_specs_are_not_invented(self):
        """A wrong focal length would silently poison plate-solve hints, so
        anything unconfirmed must stay None rather than being guessed."""
        s30 = identify({"TELESCOP": "Seestar S30"})
        assert s30.pixel_size_um is None
        assert s30.pixel_scale_arcsec() is None
        dwarf = identify({"TELESCOP": "DWARF 3"})
        assert dwarf.focal_length_mm is None
        assert dwarf.focal_ratio is None

    def test_every_entry_has_a_name_and_aliases(self):
        for scope in KNOWN_TELESCOPES:
            assert scope.name and scope.vendor and scope.aliases
            assert all(a == a.lower() for a in scope.aliases), scope.key


class TestStackDetection:
    def test_stack_count_from_various_keywords(self):
        assert stack_count({"STACKCNT": 120}) == 120
        assert stack_count({"NCOMBINE": 45}) == 45
        assert stack_count({"NIMAGES": "77"}) == 77

    def test_single_frame_is_not_a_stack(self):
        assert stack_count({"STACKCNT": 1}) is None
        assert stack_count({}) is None
        assert stack_count({"STACKCNT": "not a number"}) is None

    def test_smart_telescope_implies_prestacked(self):
        assert looks_prestacked({"TELESCOP": "Seestar S50"})

    def test_stack_count_implies_prestacked(self):
        assert looks_prestacked({"TELESCOP": "GSO RC8", "STACKCNT": 60})

    def test_very_long_exposure_implies_prestacked(self):
        assert looks_prestacked({"TELESCOP": "GSO RC8", "EXPTIME": 3600})

    def test_ordinary_sub_frame_is_not_prestacked(self):
        assert not looks_prestacked({"TELESCOP": "Takahashi FSQ", "EXPTIME": 300})
        assert not looks_prestacked({})
        assert not looks_prestacked(None)


class TestDescription:
    def test_names_the_device_and_the_next_action(self):
        text = describe({"TELESCOP": "Seestar S50"})
        assert "Seestar S50" in text
        assert "Guided Processing" in text

    def test_reports_verified_specs_only(self):
        assert "250mm" in describe({"TELESCOP": "Seestar S50"})
        # the S30's pixel scale is unverified, so it must not appear
        assert "arcsec" not in describe({"TELESCOP": "Seestar S30"})

    def test_mentions_the_sub_frame_count_when_known(self):
        assert "120 sub-frames" in describe({"TELESCOP": "Seestar S50", "STACKCNT": 120})

    def test_stack_without_a_known_device(self):
        text = describe({"TELESCOP": "GSO RC8", "NCOMBINE": 30})
        assert "Stacked image" in text and "30 sub-frames" in text

    def test_says_nothing_about_an_ordinary_frame(self):
        assert describe({"TELESCOP": "Takahashi FSQ", "EXPTIME": 300}) is None
        assert describe({}) is None

    def test_accepts_image_data_without_using_it_wrongly(self):
        data = np.zeros((3, 8, 8), np.float32)
        assert describe({"TELESCOP": "Seestar S50"}, data) is not None
