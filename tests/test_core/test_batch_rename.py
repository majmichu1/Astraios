"""Tests for astraios.core.batch_rename (ported from Seti Astro Suite Pro)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from astraios.core.batch_rename import (
    BatchRenameParams,
    batch_rename,
    find_collisions,
    plan_renames,
    render_pattern,
)


def _make_fits(path: Path, header: dict) -> Path:
    hdu = fits.PrimaryHDU(np.zeros((8, 8), dtype=np.float32))
    for k, v in header.items():
        hdu.header[k] = v
    hdu.writeto(str(path), overwrite=True)
    return path


@pytest.fixture
def sample_files(tmp_path: Path) -> list[Path]:
    files = [
        _make_fits(
            tmp_path / "IMG_0001.fits", {"OBJECT": "M42", "FILTER": "Ha", "EXPTIME": 300.0}
        ),
        _make_fits(
            tmp_path / "IMG_0002.fits", {"OBJECT": "M42", "FILTER": "OIII", "EXPTIME": 300.0}
        ),
        _make_fits(
            tmp_path / "IMG_0003.fits", {"OBJECT": "M31", "FILTER": "Ha", "EXPTIME": 120.5}
        ),
    ]
    return files


class TestRenderPattern:
    def test_basic_keyword(self, sample_files):
        out = render_pattern("{OBJECT}", {"OBJECT": "M42"}, 0, 1, sample_files[0])
        assert out == "M42"

    def test_missing_keyword_falls_back_to_empty(self):
        out = render_pattern("{MISSING}", {}, 0, 1, "x.fits")
        assert out == ""

    def test_counter_unpadded(self):
        assert render_pattern("{#}", {}, 0, 1, "x.fits") == "1"
        assert render_pattern("{#}", {}, 4, 1, "x.fits") == "5"

    def test_counter_padded(self):
        assert render_pattern("{#03}", {}, 0, 1, "x.fits") == "001"
        assert render_pattern("{#03}", {}, 9, 1, "x.fits") == "010"

    def test_index_start_offset(self):
        assert render_pattern("{#}", {}, 0, 100, "x.fits") == "100"

    def test_ext_token(self):
        assert render_pattern("{ext}", {}, 0, 1, "/a/b/frame.FITS") == "FITS"

    def test_numeric_format(self):
        out = render_pattern("{EXPTIME:.0f}", {"EXPTIME": 300.0}, 0, 1, "x.fits")
        assert out == "300"

    def test_date_obs_format(self):
        hdr = {"DATE-OBS": "2024-03-15T04:30:00"}
        out = render_pattern("{DATE-OBS:%Y%m%d}", hdr, 0, 1, "x.fits")
        assert out == "20240315"

    def test_time_obs_format(self):
        hdr = {"TIME-OBS": "04:30:15"}
        out = render_pattern("{TIME-OBS:%H%M%S}", hdr, 0, 1, "x.fits")
        assert out == "043015"

    def test_filter_upper(self):
        out = render_pattern("{OBJECT|upper}", {"OBJECT": "m42"}, 0, 1, "x.fits")
        assert out == "M42"

    def test_filter_lower(self):
        out = render_pattern("{OBJECT|lower}", {"OBJECT": "M42"}, 0, 1, "x.fits")
        assert out == "m42"

    def test_filter_regex_capture_group(self):
        out = render_pattern(r"{OBJECT|re:(\w+)}", {"OBJECT": "M42 core"}, 0, 1, "x.fits")
        assert out == "M42"

    def test_filter_slice(self):
        out = render_pattern("{OBJECT|slice:0:3}", {"OBJECT": "M42-core"}, 0, 1, "x.fits")
        assert out == "M42"

    def test_filter_chain(self):
        out = render_pattern(
            r"{OBJECT|re:(\w+)|lower}", {"OBJECT": "M42 Core"}, 0, 1, "x.fits"
        )
        assert out == "m42"

    def test_full_template(self):
        hdr = {"FILTER": "Ha", "EXPTIME": 300.0, "DATE-OBS": "2024-03-15T04:30:00"}
        out = render_pattern(
            "LIGHT_{FILTER}_{EXPTIME:.0f}s_{DATE-OBS:%Y%m%d}_{#03}", hdr, 0, 1, "x.fits"
        )
        assert out == "LIGHT_Ha_300s_20240315_001"


class TestPlanRenames:
    def test_planned_names_correct(self, sample_files):
        params = BatchRenameParams(slugify=False, keep_ext=False)
        planned = plan_renames(sample_files, "{OBJECT}_{FILTER}_{#}.fits", params)

        names = [dst.name for _src, dst in planned]
        assert names == ["M42_Ha_1.fits", "M42_OIII_2.fits", "M31_Ha_3.fits"]
        # Renamed in place: destination folder matches source folder.
        for src, dst in planned:
            assert dst.parent == src.parent

    def test_output_dir_redirects_destination(self, sample_files, tmp_path):
        dest = tmp_path / "renamed"
        params = BatchRenameParams(slugify=False, output_dir=dest)
        planned = plan_renames(sample_files, "{OBJECT}_{#}.fits", params)
        for _src, dst in planned:
            assert dst.parent == dest

    def test_keep_ext_appends_when_pattern_lacks_ext_token(self, sample_files):
        params = BatchRenameParams(slugify=False, keep_ext=True)
        planned = plan_renames(sample_files, "{OBJECT}_{#}", params)
        assert planned[0][1].name == "M42_1.fits"

    def test_keep_ext_false_no_extension_added(self, sample_files):
        params = BatchRenameParams(slugify=False, keep_ext=False)
        planned = plan_renames(sample_files, "{OBJECT}_{#}", params)
        assert planned[0][1].name == "M42_1"

    def test_lowercase_option(self, sample_files):
        params = BatchRenameParams(lowercase=True, slugify=False, keep_ext=False)
        planned = plan_renames(sample_files, "{OBJECT}_{FILTER}.fits", params)
        assert planned[0][1].name == "m42_ha.fits"

    def test_slugify_option(self, sample_files):
        params = BatchRenameParams(slugify=True, keep_ext=False)
        planned = plan_renames(
            [_make_fits(sample_files[0].parent / "sp.fits", {"OBJECT": "M 42!"})],
            "{OBJECT}.fits", params,
        )
        assert planned[0][1].name == "M_42.fits"

    def test_missing_header_fallback(self, tmp_path):
        blank = _make_fits(tmp_path / "blank.fits", {})
        params = BatchRenameParams(slugify=False, keep_ext=False)
        planned = plan_renames([blank], "{OBJECT}_{FILTER}.fits", params)
        assert planned[0][1].name == "_.fits"

    def test_no_disk_changes_from_planning(self, sample_files):
        params = BatchRenameParams(slugify=False, keep_ext=False)
        plan_renames(sample_files, "{OBJECT}_{#}.fits", params)
        for f in sample_files:
            assert f.exists()


class TestFindCollisions:
    def test_detects_collision(self, sample_files):
        # Two files share OBJECT=M42 -> same target under a pattern that
        # ignores FILTER.
        params = BatchRenameParams(slugify=False, keep_ext=False)
        planned = plan_renames(sample_files, "{OBJECT}.fits", params)
        collisions = find_collisions(planned)
        assert len(collisions) == 1
        dst, srcs = next(iter(collisions.items()))
        assert dst.name == "M42.fits"
        assert len(srcs) == 2

    def test_no_collision_when_unique(self, sample_files):
        params = BatchRenameParams(slugify=False, keep_ext=False)
        planned = plan_renames(sample_files, "{OBJECT}_{FILTER}.fits", params)
        assert find_collisions(planned) == {}


class TestBatchRename:
    def test_dry_run_performs_no_disk_change(self, sample_files):
        params = BatchRenameParams(slugify=False, keep_ext=False)
        result = batch_rename(
            sample_files, "{OBJECT}_{FILTER}.fits", params, dry_run=True
        )
        assert len(result) == 3
        for f in sample_files:
            assert f.exists()  # originals untouched
        # None of the *new* names should exist yet.
        for _src, dst in result:
            if dst not in sample_files:
                assert not dst.exists()

    def test_apply_renames_files(self, sample_files):
        params = BatchRenameParams(slugify=False, keep_ext=False)
        result = batch_rename(
            sample_files, "{OBJECT}_{FILTER}.fits", params, dry_run=False
        )
        assert len(result) == 3
        for src, dst in result:
            assert not src.exists()
            assert dst.exists()

    def test_apply_raises_on_collision_and_touches_nothing(self, sample_files):
        params = BatchRenameParams(slugify=False, keep_ext=False)
        with pytest.raises(ValueError):
            batch_rename(sample_files, "{OBJECT}.fits", params, dry_run=False)
        # Collision must abort before any rename happens.
        for f in sample_files:
            assert f.exists()

    def test_apply_creates_output_dir(self, sample_files, tmp_path):
        dest = tmp_path / "renamed"
        params = BatchRenameParams(slugify=False, output_dir=dest)
        result = batch_rename(
            sample_files, "{OBJECT}_{FILTER}.fits", params, dry_run=False
        )
        assert dest.is_dir()
        for _src, dst in result:
            assert dst.exists()
            assert dst.parent == dest

    def test_progress_callback_invoked(self, sample_files):
        events: list[tuple[float, str]] = []
        params = BatchRenameParams(slugify=False, keep_ext=False)
        batch_rename(
            sample_files, "{OBJECT}_{FILTER}.fits", params,
            progress=lambda f, m: events.append((f, m)), dry_run=True,
        )
        assert events
        assert events[-1][0] == 1.0
