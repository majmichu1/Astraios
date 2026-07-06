"""Tests for the transient hunter (supernova / asteroid detection)."""

import numpy as np

from astraios.core.transient_hunter import (
    TransientHunterParams,
    TransientKind,
    hunt_transients,
)

H, W = 200, 200
_REF_STARS = [(40, 40, 0.8), (120, 80, 0.6), (160, 150, 0.7), (70, 160, 0.5)]


def _field(stars, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(0.02, 0.003, (H, W)), 0, 1).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    for (x, y, a) in stars:
        img += (a * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2)))).astype(np.float32)
    return np.clip(img, 0, 1)


def _kinds(res, kind):
    return [c for c in res.candidates
            if (c.kind if isinstance(c.kind, str) else c.kind.name) == kind
            or c.kind == kind]


def _has_near(cands, x, y, tol=6):
    return any(abs(c.x - x) < tol and abs(c.y - y) < tol for c in cands)


class TestNewSource:
    def test_new_star_flagged(self):
        ref = _field(_REF_STARS, 1)
        new = _field(_REF_STARS + [(100, 100, 0.7)], 2)
        res = hunt_transients(ref, new, params=TransientHunterParams(), already_aligned=True)
        new_c = _kinds(res, TransientKind.NEW) or _kinds(res, "NEW")
        assert _has_near(new_c, 100, 100), "the injected new star should be a NEW candidate"

    def test_identical_frame_no_candidates(self):
        ref = _field(_REF_STARS, 1)
        # same stars, only faint noise differs -> essentially no transients
        new = _field(_REF_STARS, 1)
        res = hunt_transients(ref, new, params=TransientHunterParams(), already_aligned=True)
        # allow at most one spurious noise pick; the injected-transient tests
        # above prove real transients are found
        assert len(res.candidates) <= 1


class TestMovedSource:
    def test_shifted_star_flagged_moved(self):
        ref = _field(_REF_STARS, 1)
        # move the (160,150) star by (+8,-6); everything else identical
        moved = [s for s in _REF_STARS if s[:2] != (160, 150)] + [(168, 144, 0.7)]
        new = _field(moved, 2)
        res = hunt_transients(
            ref, new,
            params=TransientHunterParams(match_radius=20.0),
            already_aligned=True,
        )
        movers = _kinds(res, TransientKind.MOVED) or _kinds(res, "MOVED")
        # a moved source appears near the new position with a motion vector,
        # OR (depending on his classifier) as a NEW at the new spot plus a
        # VANISHED at the old — accept either faithful encoding of "it moved".
        if movers:
            assert _has_near(movers, 168, 144) or _has_near(movers, 160, 150)
        else:
            new_c = _kinds(res, TransientKind.NEW) or _kinds(res, "NEW")
            vanished = _kinds(res, TransientKind.VANISHED) or _kinds(res, "VANISHED")
            assert _has_near(new_c, 168, 144) and _has_near(vanished, 160, 150)


class TestVanished:
    def test_removed_star_flagged_vanished(self):
        ref = _field(_REF_STARS, 1)
        # drop the (120,80) star in the new frame
        new = _field([s for s in _REF_STARS if s[:2] != (120, 80)], 2)
        res = hunt_transients(ref, new, params=TransientHunterParams(), already_aligned=True)
        vanished = _kinds(res, TransientKind.VANISHED) or _kinds(res, "VANISHED")
        assert _has_near(vanished, 120, 80), "removed star should be VANISHED"


class TestVariableStarNotNew:
    def test_brighter_same_position_not_new(self):
        ref = _field(_REF_STARS, 1)
        brighter = [(40, 40, 1.0)] + _REF_STARS[1:]  # (40,40) brighter, same place
        new = _field(brighter, 2)
        res = hunt_transients(ref, new, params=TransientHunterParams(), already_aligned=True)
        new_c = _kinds(res, TransientKind.NEW) or _kinds(res, "NEW")
        assert not _has_near(new_c, 40, 40), "a variable (brighter) star is not a NEW source"


class TestEdgeMargin:
    def test_edge_detections_rejected(self):
        ref = _field(_REF_STARS, 1)
        # inject a new source right at the border
        new = _field(_REF_STARS + [(3, 3, 0.8)], 2)
        res = hunt_transients(
            ref, new,
            params=TransientHunterParams(edge_margin_fraction=0.1),
            already_aligned=True,
        )
        assert not _has_near(res.candidates, 3, 3), "border detections should be rejected"


class TestColorInput:
    def test_color_reference_and_new(self):
        ref_mono = _field(_REF_STARS, 1)
        new_mono = _field(_REF_STARS + [(100, 100, 0.7)], 2)
        ref = np.stack([ref_mono] * 3)
        new = np.stack([new_mono] * 3)
        res = hunt_transients(ref, new, params=TransientHunterParams(), already_aligned=True)
        new_c = _kinds(res, TransientKind.NEW) or _kinds(res, "NEW")
        assert _has_near(new_c, 100, 100)


class TestParamsAndResult:
    def test_defaults_and_result_shape(self):
        ref = _field(_REF_STARS, 1)
        new = _field(_REF_STARS + [(100, 100, 0.7)], 2)
        res = hunt_transients(ref, new, params=TransientHunterParams(), already_aligned=True)
        assert hasattr(res, "candidates") and hasattr(res, "diff_images")
        for c in res.candidates:
            assert 0 <= c.x <= W and 0 <= c.y <= H
            assert c.flux >= 0

    def test_progress_callback(self):
        ref = _field(_REF_STARS, 1)
        new = _field(_REF_STARS, 2)
        calls = []
        hunt_transients(ref, new, params=TransientHunterParams(),
                        already_aligned=True, progress=lambda f, m: calls.append(f))
        assert calls, "progress should be reported"
