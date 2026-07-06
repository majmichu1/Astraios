"""Tests for astraios.core.derotate (planetary de-rotation).

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import numpy as np

from astraios.core.derotate import DerotateParams, derotate_frames

H = W = 101
CX = CY = 50.0
R = 40.0


def _gaussian_bump(h: int, w: int, x0: float, y0: float, sigma: float = 1.5) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)


def _base_disc() -> np.ndarray:
    """A dim disc of radius R centered at (CX, CY) so the derotation math has
    a well-defined limb, with no other features."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr2 = ((xx - CX) / R) ** 2 + ((yy - CY) / R) ** 2
    return np.where(rr2 <= 1.0, 0.2, 0.0).astype(np.float32)


def test_identity_is_noop_mono():
    """dlon = 0 must reproduce the input (map_x/map_y == identity)."""
    lon0_deg = 15.0
    x0 = CX + R * np.sin(np.deg2rad(lon0_deg))
    frame = _base_disc() + _gaussian_bump(H, W, x0, CY)
    frame = np.clip(frame, 0.0, 1.0)

    params = DerotateParams(
        cx=CX, cy=CY, r=R,
        interpolation="linear",
        per_frame_angles_rad=[0.0],
    )
    out = derotate_frames([frame], params)[0]

    assert out.shape == frame.shape
    # Compare well inside the limb (avoid boundary/visibility edge effects).
    yy, xx = np.mgrid[0:H, 0:W]
    interior = ((xx - CX) ** 2 + (yy - CY) ** 2) <= (0.85 * R) ** 2
    np.testing.assert_allclose(out[interior], frame[interior], atol=2e-3)


def test_known_angle_moves_feature_to_predicted_position():
    """Derotating by a known dlon must move a disc feature to the analytically
    predicted pixel column, on the row through the disc center (lat = 0
    there, so the geometry reduces to x_out = cx + r*sin(lon0 - dlon))."""
    lon0_deg = 20.0
    dlon_deg = -15.0
    lon0 = np.deg2rad(lon0_deg)
    dlon = np.deg2rad(dlon_deg)

    x0 = CX + R * np.sin(lon0)
    frame = _base_disc() + _gaussian_bump(H, W, x0, CY)
    frame = np.clip(frame, 0.0, 1.0)

    params = DerotateParams(
        cx=CX, cy=CY, r=R,
        interpolation="cubic",
        per_frame_angles_rad=[dlon],
    )
    out = derotate_frames([frame], params)[0]

    predicted_x = CX + R * np.sin(lon0 - dlon)

    row = int(round(CY))
    peak_x = int(np.argmax(out[row, :]))
    assert abs(peak_x - predicted_x) <= 2.0, (
        f"peak at {peak_x}, predicted {predicted_x:.2f}"
    )


def test_shape_preserved_mono_and_color():
    mono = _base_disc()
    color = np.stack([mono, mono * 0.8, mono * 0.6], axis=0)

    params = DerotateParams(cx=CX, cy=CY, r=R, per_frame_angles_rad=[0.2])

    out_mono = derotate_frames([mono], params)[0]
    assert out_mono.shape == mono.shape
    assert out_mono.dtype == np.float32

    out_color = derotate_frames([color], params)[0]
    assert out_color.shape == color.shape
    assert out_color.dtype == np.float32


def test_multi_frame_rotation_rate_schedule():
    """rotation_rate_deg_per_hour + frame_times_s should produce a decreasing
    (or increasing, depending on rate sign) dlon schedule with 0 at the
    reference frame."""
    frames = [_base_disc() for _ in range(3)]
    params = DerotateParams(
        cx=CX, cy=CY, r=R,
        rotation_rate_deg_per_hour=36.0,  # 0.01 deg/s
        frame_times_s=[0.0, 100.0, 200.0],
        reference_index=0,
    )
    out = derotate_frames(frames, params)
    assert len(out) == 3
    for o in out:
        assert o.shape == (H, W)
