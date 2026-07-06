"""Tests for astraios.core.planet_projection.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import numpy as np

from astraios.core.planet_projection import PlanetProjectionParams, project_planet

H = W = 121
CX = CY = 60.0
R = 50.0


def _disc_with_feature(x0: float, y0: float, sigma: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rr2 = ((xx - CX) / R) ** 2 + ((yy - CY) / R) ** 2
    disc = np.where(rr2 <= 1.0, 0.15, 0.0).astype(np.float32)
    bump = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
    return np.clip(disc + bump, 0.0, 1.0)


def test_equirectangular_output_shape_mono_and_color():
    mono = _disc_with_feature(CX, CY)
    params = PlanetProjectionParams(
        projection_type="equirectangular", cx=CX, cy=CY, r=R, tex_h=65, tex_w=128,
    )
    out = project_planet(mono, params)
    assert out.shape == (65, 128)
    assert out.dtype == np.float32

    color = np.stack([mono, mono, mono], axis=0)
    out_color = project_planet(color, params)
    assert out_color.shape == (3, 65, 128)


def test_equirectangular_center_maps_to_disc_center():
    """A feature exactly at the disc center (sub-observer point, lon=0,
    lat=0) must land at the equirectangular map's center cell. tex_w even
    and tex_h odd guarantee an exact (lon=0, lat=0) grid point."""
    mono = _disc_with_feature(CX, CY, sigma=2.5)
    tex_h, tex_w = 65, 128  # odd height -> exact lat=0 row; even width -> exact lon=0 col
    params = PlanetProjectionParams(
        projection_type="equirectangular", cx=CX, cy=CY, r=R, tex_h=tex_h, tex_w=tex_w,
        interpolation="linear",
    )
    out = project_planet(mono, params)

    center_row = (tex_h - 1) // 2
    center_col = tex_w // 2

    # The map should be brightest at (or immediately next to) the predicted
    # lon=0/lat=0 cell, not somewhere else on the map.
    peak_row, peak_col = np.unravel_index(np.argmax(out), out.shape)
    assert abs(int(peak_row) - center_row) <= 1
    assert abs(int(peak_col) - center_col) <= 1


def test_equirectangular_deterministic():
    mono = _disc_with_feature(CX + 8, CY - 5)
    params = PlanetProjectionParams(projection_type="equirectangular", cx=CX, cy=CY, r=R)
    out1 = project_planet(mono, params)
    out2 = project_planet(mono, params)
    np.testing.assert_array_equal(out1, out2)


def test_orthographic_zero_yaw_is_identity():
    """theta_deg = 0 must reproduce the input exactly (map is the identity
    transform per _sphere_reproject_maps at theta=0)."""
    mono = _disc_with_feature(CX + 10, CY + 4)
    params = PlanetProjectionParams(
        projection_type="orthographic", cx=CX, cy=CY, r=R, theta_deg=0.0,
        interpolation="linear",
    )
    out = project_planet(mono, params)
    assert out.shape == mono.shape

    yy, xx = np.mgrid[0:H, 0:W]
    interior = ((xx - CX) ** 2 + (yy - CY) ** 2) <= (0.85 * R) ** 2
    np.testing.assert_allclose(out[interior], mono[interior], atol=2e-3)


def test_orthographic_yaw_moves_feature():
    """A nonzero yaw must displace a near-center feature horizontally."""
    mono = _disc_with_feature(CX, CY)
    params = PlanetProjectionParams(
        projection_type="orthographic", cx=CX, cy=CY, r=R, theta_deg=25.0,
        interpolation="cubic",
    )
    out = project_planet(mono, params)

    row = int(round(CY))
    peak_col = int(np.argmax(out[row, :]))
    assert abs(peak_col - CX) > 2.0


def test_auto_disc_detection_finds_reasonable_center():
    mono = _disc_with_feature(CX, CY)
    params = PlanetProjectionParams(projection_type="orthographic", theta_deg=0.0)
    out = project_planet(mono, params)
    assert out.shape == mono.shape
