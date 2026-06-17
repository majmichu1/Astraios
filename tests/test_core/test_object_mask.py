"""Tests for the soft elliptical object mask."""

import numpy as np

from astraios.core.object_mask import build_object_mask


def test_none_when_no_objects():
    assert build_object_mask((100, 100), [], plate_scale=2.0) is None
    assert build_object_mask((100, 100), [{"center_x": 50, "center_y": 50}],
                             plate_scale=2.0) is None  # no size


def test_centered_object_is_bright_center_dark_edges():
    h = w = 400
    # 30 arcmin object at 2"/px -> semi-axis = 30*60/2/2 = 450px (fills frame),
    # so use a smaller object to see falloff.
    objs = [{"center_x": w / 2, "center_y": h / 2,
             "major_axis_arcmin": 6.0, "minor_axis_arcmin": 6.0}]
    m = build_object_mask((h, w), objs, plate_scale=2.0)
    assert m is not None
    assert m.shape == (h, w)
    assert m.dtype == np.float32
    assert m.min() >= 0.0 and m.max() <= 1.0
    # Center is fully inside the object.
    assert m[h // 2, w // 2] == 1.0
    # Corners are sky.
    assert m[0, 0] < 0.05
    # Mask is contiguous-ish: more "object" pixels than a single point.
    assert float(np.mean(m > 0.5)) > 0.005


def test_elliptical_shape_follows_axes():
    h = w = 400
    objs = [{"center_x": w / 2, "center_y": h / 2,
             "major_axis_arcmin": 20.0, "minor_axis_arcmin": 5.0}]
    m = build_object_mask((h, w), objs, plate_scale=2.0)
    # Wider along the major (x) axis than the minor (y) axis.
    width_at_center = float(np.sum(m[h // 2, :] > 0.5))
    height_at_center = float(np.sum(m[:, w // 2] > 0.5))
    assert width_at_center > height_at_center


def test_optional_consumers_can_skip():
    # The whole point: None is a valid, safe result downstream code branches on.
    assert build_object_mask((50, 50), [{"major_axis_arcmin": 0.0,
                                         "center_x": 25, "center_y": 25}],
                             plate_scale=2.0) is None
