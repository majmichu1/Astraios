"""The GPU triangle matcher: invariant to shift, rotation and scale, and
honest about a field it cannot match."""

from __future__ import annotations

import math

import numpy as np
import pytest

from astraios.core.gpu_registration import triangle_transform_gpu
from astraios.core.star_detection import Star


def _field(seed, shift, deg, scale, spurious=45, missing=0.15):
    rng = np.random.default_rng(seed)
    n = 150
    x, y = rng.uniform(0, 6000, n), rng.uniform(0, 4000, n)
    fl = rng.pareto(1.5, n)
    th = math.radians(deg)
    c, s = math.cos(th) * scale, math.sin(th) * scale
    X = c * x - s * y + shift
    Y = s * x + c * y - shift * 0.7
    order = np.argsort(-fl)
    ref = [Star(x=float(x[i]), y=float(y[i]), flux=float(fl[i])) for i in order]
    keep = rng.random(n) > missing
    tgt = [
        Star(x=float(X[i] + rng.normal(0, 0.1)), y=float(Y[i] + rng.normal(0, 0.1)), flux=float(fl[i]))
        for i in order if keep[i]
    ]
    tgt += [
        Star(x=float(rng.uniform(0, 6000)), y=float(rng.uniform(0, 4000)), flux=float(rng.pareto(1.5)))
        for _ in range(spurious)
    ]
    tgt.sort(key=lambda st: -st.flux)
    truth_t = np.array([[X[i], Y[i]] for i in order if keep[i]])
    truth_r = np.array([[x[i], y[i]] for i in order if keep[i]])
    return ref, tgt, truth_t, truth_r


@pytest.mark.parametrize("shift,deg,scale", [(5, 0.1, 1.0), (210, 3.0, 1.0), (400, 25.0, 1.02), (900, 90.0, 0.98)])
def test_recovers_the_transform(shift, deg, scale):
    ref, tgt, truth_t, truth_r = _field(1, shift, deg, scale)
    T = triangle_transform_gpu(ref, tgt)
    assert T is not None
    moved = truth_t @ T[:, :2].T + T[:, 2]
    assert np.median(np.hypot(*(moved - truth_r).T)) < 0.1


def test_refuses_a_foreign_field():
    ref, _, _, _ = _field(2, 0, 0, 1.0)
    rng = np.random.default_rng(9)
    foreign = [
        Star(x=float(rng.uniform(0, 6000)), y=float(rng.uniform(0, 4000)), flux=float(rng.pareto(1.5)))
        for _ in range(150)
    ]
    foreign.sort(key=lambda st: -st.flux)
    assert triangle_transform_gpu(ref, foreign) is None


def test_too_few_stars_is_none():
    ref, tgt, _, _ = _field(3, 10, 0.5, 1.0)
    assert triangle_transform_gpu(ref[:3], tgt) is None
