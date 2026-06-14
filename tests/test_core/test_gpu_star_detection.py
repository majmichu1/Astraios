"""Tests for GPU star detection — especially saturated stars.

Regression: detect_stars_gpu used max-pool local maxima, which flags every
pixel of a saturated flat-top plateau as a maximum -> a single bright star was
reported as hundreds of detections at wrong positions. Now uses connected-
component centroids (one sub-pixel star per blob), matching the CPU detector.
"""

import numpy as np
import pytest
import torch

from cosmica.core.device_manager import get_device_manager
from cosmica.core.gpu_stars import detect_stars_gpu
from cosmica.core.star_detection import detect_stars


def _field(positions, sigma=2.0, amp=0.8, saturate=False, h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 0.04, np.float32) + rng.normal(0, 0.004, (h, w)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in positions:
        img += (amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))).astype(np.float32)
    img = np.clip(img, 0, 1)
    if saturate:
        img = np.clip(img * 2.5, 0, 1)  # flat-topped (saturated) cores
    return img.astype(np.float32)


def _gpu(img, **kw):
    t = torch.from_numpy(img).to(get_device_manager().device)
    return detect_stars_gpu(t, threshold_sigma=5.0, max_stars=200, **kw)


def _matched(det, truth, tol=2.5):
    return sum(1 for cy, cx in truth if any(abs(s.x - cx) < tol and abs(s.y - cy) < tol for s in det))


def _dups(det, tol=4):
    return sum(1 for i, a in enumerate(det) for b in det[i + 1:]
               if abs(a.x - b.x) < tol and abs(a.y - b.y) < tol)


class TestGPUStarDetection:
    def test_saturated_stars_not_overcounted(self):
        pos = [(60, 60), (60, 180), (180, 60), (180, 180)]
        det = _gpu(_field(pos, sigma=3.0, saturate=True))
        assert len(det) == len(pos)            # not hundreds
        assert _dups(det) == 0
        assert _matched(det, pos) == len(pos)  # at the true centres

    def test_clean_field_subpixel(self):
        rng = np.random.default_rng(1)
        pos = []
        while len(pos) < 10:
            p = (int(rng.integers(20, 236)), int(rng.integers(20, 236)))
            if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 25 for q in pos):
                pos.append(p)
        img = _field(pos, sigma=2.0)
        det = _gpu(img)
        assert _matched(det, pos) == len(pos)
        assert _dups(det) == 0

    def test_agrees_with_cpu_on_count(self):
        rng = np.random.default_rng(3)
        pos = []
        while len(pos) < 15:
            p = (int(rng.integers(20, 236)), int(rng.integers(20, 236)))
            if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 22 for q in pos):
                pos.append(p)
        img = _field(pos, sigma=2.0)
        cpu = list(detect_stars(img, max_stars=200, sigma_threshold=5.0).stars)
        gpu = _gpu(img)
        # Both should find all 15 (well-separated, unsaturated) with no duplicates.
        assert _matched(cpu, pos) == 15
        assert _matched(gpu, pos) == 15
        assert _dups(gpu) == 0
