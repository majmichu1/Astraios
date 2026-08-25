#!/usr/bin/env python
"""Verify Astraios actually works on Apple's Metal (MPS) backend, and how fast.

Why this exists
---------------
Nobody on this project owns a Mac. The macOS CI job proves the installer runs
and that Apple Silicon *selects* the Metal backend, which is worth proving but
is not the same as the software working there. MPS is a real backend with real
holes: float64 is unsupported, some ops are simply not implemented, and a few
silently take a CPU fallback path. A tool can select MPS, run, and return wrong
pixels, and an import check will never notice.

So this runs the GPU-capable tools twice -- once on whatever backend the device
manager picked, once forced onto the CPU -- and compares the results. That is
the only claim worth making: not "Metal was selected" but "Metal produces the
same image the CPU does".

It doubles as the optimisation instrument. The timings printed here come from
real Apple hardware, which is the only place a number about Apple hardware
means anything. A local workstation cannot answer whether MPS is worth using
for a given tool.

Usage
-----
    python scripts/mps_check.py            # verify parity, print timings
    python scripts/mps_check.py --bench 3  # average over 3 runs

Exits non-zero if any tool errors or diverges, so CI can gate on it. On a
machine with no GPU it reports that and exits 0, since there is nothing to
compare.
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import sys
import time

import numpy as np
import torch

from astraios.core.device_manager import Backend, get_device_manager

# Results differ in the last bits between backends; this is the same tolerance
# the existing GPU/CPU agreement test uses.
TOLERANCE = 2e-3


@contextlib.contextmanager
def forced_cpu():
    """Temporarily force the device manager singleton onto the CPU backend."""
    dm = get_device_manager()
    orig_device, orig_backend = dm._device, dm._backend
    dm._device = torch.device("cpu")
    dm._backend = Backend.CPU
    try:
        yield
    finally:
        dm._device = orig_device
        dm._backend = orig_backend


def _scene(color: bool = True, size: int = 256) -> np.ndarray:
    """A synthetic frame with stars and a gradient, in the project's format."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 0.08 + 0.05 * (xx / size)          # a sky gradient
    base = base + rng.normal(0, 0.004, base.shape)
    for _ in range(24):                        # stars
        cy, cx = rng.uniform(10, size - 10, 2)
        base += 0.6 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 3.0)
    frame = np.clip(base, 0, 1).astype(np.float32)
    if color:
        return np.stack([frame, frame * 0.92, frame * 0.85]).astype(np.float32)
    return frame


def _tools():
    """(name, callable, image) for each GPU-capable tool worth checking."""
    from astraios.core.curves import CurvePoints, CurvesParams, curves_transform
    from astraios.core.histogram_transform import (
        HistogramTransformParams,
        histogram_transform,
    )
    from astraios.core.morphology import MorphologyParams, morphology_transform
    from astraios.core.vignette import VignetteParams, correct_vignette
    from astraios.core.wavelets import WaveletParams, wavelet_sharpen
    from astraios.core.wavescale_hdr import WaveScaleHDRParams, apply_wavescale_hdr

    color = _scene(color=True)
    mono = _scene(color=False)

    curve = CurvesParams(master=CurvePoints(points=[(0.0, 0.0), (0.4, 0.55), (1.0, 1.0)]))

    return [
        ("wavelet_sharpen", lambda d: wavelet_sharpen(d, WaveletParams()), color),
        ("histogram_transform",
         lambda d: histogram_transform(d, HistogramTransformParams(midtone=0.35)), color),
        ("curves_transform", lambda d: curves_transform(d, curve), color),
        ("correct_vignette",
         lambda d: correct_vignette(d, VignetteParams(strength=0.8)), color),
        ("morphology_transform",
         lambda d: morphology_transform(d, MorphologyParams()), mono),
        ("wavescale_hdr",
         lambda d: apply_wavescale_hdr(d, WaveScaleHDRParams()), color),
    ]


def _time(fn, data, repeats: int) -> tuple[np.ndarray, float]:
    dm = get_device_manager()
    result = fn(data.copy())          # warm up; first call pays lazy init
    dm.synchronize()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(data.copy())
        dm.synchronize()
        best = min(best, time.perf_counter() - t0)
    return result, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=int, default=1,
                    help="timed repeats per tool (best of N)")
    args = ap.parse_args()

    dm = get_device_manager()
    mps = getattr(torch.backends, "mps", None)
    print(f"platform     : {platform.system()} {platform.machine()}")
    print(f"torch        : {torch.__version__}")
    print(f"mps available: {bool(mps and mps.is_available())}")
    print(f"backend      : {dm.backend.name}  device={dm.device}")
    print()

    if not dm.is_gpu:
        print("No GPU backend active, so there is nothing to compare against the CPU.")
        print("Run this on Apple Silicon (or CUDA) for it to mean anything.")
        return 0

    gpu_name = dm.backend.name
    print(f"{'tool':<22} {gpu_name:>9} {'cpu':>9} {'speedup':>8}  {'max diff':>10}  result")
    print("-" * 76)

    failures: list[str] = []
    for name, fn, data in _tools():
        try:
            gpu_result, gpu_t = _time(fn, data, args.bench)
            with forced_cpu():
                cpu_result, cpu_t = _time(fn, data, args.bench)
        except Exception as exc:                       # noqa: BLE001
            print(f"{name:<22} {'-':>9} {'-':>9} {'-':>8}  {'-':>10}  ERROR: {exc}")
            failures.append(f"{name}: {exc}")
            continue

        if gpu_result.shape != cpu_result.shape:
            print(f"{name:<22} shape mismatch {gpu_result.shape} vs {cpu_result.shape}")
            failures.append(f"{name}: shape mismatch")
            continue

        diff = float(np.max(np.abs(gpu_result.astype(np.float64)
                                   - cpu_result.astype(np.float64))))
        ok = np.isfinite(gpu_result).all() and diff <= TOLERANCE
        speed = cpu_t / gpu_t if gpu_t > 0 else float("nan")
        verdict = "ok" if ok else "DIVERGED"
        print(f"{name:<22} {gpu_t * 1000:8.1f}ms {cpu_t * 1000:8.1f}ms "
              f"{speed:7.2f}x  {diff:10.2e}  {verdict}")
        if not ok:
            failures.append(f"{name}: max diff {diff:.2e} > {TOLERANCE:.0e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} tool(s) do not agree with the CPU")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"All tools agree with the CPU within {TOLERANCE:.0e}.")
    print("Speedups below 1.00x are tools where the transfer costs more than the "
          "compute at this size; that is data for a decision, not a failure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
