#!/usr/bin/env python
"""Ground-truth check of detect -> match -> align -> normalise -> stack.

Run with:  QT_QPA_PLATFORM=offscreen poetry run python scripts/verify_stacking_basics.py
No timing is measured; it can run on a busy machine.

Everything is synthetic with KNOWN answers: star positions, per-frame affine
transforms, sky level, gain, noise, cosmic rays and one satellite trail.
The real pipeline functions are run and their output compared to the truth.
"""
import math
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import logging; logging.getLogger("astraios").setLevel(logging.ERROR)
from astraios.core.image_io import ImageData
from astraios.core.stacking import (
    IntegrationMethod,
    NormalizationMethod,
    RegistrationMode,
    RejectionMethod,
    StackingParams,
    align_frames,
    stack_images,
)
from astraios.core.star_detection import detect_stars

H, W = 600, 800
SIG = 1.5
rng = np.random.default_rng(42)
N_STARS = 220
cat_x = rng.uniform(20, W - 20, N_STARS); cat_y = rng.uniform(20, H - 20, N_STARS)
cat_f = np.clip(rng.pareto(1.6, N_STARS) * 0.04 + 0.02, 0, 0.9)

def render(xs, ys, fs, sky, gain, noise, seed, trail=False, n_cr=25):
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), sky, np.float32)
    img += (xx / W) * 0.01 + (yy / H) * 0.006  # gradient
    for x, y, f in zip(xs, ys, fs):
        x0, y0 = int(x), int(y)
        sl = (slice(max(0, y0 - 7), min(H, y0 + 8)), slice(max(0, x0 - 7), min(W, x0 + 8)))
        g = np.exp(-((yy[sl] - y) ** 2 + (xx[sl] - x) ** 2) / (2 * SIG ** 2))
        img[sl] += f * g
    img *= gain
    img += r.normal(0, noise * gain * np.sqrt(sky / 0.05), img.shape).astype(np.float32)
    cr = (r.integers(0, H, n_cr), r.integers(0, W, n_cr))
    img[cr] = 0.95
    if trail:
        for t in np.linspace(0, 1, 2000):
            x = int(50 + t * (W - 100)); y = int(120 + t * 300)
            img[max(0, y - 1):y + 2, x] = 0.7
    return np.clip(img, 0, 1).astype(np.float32), cr

def affine(dx, dy, deg, scale=1.0):
    th = math.radians(deg); c, s = math.cos(th) * scale, math.sin(th) * scale
    cx, cy = W / 2, H / 2
    # maps reference coords -> frame coords (rotation about centre + shift)
    A = np.array([[c, -s], [s, c]], np.float64)
    t = np.array([cx, cy]) - A @ np.array([cx, cy]) + np.array([dx, dy])
    return A, t

def make_frames(shifts, n=None, seed0=100):
    frames, truths = [], []
    for k, (dx, dy, deg) in enumerate(shifts):
        A, t = affine(dx, dy, deg)
        pts = (A @ np.stack([cat_x, cat_y])).T + t
        keep = (pts[:, 0] > 3) & (pts[:, 0] < W - 3) & (pts[:, 1] > 3) & (pts[:, 1] < H - 3)
        img, _ = render(pts[keep, 0], pts[keep, 1], cat_f[keep], 0.05, 1.0, 0.004, seed0 + k)
        frames.append(ImageData(data=img, header={}))
        truths.append((A, t))
    return frames, truths

def residual_vs_reference(aligned_img, ref_img):
    """Median position residual between stars re-detected in aligned frame and the reference."""
    ref = detect_stars(ref_img, max_stars=150, sigma_threshold=6).positions
    tgt = detect_stars(aligned_img, max_stars=150, sigma_threshold=6).positions
    if len(tgt) < 10:
        return float("nan"), len(tgt)
    d = np.sqrt(((ref[:, None, :] - tgt[None, :, :]) ** 2).sum(-1))
    nn = d.min(axis=1)
    ok = nn < 3.0
    return float(np.median(nn[ok])) if ok.sum() > 10 else float("nan"), int(ok.sum())

def section(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

# ── 1. Detection ───────────────────────────────────────────────────────────
section("1. Star detection on the reference frame (precision / recall vs truth)")
ref_img, _ = render(cat_x, cat_y, cat_f, 0.05, 1.0, 0.004, 7)
for sig in (3.0, 5.0, 8.0):
    sf = detect_stars(ref_img, max_stars=400, sigma_threshold=sig)
    det = sf.positions
    d = np.sqrt(((det[:, None, :] - np.stack([cat_x, cat_y], 1)[None]) ** 2).sum(-1))
    hit = d.min(1) < 1.0
    found = (d.min(0) < 1.0)
    bright = cat_f > 0.06
    print(f"  sigma={sig:4.1f}: detected={len(det):3d} precision={hit.mean():.2f} "
          f"recall(all)={found.mean():.2f} recall(bright)={found[bright].mean():.2f} "
          f"median pos err={np.median(d.min(1)[hit]):.3f}px")

# ── 2. Alignment across modes / shifts / GPU ──────────────────────────────
section("2. Alignment: recovered registration vs ground truth (median star residual, px)")
scenarios = {
    "small dither  (<=8px, 0.1deg)": [(0, 0, 0), (5, -3, 0.1), (-7, 6, -0.05), (3, 8, 0.08)],
    "medium dither (30px, 1deg)":    [(0, 0, 0), (28, -19, 1.0), (-31, 24, -0.7), (15, 33, 0.5)],
    "large dither  (80px, 3deg)":    [(0, 0, 0), (75, -40, 3.0), (-60, 70, -2.5), (85, 20, 1.5)],
}
modes = [RegistrationMode.STAR_1_PASS, RegistrationMode.STAR_2_PASS, RegistrationMode.TRIANGLE]
from astraios.core.device_manager import get_device_manager

has_gpu = get_device_manager().is_gpu
for name, shifts in scenarios.items():
    frames, truths = make_frames(shifts)
    print(f"\n  {name}")
    for mode in modes + ([RegistrationMode.FFT_TRANSLATION] if "small" in name else []):
        for use_gpu in ([False, True] if (has_gpu and mode != RegistrationMode.FFT_TRANSLATION) else [False]):
            p = StackingParams(registration_mode=mode, reference_frame_index=0, use_gpu=use_gpu)
            t0 = time.time()
            try:
                out = align_frames(frames, p)
            except Exception as e:
                print(f"    {mode.name:15s} gpu={use_gpu!s:5s} CRASH {type(e).__name__}: {str(e)[:60]}"); continue
            dt = time.time() - t0
            res = []
            for a in out[1:]:
                r, n = residual_vs_reference(a.data, frames[0].data); res.append((r, n))
            worst = max((r for r, _ in res if not math.isnan(r)), default=float("nan"))
            nnan = sum(1 for r, _ in res if math.isnan(r))
            print(f"    {mode.name:15s} gpu={use_gpu!s:5s} frames={len(out)}/{len(frames)} "
                  f"worst residual={worst:6.3f}px  failed={nnan + (len(frames)-len(out))}  ({dt:.1f}s)")

# ── 3. Stacking on perfectly registered frames ────────────────────────────
section("3. Stacking: normalisation, rejection, integration on registered frames")
N = 8
frames, trails, crs = [], None, []
for k in range(N):
    sky = 0.05 + 0.02 * k          # sky rises through the night
    gain = 1.0 - 0.05 * k          # transparency drops
    img, cr = render(cat_x, cat_y, cat_f, sky, gain, 0.004, 300 + k, trail=(k == 3))
    frames.append(ImageData(data=img, header={})); crs.append(cr)
truth, _ = render(cat_x, cat_y, cat_f, 0.05, 1.0, 0.0, 999, n_cr=0)   # noiseless reference-level frame
blank = (slice(20, 100), slice(600, 780))
trail_px = []
for t in np.linspace(0, 1, 2000):
    x = int(50 + t * (W - 100)); y = int(120 + t * 300); trail_px.append((y, x))
trail_px = list(set(trail_px)); ty = np.array([p[0] for p in trail_px]); tx = np.array([p[1] for p in trail_px])
cr_y = np.concatenate([c[0] for c in crs]); cr_x = np.concatenate([c[1] for c in crs])
print(f"  frames={N}, sky 0.05->{0.05+0.02*(N-1):.2f}, gain 1.0->{1-0.05*(N-1):.2f}, noise 0.004, "
      f"one satellite trail, {len(cr_y)} cosmic rays. Expected stack noise ~{0.004/math.sqrt(N):.4f}")
print(f"  {'normalisation':18s} {'rejection':16s} {'integ':8s} | bg err | star flux err | trail leak | CR leak | noise  | GPU-CPU")
for norm in (NormalizationMethod.NONE, NormalizationMethod.ADDITIVE, NormalizationMethod.MULTIPLICATIVE, NormalizationMethod.ADDITIVE_SCALING):
    for rej in (RejectionMethod.NONE, RejectionMethod.SIGMA_CLIP, RejectionMethod.WINSORIZED_SIGMA, RejectionMethod.LINEAR_FIT, RejectionMethod.PERCENTILE_CLIP, RejectionMethod.ESD, RejectionMethod.MIN_MAX):
        for integ in ((IntegrationMethod.AVERAGE, IntegrationMethod.MEDIAN) if rej == RejectionMethod.SIGMA_CLIP else (IntegrationMethod.AVERAGE,)):
            outs = {}
            for use_gpu in ([True, False] if has_gpu else [False]):
                p = StackingParams(rejection=rej, integration=integ, normalization=norm, use_gpu=use_gpu, kappa_low=3.0, kappa_high=3.0)
                try:
                    outs[use_gpu] = stack_images(frames, p, align=False).image.data
                except Exception as e:
                    outs[use_gpu] = None; print(f"  {norm.name:18s} {rej.name:16s} {integ.name:8s} | CRASH gpu={use_gpu}: {type(e).__name__}: {str(e)[:50]}")
            r = outs.get(False) if outs.get(False) is not None else outs.get(True)
            if r is None: continue
            bg_err = float(np.median(r[blank]) - np.median(truth[blank]))
            # star flux: compare peak of the 30 brightest stars to truth (relative)
            idx = np.argsort(-cat_f)[5:35]
            sy, sx = np.round(cat_y[idx]).astype(int), np.round(cat_x[idx]).astype(int)
            flux_err = float(np.median((r[sy, sx] - np.median(r[blank])) / (truth[sy, sx] - np.median(truth[blank])))) - 1.0
            trail_leak = float(np.median(r[ty, tx] - truth[ty, tx]))
            cr_leak = float(np.median(r[cr_y, cr_x] - truth[cr_y, cr_x]))
            noise = float(np.std(r[blank] - truth[blank]))
            gpu_cpu = float(np.abs(outs[True] - outs[False]).max()) if has_gpu and outs.get(True) is not None else float("nan")
            print(f"  {norm.name:18s} {rej.name:16s} {integ.name:8s} | {bg_err:+.4f} | {flux_err:+6.1%}     | {trail_leak:+.4f}   | {cr_leak:+.4f} | {noise:.4f} | {gpu_cpu:.1e}")
print("\n  Reading: bg err ~0 means normalisation brought every frame to the reference sky; "
      "flux err ~0 means gains were matched; trail/CR leak ~0 means rejection removed them "
      "(NONE should leak ~trail/N); noise ~0.0014 = sqrt(N) gain; GPU-CPU is max abs difference.")
