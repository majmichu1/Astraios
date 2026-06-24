"""Astraios performance benchmark harness.

Times the heavy image-processing operations on the active device (GPU or CPU)
so you can produce *real*, reproducible numbers — for tuning, for regression
tracking, and as the Astraios side of an honest Astraios-vs-Siril comparison.

It does NOT fabricate or compare against Siril automatically; it measures
Astraios. To compare with Siril, run the same operations on the same frames in
Siril (see docs/BENCHMARKS.md for a fair methodology) and drop the two timings
side by side.

Usage:
    poetry run python scripts/benchmark.py                  # synthetic, defaults
    poetry run python scripts/benchmark.py --size 3000x4000 --frames 30 --color
    poetry run python scripts/benchmark.py --frames-dir /path/to/fits --runs 5
    poetry run python scripts/benchmark.py --json results.json

Notes:
    * Synthetic frames are fine for *timing* (runtime barely depends on content),
      but for a headline number use your own real subs via --frames-dir.
    * Each op is timed as the median of --runs runs after one warm-up run, with a
      device synchronize around the timed region so GPU numbers are honest.
    * Each op is isolated: if one fails (e.g. missing optional dep), it is marked
      SKIP and the rest still run.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from astraios.core.device_manager import get_device_manager


def _synthetic_frame(h: int, w: int, color: bool, rng: np.random.Generator) -> np.ndarray:
    """A starfield-ish frame: smooth gradient + gaussian stars + read noise."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = 0.05 + 0.03 * (xx / max(w, 1)) + 0.02 * (yy / max(h, 1))
    img = grad.copy()
    n_stars = max(50, (h * w) // 4000)
    sx = rng.integers(0, w, n_stars)
    sy = rng.integers(0, h, n_stars)
    amp = rng.uniform(0.2, 0.9, n_stars).astype(np.float32)
    img[sy, sx] += amp  # delta stars; a light blur below spreads them to PSFs
    img += rng.normal(0, 0.01, (h, w)).astype(np.float32)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    if color:
        # Slightly decorrelated channels so colour ops have real work to do.
        c2 = np.clip(img + rng.normal(0, 0.01, (h, w)).astype(np.float32), 0, 1)
        c3 = np.clip(img + rng.normal(0, 0.01, (h, w)).astype(np.float32), 0, 1)
        return np.stack([img, c2, c3]).astype(np.float32)
    return img


def _shifted(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Integer-roll a frame so registration has a real shift to solve."""
    return np.roll(np.roll(frame, dy, axis=-2), dx, axis=-1)


def _load_frames_dir(path: Path, limit: int) -> list[np.ndarray]:
    from astraios.core.image_io import load_image

    files = sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in {".fits", ".fit", ".fts", ".tif", ".tiff", ".png", ".jpg"}
    )[:limit]
    if not files:
        raise SystemExit(f"No image files found in {path}")
    return [load_image(p).data.astype(np.float32) for p in files]


def _time_op(fn: Callable[[], object], runs: int, dm) -> float:
    """Median wall-clock (seconds) over *runs* runs, after one warm-up."""
    fn()  # warm-up (kernel compile, model load, caches)
    dm.synchronize()
    samples = []
    for _ in range(runs):
        dm.synchronize()
        t0 = time.perf_counter()
        fn()
        dm.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _build_ops(frames: list[np.ndarray], color: bool) -> dict[str, Callable[[], object]]:
    """Map operation name -> zero-arg callable. Imports are local so a missing
    symbol only skips that one op."""
    ops: dict[str, Callable[[], object]] = {}
    one = frames[0]
    n = len(frames)

    def stack(align: bool):
        from astraios.core.image_io import ImageData
        from astraios.core.stacking import (
            IntegrationMethod,
            RejectionMethod,
            StackingParams,
            stack_images,
        )
        imgs = [ImageData(data=f.copy()) for f in frames]
        params = StackingParams(
            rejection=RejectionMethod.SIGMA_CLIP,
            integration=IntegrationMethod.AVERAGE,
        )
        return stack_images(imgs, params=params, align=align)

    ops[f"stack {n}f  (align+reject+integrate)"] = lambda: stack(True)
    ops[f"stack {n}f  (reject+integrate only)"] = lambda: stack(False)

    def drizzle():
        from astraios.core.drizzle import DrizzleParams, drizzle_integrate
        transforms = [np.eye(2, 3, dtype=np.float32) for _ in frames]
        return drizzle_integrate(frames, transforms=transforms, params=DrizzleParams(scale=2))

    ops[f"drizzle x2  {n}f"] = drizzle

    def denoise_wavelet():
        from astraios.core.denoise import DenoiseMethod, DenoiseParams, denoise
        return denoise(one, DenoiseParams(method=DenoiseMethod.WAVELET, strength=0.5))

    ops["denoise (wavelet)  1f"] = denoise_wavelet

    if color:
        def chroma():
            from astraios.core.chroma_denoise import chroma_denoise
            return chroma_denoise(one, strength=1.0)

        ops["chroma denoise  1f"] = chroma

    def deconv():
        from astraios.core.deconvolution import DeconvolutionParams, richardson_lucy
        return richardson_lucy(one, DeconvolutionParams(psf_fwhm=3.0, iterations=30))

    ops["deconvolution (RL x30)  1f"] = deconv

    def wsharpen():
        from astraios.core.wavelets import WaveletParams, wavelet_sharpen
        return wavelet_sharpen(one, WaveletParams(n_scales=4,
                                                  scale_weights=[1.5, 1.2, 1.0, 1.0]))

    ops["wavelet sharpen  1f"] = wsharpen

    def background():
        from astraios.core.background import BackgroundParams, extract_background
        return extract_background(one, BackgroundParams())

    ops["background extract  1f"] = background

    def stretch():
        from astraios.core.stretch import ArcsinhStretchParams, arcsinh_stretch
        return arcsinh_stretch(one, ArcsinhStretchParams())

    ops["arcsinh stretch  1f"] = stretch
    return ops


def main() -> None:
    ap = argparse.ArgumentParser(description="Astraios performance benchmark")
    ap.add_argument("--size", default="2000x3000", help="HxW of synthetic frames")
    ap.add_argument("--frames", type=int, default=20, help="number of frames")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per op (median)")
    ap.add_argument("--color", action="store_true", help="use 3-channel frames")
    ap.add_argument("--frames-dir", type=Path, default=None,
                    help="benchmark on real frames from this directory instead")
    ap.add_argument("--json", type=Path, default=None, help="also write results as JSON")
    args = ap.parse_args()

    dm = get_device_manager()
    import torch

    if args.frames_dir:
        frames = _load_frames_dir(args.frames_dir, args.frames)
        color = frames[0].ndim == 3
        f0 = frames[0]
        h, w = (f0.shape[-2], f0.shape[-1])
    else:
        h, w = (int(x) for x in args.size.lower().split("x"))
        color = args.color
        rng = np.random.default_rng(1234)
        base = _synthetic_frame(h, w, color, rng)
        # Small per-frame shifts so registration is not a no-op.
        frames = [_shifted(base, int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
                  for _ in range(args.frames)]

    megapix = (h * w) / 1e6
    print("=" * 64)
    print("Astraios benchmark")
    print(f"  device      : {dm.info.name}  ({dm.backend.name})")
    if dm.info.vram_total_mb:
        print(f"  VRAM        : {dm.info.vram_total_mb} MB")
    print(f"  torch       : {torch.__version__}")
    print(f"  frames      : {len(frames)} x {'3x' if color else ''}{h}x{w} "
          f"({megapix:.1f} MP each)")
    print(f"  runs/op     : {args.runs} (median reported)")
    print("=" * 64)
    print(f"{'operation':<38}{'median':>12}")
    print("-" * 64)

    results = {}
    for name, fn in _build_ops(frames, color).items():
        try:
            t = _time_op(fn, args.runs, dm)
            results[name] = t
            print(f"{name:<38}{t * 1000:>9.0f} ms")
        except Exception as e:  # noqa: BLE001 - one bad op must not kill the run
            results[name] = None
            print(f"{name:<38}{'SKIP':>12}   ({type(e).__name__}: {e})")
        finally:
            dm.empty_cache()

    print("-" * 64)
    print("Tip: --frames-dir <your subs> for a headline number on real data.")
    print("To compare with Siril, run the same ops on the same frames in Siril")
    print("(see docs/BENCHMARKS.md) and put the two timings side by side.")

    if args.json:
        import json
        payload = {
            "device": dm.info.name,
            "backend": dm.backend.name,
            "vram_mb": dm.info.vram_total_mb,
            "torch": torch.__version__,
            "frames": len(frames),
            "shape": [h, w],
            "color": color,
            "runs": args.runs,
            "timings_ms": {k: (None if v is None else v * 1000) for k, v in results.items()},
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
