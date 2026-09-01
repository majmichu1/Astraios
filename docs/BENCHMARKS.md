# Benchmarks: Astraios vs Siril

Astraios runs the heavy parts of the processing pipeline on the GPU. Siril's own
engine runs on the CPU (multithreaded; its homepage says all algorithms use all
cores, and GPU offload is an open discussion in its issue tracker as of
2026-09-01), while GPU use in the Siril ecosystem happens inside external tools
it can call, such as GraXpert or RC Astro's paid programs. On a machine with a
capable GPU, that difference shows up most on the operations that are pure pixel
math over the whole frame: registration, integration, drizzle, convolution-based
denoise and sharpening, deconvolution, and stretching.

No Astraios-versus-Siril speedup is published here or anywhere else by this
project, because the two have not yet been timed on the same data with
comparable settings. The quality comparison below has been done that way.

This page is a method for producing **honest, reproducible** numbers on your own
hardware, not a set of marketing claims. Speedups depend heavily on your GPU, your
CPU, frame size, and frame count, so the only number worth publishing is one you
measured yourself, on the same frames, with comparable settings, on both tools.

## What is and is not GPU-accelerated

Being honest about this matters: some steps run on the CPU in Astraios too,
either because the proven library implementation is the right call (sigma-clip
rejection uses astropy; non-local-means uses OpenCV's native code) or because the
work is I/O or network bound (plate solving, catalog lookups). The table below is
where the GPU actually does the work.

| Stage | Astraios | Siril |
|---|---|---|
| Registration (star detect, match, warp) | GPU | CPU |
| Pixel rejection (sigma/winsorized/linear-fit/percentile/min-max) | GPU | CPU |
| Integration | GPU | CPU |
| Drizzle | GPU | CPU |
| Wavelet / chroma / TGV denoise, AI denoise | GPU | CPU (AI via external tools) |
| Deconvolution (Richardson-Lucy) | GPU | CPU |
| Wavelet sharpen, unsharp, convolution | GPU | CPU |
| Background extraction, stretch, curves | GPU | CPU |
| Master calibration combine | CPU (astropy/numpy) | CPU |
| Plate solve / photometric color cal. | CPU (external solver + catalog) | CPU |

Because both tools do calibration combine and plate solving on the CPU, a fair
benchmark should isolate the GPU-bound stages (registration through integration,
plus the post-processing operators) rather than timing a whole end-to-end run
that is dominated by disk reads.

## Reproducing the Astraios numbers

```bash
# Synthetic frames (timing barely depends on content):
poetry run python scripts/benchmark.py --size 3000x4000 --frames 30 --color

# Your own subs (for a headline number):
poetry run python scripts/benchmark.py --frames-dir /path/to/lights --runs 5 --json astraios.json
```

The harness prints your device, frame size and count, and the median wall-clock
of each operation (after a warm-up run, with a device synchronize around the
timed region so the GPU number is honest). Record the device line; it is the
context every number depends on.

## Reproducing the Siril numbers

Siril has a command-line and a script format (`.ssf`). Run the equivalent
registration + stacking on the **same frames** and time the whole call:

```bash
# Example: time a global-star registration + stack of the same lights.
# Adjust the working directory and sequence name to your data.
/usr/bin/time -v siril-cli -d /path/to/lights -s scripts/siril_stack.ssf
```

A minimal `scripts/siril_stack.ssf` is included. It loads the light sequence,
registers it, and stacks with sigma rejection. Match the settings to what
Astraios does (sigma-clip thresholds, no drizzle unless you also drizzle in
Astraios, same output bit depth) or the comparison is meaningless.

## Results template

Fill this in from your two runs. Keep the hardware and settings line; without it
the numbers are not interpretable. Do not publish a row you did not measure.

```
Hardware: <CPU model> / <GPU model, VRAM> / <RAM>
Data:     <N> lights, <W>x<H>, <mono|color>, <bit depth>
Settings: registration=<method>, rejection=sigma-clip k_low=<>, k_high=<>
Software: Astraios <version>, Siril <version>

| Operation                         | Siril (CPU) | Astraios (GPU) | Speedup |
|-----------------------------------|-------------|----------------|---------|
| Register + stack (N frames)       |   <fill>    |     <fill>     |  <fill> |
| Drizzle x2 (N frames)             |   <fill>    |     <fill>     |  <fill> |
| Wavelet denoise (1 frame)         |   <fill>    |     <fill>     |  <fill> |
| Richardson-Lucy deconv (1 frame)  |   <fill>    |     <fill>     |  <fill> |
| Background extraction (1 frame)   |   <fill>    |     <fill>     |  <fill> |
```

Example shape of an Astraios run (synthetic, 1.5 MP color x 8, RTX 5060 Laptop,
machine under other load, so treat as illustrative only, not a published number):

```
stack 8f  (align+reject+integrate)   1898 ms
drizzle x2  8f                        133 ms
denoise (wavelet)  1f                 129 ms
deconvolution (RL x30)  1f             78 ms
background extract  1f                 53 ms
```

## Rules for a fair comparison

- Same frames, same count, same size, same bit depth.
- Comparable quality settings. Do not pit Astraios drizzle against a plain Siril
  stack, or different rejection thresholds, and call it a speedup.
- Warm runs. Time the second run so disk cache and library init do not dominate.
- Report the hardware. A GPU speedup is a statement about a specific GPU vs a
  specific CPU. State both.
- Separate compute from I/O. If most of the wall-clock is reading 60 raw frames
  off a slow disk, you are benchmarking the disk, not either program.
- Do not claim a number you did not measure. A single fabricated "10 minutes vs
  30 seconds" that someone reproduces and disproves costs more trust than it buys.

## Quality comparison with Siril

Measured 2026-08-28 on the maintainer's machine (NVIDIA RTX 5060 Laptop GPU,
7.7 GB VRAM, Fedora-based Linux): 50 JPEG subs of M42 (Sony A7 III,
6000x4000, 240 mm, one night) registered and stacked in both Astraios 0.1.25
(`main`, GPU) and Siril 1.4.4 (`siril-cli`, Lanczos-4 registration,
winsorized sigma clipping 3/3, additive-plus-scaling normalization, no
weighting), with the same reference frame. Siril's output was flipped for
`ROWORDER`, the two stacks were registered to each other, and identical sky
and the same 554 stars were compared.

| Measure | Astraios | Siril 1.4.4 |
|---|---|---|
| Same-star peak height, ratio | 1.005 | 1.000 |
| Same-star FWHM | 3.76 px | 3.79 px |
| Sky noise, 1 px scale (Siril = 1) | 1.10 | 1.00 |
| Sky noise, 4x4 and 8x8 binning | 1.00 | 1.00 |

Stacking Astraios' own aligned frames in Siril gives the same result as
stacking them in Astraios (noise ratio 0.996 to 1.006), so normalization,
rejection and integration are equivalent; the single-pixel difference comes
from Siril's clamped Lanczos-4 resampling versus Astraios' bicubic warp.
These are quality measurements, not timings; no timing comparison is claimed.
