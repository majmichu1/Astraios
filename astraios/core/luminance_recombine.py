"""Luminance Recombine — replace a color image's luminance with a separately
processed L frame (the classic LRGB / narrowband finishing step).

Ported from Seti Astro Suite Pro (setiastrosuitepro) `luminancerecombine.py`,
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro

After a luminance frame and a color (RGB) frame have been stacked/processed
independently, this replaces the color image's own luminance with the
separately-processed L by per-pixel linear scaling::

    s = new_L / (Y + eps)
    RGB' = RGB * s

where Y is the color image's own weighted luminance. This preserves hue and
chroma exactly and round-trips when ``new_L == Y``. It is a different
algorithm from Astraios's existing ``astraios.core.lrgb.lrgb_combine``, which
works by Lab-*lightness* replacement (blend the Lab L* channel, independent
of the source RGB's own brightness). ``lrgb_combine`` had no equivalent of:
SASpro's per-pixel linear-scale recombine; its selectable luma-weight
profiles (Rec.709/601/2020, equal, max, median, SNR-weighted, and a table of
sensor-specific custom weights); pedestal noise-floor protection; highlight
soft-knee protection; a pre-recombine HSV saturation boost; or pre-recombine
YCbCr chrominance noise reduction. Both modules are kept side by side:
``lrgb_combine`` for perceptual Lab-space blending, and ``recombine_luminance``
here for SASpro's exact per-pixel linear-scale math and settings.

GPU note: the core per-pixel recombine (luminance weighting, pedestal lift,
scale factor, highlight soft-knee, blend) is pure elementwise arithmetic over
every pixel of a potentially huge image — an ideal GPU candidate — so it runs
through `device_manager` on GPU when available, with a numerically identical
numpy CPU fallback. The optional pre-recombine saturation boost and
chrominance-NR blur are comparatively small one-shot preprocessing passes
(only run when the user enables them) that need an HSV/YCbCr round trip;
they stay on CPU via cv2 (already a project dependency, see e.g.
`astraios/core/masks.py`) to keep that conversion exact and simple rather
than reimplementing colorspace math twice.

Images are float32 in [0, 1]; mono is (H, W), color is (C, H, W)
channels-first (SASpro's original works on (H, W, C); the math is
axis-order-independent so only the channel axis moved).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask, apply_mask

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


_LUMA_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_LUMA_REC601 = np.array([0.2990, 0.5870, 0.1140], dtype=np.float32)
_LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593], dtype=np.float32)

# ---- Luma profiles (SASpro's exact table) ----
# Key = the value stored for "luma_method"/preset["mode"].
# weights are length-3 (RGB), assumed linear.
_RAW_LUMA_PROFILES: dict[str, dict] = {
    # --- Standard ---
    "rec709": {"method": "rec709", "weights": _LUMA_REC709, "category": "Standard",
               "description": "Broadband RGB (Rec.709)"},
    "rec601": {"method": "rec601", "weights": _LUMA_REC601, "category": "Standard",
               "description": "Rec.601"},
    "rec2020": {"method": "rec2020", "weights": _LUMA_REC2020, "category": "Standard",
                "description": "Rec.2020"},
    "equal": {"method": "equal", "weights": None, "category": "Standard",
              "description": "Equal RGB"},
    "max": {"method": "max", "weights": None, "category": "Standard",
            "description": "Max (Narrowband mappings)"},
    "median": {"method": "median", "weights": None, "category": "Standard",
               "description": "Median RGB"},
    "snr": {"method": "snr", "weights": None, "category": "Standard",
            "description": "Unequal Noise (SNR)"},

    # --- Sensors ---
    "sensor:Sony IMX571 (ASI2600/QHY268)": {
        "method": "custom", "weights": (0.2944, 0.5021, 0.2035),
        "category": "Sensors/Sony Modern BSI",
        "description": "Sony IMX571 26MP APS-C BSI (STARVIS)",
        "info": "Gold standard APS-C. Excellent balance for broadband.",
    },
    "sensor:Sony IMX533 (ASI533)": {
        "method": "custom", "weights": (0.2910, 0.5072, 0.2018),
        "category": "Sensors/Sony Modern BSI",
        "description": "Sony IMX533 9MP 1\" Square BSI (STARVIS)",
        "info": "Popular square format. Very low noise.",
    },
    "sensor:Sony IMX455 (ASI6200/QHY600)": {
        "weights": (0.2987, 0.5001, 0.2013),
        "description": "Sony IMX455 61MP Full Frame BSI (STARVIS)",
        "info": "Full frame reference sensor.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX294 (ASI294)": {
        "weights": (0.3068, 0.5008, 0.1925),
        "description": "Sony IMX294 11.7MP 4/3\" BSI",
        "info": "High sensitivity 4/3 format.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX183 (ASI183)": {
        "weights": (0.2967, 0.4983, 0.2050),
        "description": "Sony IMX183 20MP 1\" BSI",
        "info": "High resolution 1-inch sensor.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX178 (ASI178)": {
        "weights": (0.2346, 0.5206, 0.2448),
        "description": "Sony IMX178 6.4MP 1/1.8\" BSI",
        "info": "High resolution entry-level sensor.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX224 (ASI224)": {
        "weights": (0.3402, 0.4765, 0.1833),
        "description": "Sony IMX224 1.27MP 1/3\" BSI",
        "info": "Classic planetary sensor. High Red response.",
        "category": "Sony / Modern BSI",
    },
    "sensor:Sony IMX585 (ASI585) - STARVIS 2": {
        "weights": (0.3431, 0.4822, 0.1747),
        "description": "Sony IMX585 8.3MP 1/1.2\" BSI (STARVIS 2)",
        "info": "NIR optimized. Excellent for H-Alpha/Narrowband.",
        "category": "Sony / STARVIS 2",
    },
    "sensor:Sony IMX662 (ASI662) - STARVIS 2": {
        "weights": (0.3430, 0.4821, 0.1749),
        "description": "Sony IMX662 2.1MP 1/2.8\" BSI (STARVIS 2)",
        "info": "Planetary/Guiding. High Red/NIR sensitivity.",
        "category": "Sony / STARVIS 2",
    },
    "sensor:Sony IMX678/715 - STARVIS 2": {
        "weights": (0.3426, 0.4825, 0.1750),
        "description": "Sony IMX678/715 BSI (STARVIS 2)",
        "info": "High resolution planetary/security sensors.",
        "category": "Sony / STARVIS 2",
    },
    "sensor:Panasonic MN34230 (ASI1600/QHY163)": {
        "weights": (0.2650, 0.5250, 0.2100),
        "description": "Panasonic MN34230 4/3\" CMOS",
        "info": "Classic Mono/OSC sensor. Optimized weights.",
        "category": "Panasonic",
    },
    "sensor:Canon EOS (Modern - 60D/6D/R)": {
        "weights": (0.2550, 0.5250, 0.2200),
        "description": "Canon CMOS Profile (Modern)",
        "info": "Balanced profile for most Canon EOS cameras (60D, 6D, 5D, R-series).",
        "category": "Canon",
    },
    "sensor:Canon EOS (Legacy - 300D/40D)": {
        "weights": (0.2400, 0.5400, 0.2200),
        "description": "Canon CMOS Profile (Legacy)",
        "info": "For older Canon models (Digic 2/3 era).",
        "category": "Canon",
    },
    "sensor:Nikon DSLR (Modern - D5300/D850)": {
        "weights": (0.2600, 0.5100, 0.2300),
        "description": "Nikon CMOS Profile (Modern)",
        "info": "Balanced profile for Nikon Expeed 4+ cameras.",
        "category": "Nikon",
    },
    "sensor:ZWO Seestar S50": {
        "weights": (0.3333, 0.4866, 0.1801),
        "description": "ZWO Seestar S50 (IMX462)",
        "info": "Specific profile for Seestar S50 smart telescope.",
        "category": "Smart Telescopes",
    },
    "sensor:ZWO Seestar S30": {
        "weights": (0.2928, 0.5053, 0.2019),
        "description": "ZWO Seestar S30",
        "info": "Specific profile for Seestar S30 smart telescope.",
        "category": "Smart Telescopes",
    },
}


def _build_luma_profiles() -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for key, raw in _RAW_LUMA_PROFILES.items():
        w = raw.get("weights")
        if w is not None:
            w = np.asarray(w, dtype=np.float32)
        profiles[key] = {**raw, "weights": w, "method": raw.get("method", "custom")}
    return profiles


LUMA_PROFILES: dict[str, dict] = _build_luma_profiles()

_ALIASES = {
    "rec.709": "rec709",
    "rec-709": "rec709",
    "rgb": "rec709",
    "k": "rec709",
    "rec.601": "rec601",
    "rec-601": "rec601",
    "rec.2020": "rec2020",
    "rec-2020": "rec2020",
    "nb_max": "max",
    "narrowband": "max",
    "snr_unequal": "snr",
    "unequal_noise": "snr",
}


def resolve_luma_profile_weights(
    mode: str | None,
) -> tuple[str, np.ndarray | None, str | None]:
    """Resolve a luma-method key into ``(resolved_method, weights, profile_name)``.

    Standard modes return ``(mode, None or standard weights, None)``.
    Sensor profiles return ``("rec709", weights, <profile display name>)`` —
    matching SASpro's original convention of routing sensor profiles through
    the "custom weights" path while reporting the sensor name separately.
    """
    if mode is None:
        mode = "rec709"
    key = str(mode).strip()
    key = _ALIASES.get(key.lower(), key)

    prof = LUMA_PROFILES.get(key)
    if not prof:
        return ("rec709", _LUMA_REC709, None)

    w = prof.get("weights", None)

    if key.startswith("sensor:"):
        profile_name = key.split("sensor:", 1)[1].strip()
        return ("rec709", w, profile_name)

    return (key, w, None)


@dataclass
class LuminanceRecombineParams:
    """Parameters for luminance recombine (LRGB / narrowband finishing step).

    Attributes
    ----------
    luma_method : str
        Key into `LUMA_PROFILES` used to derive luminance from the color
        image (when needed, see `compute_luminance`) and to weight the
        recombine itself: ``"rec709"``, ``"rec601"``, ``"rec2020"``,
        ``"equal"``, ``"max"``, ``"median"``, ``"snr"``, or a
        ``"sensor:<name>"`` key for a camera-specific weight table.
    luma_weights : list[float] or None
        Explicit 3-element RGB weight override; wins over `luma_method` when
        given (mirrors SASpro's ``weights=`` caller override).
    noise_sigma : list[float] or None
        Explicit per-channel noise sigma for `luma_method` == "snr". When
        None, sigma is auto-estimated from the luminance source via a
        MAD-based per-channel estimator.
    blend : float
        Mix ratio between the original color image (0.0) and the fully
        recombined result (1.0).
    highlight_soft_knee : float
        0..1 — softens (rolls off) the per-pixel scale factor `s` in
        highlights to reduce clipping/halos when new_L is much brighter than
        the color image's own luminance. 0.0 disables it (pure linear scale).
    pedestal : float
        Noise-floor compression (lift-then-compress) amount applied to both
        images before computing the scale factor, clamped to [0, 0.5].
        Protects near-zero pixels from hue skew; 0.0 disables it.
    eps : float
        Small constant added to the denominator (`Y + eps`) to avoid
        division by zero in near-black pixels.
    saturation_boost : float
        -1..1 — HSV saturation adjustment applied to the color image *before*
        its luminance is measured and replaced (so it changes color
        richness, not luminance structure). 0.0 = no change, 1.0 = double
        saturation, -1.0 = grayscale.
    chrominance_nr_sigma : float
        Gaussian sigma (pixels) for a chrominance-only noise reduction pass
        (blurs Cb/Cr in YCbCr, leaving Y untouched) applied to the color
        image before luminance is measured and replaced. 0.0 disables it.
    """

    luma_method: str = "rec709"
    luma_weights: list[float] | None = None
    noise_sigma: list[float] | None = None
    blend: float = 1.0
    highlight_soft_knee: float = 0.0
    pedestal: float = 0.05
    eps: float = 1e-6
    saturation_boost: float = 0.0
    chrominance_nr_sigma: float = 0.0


def _as_float01(img: np.ndarray) -> np.ndarray:
    """Return float32 image; compress a stray >1.0 range down to ~[0, 1]."""
    a = np.asarray(img)
    if a.dtype != np.float32:
        a = a.astype(np.float32, copy=False)
    if a.size:
        mx = float(a.max())
        if mx > 5.0:
            a = a / mx
    return a


def _estimate_noise_sigma_per_channel(img_chw: np.ndarray) -> np.ndarray:
    """MAD-based per-channel noise sigma estimate, subsampled every 4 px."""
    a = img_chw
    if a.ndim == 2:
        a = a[np.newaxis, ...]
    a = a[:, ::4, ::4].astype(np.float32, copy=False)
    med = np.median(a, axis=(1, 2))
    mad = np.median(np.abs(a - med[:, np.newaxis, np.newaxis]), axis=(1, 2))
    sigma = 1.4826 * mad
    sigma[sigma <= 1e-12] = 1e-12
    return sigma.astype(np.float32)


def compute_luminance(
    img: np.ndarray,
    method: str | None = "rec709",
    weights: np.ndarray | None = None,
    noise_sigma: np.ndarray | None = None,
    normalize_weights: bool = True,
) -> np.ndarray:
    """Compute 2-D linear luminance Y in [0, 1] (float32) from a channels-first image.

    Parameters
    ----------
    img : ndarray
        Mono (H, W) or color (C, H, W) image.
    method : str
        One of "rec709"/"rec601"/"rec2020"/"equal"/"max"/"median"/"snr".
        Ignored when `weights` is given.
    weights : ndarray, optional
        Explicit per-channel weight vector (length C or 3). When given, this
        takes priority over `method`.
    noise_sigma : ndarray, optional
        Required when method == "snr": per-channel noise sigma used to build
        inverse-variance weights.
    normalize_weights : bool
        Normalize `weights` to sum to 1 before use (default True).

    Returns
    -------
    ndarray
        (H, W) float32 luminance in [0, 1].
    """
    f = _as_float01(img)

    if f.ndim == 2:
        return np.ascontiguousarray(f.astype(np.float32, copy=False))
    if f.ndim != 3:
        raise ValueError("compute_luminance: expected 2-D or 3-D array.")

    c = f.shape[0]
    if c == 1:
        return np.ascontiguousarray(f[0].astype(np.float32, copy=False))

    if weights is not None:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim != 1 or w.size not in (c, 3):
            raise ValueError("weights must be 1-D with length equal to channel count or 3.")
        if normalize_weights:
            s = float(w.sum())
            if s != 0.0:
                w = w / s
        use_c = w.size
        lum = np.einsum("c,chw->hw", w, f[:use_c])
    elif method == "equal":
        lum = f[:3].mean(axis=0)
    elif method == "snr":
        if noise_sigma is None:
            raise ValueError("snr method requires noise_sigma per channel.")
        ns = np.asarray(noise_sigma, dtype=np.float32)
        if ns.ndim != 1 or ns.size not in (c, 3):
            raise ValueError("noise_sigma must be 1-D with length equal to channel count or 3.")
        use_c = ns.size
        w = 1.0 / (ns[:use_c] ** 2 + 1e-12)
        w = w / w.sum()
        lum = np.einsum("c,chw->hw", w, f[:use_c])
    elif method == "max":
        lum = f.max(axis=0)
    elif method == "median":
        lum = np.median(f, axis=0)
    elif method == "rec601":
        lum = np.einsum("c,chw->hw", _LUMA_REC601, f[:3])
    elif method == "rec2020":
        lum = np.einsum("c,chw->hw", _LUMA_REC2020, f[:3])
    else:  # default rec709
        lum = np.einsum("c,chw->hw", _LUMA_REC709, f[:3])

    return np.clip(lum.astype(np.float32, copy=False), 0.0, 1.0)


def _recombine_linear_scale_numpy(
    rgb: np.ndarray,
    new_l: np.ndarray,
    weights: np.ndarray,
    eps: float,
    blend: float,
    highlight_soft_knee: float,
    pedestal: float,
) -> np.ndarray:
    _, h, w = rgb.shape
    lum = new_l.astype(np.float32, copy=False)
    if lum.shape != (h, w):
        lum = cv2.resize(lum, (w, h), interpolation=cv2.INTER_LINEAR)

    wgt = np.asarray(weights, dtype=np.float32)
    p = float(np.clip(pedestal, 0.0, 0.5))
    denom = 1.0 + p

    if p > 0.0:
        rgb_p = (rgb + p) / denom
        l_p = (lum + p) / denom
    else:
        rgb_p = rgb
        l_p = lum

    y = wgt[0] * rgb_p[0] + wgt[1] * rgb_p[1] + wgt[2] * rgb_p[2]
    s = l_p / (y + eps)

    if highlight_soft_knee > 0.0:
        k = np.clip(highlight_soft_knee, 0.0, 1.0)
        s = s / (1.0 + k * (s - 1.0))

    out = rgb_p * s[np.newaxis, :, :]

    if p > 0.0:
        out = out * denom - p

    out = np.clip(out, 0.0, 1.0)

    if 0.0 <= blend < 1.0:
        out = rgb * (1.0 - blend) + out * blend

    return out.astype(np.float32, copy=False)


def _recombine_linear_scale_torch(
    rgb: np.ndarray,
    new_l: np.ndarray,
    weights: np.ndarray,
    eps: float,
    blend: float,
    highlight_soft_knee: float,
    pedestal: float,
    dm,
) -> np.ndarray:
    import torch

    _, h, w = rgb.shape
    lum = new_l.astype(np.float32, copy=False)
    if lum.shape != (h, w):
        lum = cv2.resize(lum, (w, h), interpolation=cv2.INTER_LINEAR)

    t_rgb = dm.from_numpy(np.ascontiguousarray(rgb))
    t_l = dm.from_numpy(np.ascontiguousarray(lum))
    t_w = dm.from_numpy(np.asarray(weights, dtype=np.float32))

    p = float(min(max(pedestal, 0.0), 0.5))
    denom = 1.0 + p

    if p > 0.0:
        rgb_p = (t_rgb + p) / denom
        l_p = (t_l + p) / denom
    else:
        rgb_p = t_rgb
        l_p = t_l

    y = t_w[0] * rgb_p[0] + t_w[1] * rgb_p[1] + t_w[2] * rgb_p[2]
    s = l_p / (y + eps)

    if highlight_soft_knee > 0.0:
        k = min(max(highlight_soft_knee, 0.0), 1.0)
        s = s / (1.0 + k * (s - 1.0))

    out = rgb_p * s.unsqueeze(0)

    if p > 0.0:
        out = out * denom - p

    out = torch.clamp(out, 0.0, 1.0)

    if 0.0 <= blend < 1.0:
        out = t_rgb * (1.0 - blend) + out * blend

    return dm.to_cpu(out).numpy().astype(np.float32, copy=False)


def recombine_luminance_linear_scale(
    target_rgb: np.ndarray,
    new_l: np.ndarray,
    weights: np.ndarray = _LUMA_REC709,
    eps: float = 1e-6,
    blend: float = 1.0,
    highlight_soft_knee: float = 0.0,
    pedestal: float = 0.05,
    use_gpu: bool = True,
) -> np.ndarray:
    """Replace linear luminance Y (w.RGB) with `new_l` by per-pixel scaling.

    ``s = new_l / (Y + eps); RGB' = RGB * s``. Preserves hue/chroma in linear
    space and round-trips when ``new_l == Y``.

    Parameters
    ----------
    target_rgb : ndarray
        (3, H, W) float32 [0, 1] RGB image.
    new_l : ndarray
        (H, W) new luminance to inject. Resized to match `target_rgb` if the
        spatial size differs.
    weights : ndarray
        Length-3 RGB weight vector used to measure the target's own Y.
    eps : float
        Denominator floor.
    blend : float
        Mix with the original target (0..1).
    highlight_soft_knee : float
        0..1 — see `LuminanceRecombineParams`.
    pedestal : float
        Noise-floor compression amount (default 0.05 = 5%). Higher values
        protect more shadow detail from hue skew at the cost of slightly
        reducing contrast in very dark areas. 0.0 disables the lift entirely.
    use_gpu : bool
        Route the elementwise math through `device_manager` when a GPU is
        available (default True); set False to force the CPU path.

    Returns
    -------
    ndarray
        (3, H, W) float32 [0, 1].
    """
    rgb = _as_float01(target_rgb)
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("Recombine Luminance requires an RGB (3, H, W) target image.")

    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (3,):
        raise ValueError("weights must be length-3 for RGB recombine.")

    dm = get_device_manager()
    if use_gpu and dm.is_gpu:
        return _recombine_linear_scale_torch(
            rgb, new_l, w, eps, blend, highlight_soft_knee, pedestal, dm
        )
    return _recombine_linear_scale_numpy(rgb, new_l, w, eps, blend, highlight_soft_knee, pedestal)


def _boost_saturation(rgb_chw: np.ndarray, amount: float) -> np.ndarray:
    """Boost saturation of a (3, H, W) float32 RGB image in [0, 1] via HSV.

    amount=0.0 -> no change, amount=1.0 -> double saturation, amount=-1.0 -> grayscale.
    """
    if abs(amount) < 1e-4:
        return rgb_chw
    hwc = np.transpose(np.clip(rgb_chw, 0.0, 1.0).astype(np.float32), (1, 2, 0))
    bgr = cv2.cvtColor(hwc, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + float(amount)), 0.0, 1.0)
    bgr_out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb_out = np.clip(cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB), 0.0, 1.0)
    return np.transpose(rgb_out, (2, 0, 1)).astype(np.float32, copy=False)


def _chrominance_nr(rgb_chw: np.ndarray, sigma: float) -> np.ndarray:
    """Blur only the Cb/Cr color channels of a (3, H, W) RGB image, preserving luma."""
    if sigma < 0.1:
        return rgb_chw

    f = np.clip(rgb_chw, 0.0, 1.0).astype(np.float32)
    r, g, b = f[0], f[1], f[2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.16875 * r - 0.33126 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.41869 * g - 0.08131 * b + 0.5

    ksize = int(sigma * 6) | 1
    ksize = max(ksize, 3)
    cb_b = cv2.GaussianBlur(cb, (ksize, ksize), sigma)
    cr_b = cv2.GaussianBlur(cr, (ksize, ksize), sigma)

    cb_s = cb_b - 0.5
    cr_s = cr_b - 0.5
    r2 = y + 1.402 * cr_s
    g2 = y - 0.34414 * cb_s - 0.71414 * cr_s
    b2 = y + 1.772 * cb_s

    return np.clip(np.stack([r2, g2, b2], axis=0), 0.0, 1.0).astype(np.float32)


def recombine_luminance(
    color: np.ndarray,
    luma: np.ndarray,
    params: LuminanceRecombineParams | None = None,
    mask: Mask | None = None,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Replace `color`'s luminance with the separately-processed `luma` frame.

    Parameters
    ----------
    color : ndarray
        (3, H, W) float32 [0, 1] RGB target image whose luminance is replaced.
    luma : ndarray
        The luminance source: either a mono (H, W) frame (used directly), or
        an RGB (3, H, W) frame (its luminance is derived per `params`).
        Resized to match `color` if the spatial size differs.
    params : LuminanceRecombineParams, optional
        Defaults to Rec.709 weighting, full blend, 5% pedestal.
    mask : Mask, optional
        Restrict the effect to a region; see `astraios.core.masks`.
    progress : callable, optional
        `progress(fraction, message)` callback.

    Returns
    -------
    ndarray
        (3, H, W) float32 [0, 1].
    """
    if params is None:
        params = LuminanceRecombineParams()

    base = _as_float01(color)
    if base.ndim != 3 or base.shape[0] != 3:
        raise ValueError("Recombine Luminance requires an RGB (3, H, W) target image.")

    progress(0.05, "Resolving luma profile…")
    resolved_method, profile_w, _profile_name = resolve_luma_profile_weights(params.luma_method)

    if params.luma_weights is not None:
        w = np.asarray(params.luma_weights, dtype=np.float32).reshape(-1)
        if w.size != 3:
            raise ValueError("luma_weights must be a 3-element RGB vector")
    elif profile_w is not None:
        w = np.asarray(profile_w, dtype=np.float32).reshape(-1)
        if w.size != 3:
            w = None
    else:
        w = None

    progress(0.15, "Pre-processing color image…")
    rgb_pre = _boost_saturation(base, params.saturation_boost)
    rgb_pre = _chrominance_nr(rgb_pre, params.chrominance_nr_sigma)

    progress(0.35, "Building luminance source…")
    src = _as_float01(luma)
    if src.ndim == 2 or (src.ndim == 3 and src.shape[0] == 1):
        new_l = src if src.ndim == 2 else src[0]
    else:
        ns = None
        if resolved_method == "snr":
            ns = (
                np.asarray(params.noise_sigma, dtype=np.float32).reshape(-1)
                if params.noise_sigma is not None
                else _estimate_noise_sigma_per_channel(src)
            )
        new_l = compute_luminance(src, method=resolved_method, weights=w, noise_sigma=ns)

    if w is not None and w.size == 3:
        recombine_w = w
    elif resolved_method == "rec601":
        recombine_w = _LUMA_REC601
    elif resolved_method == "rec2020":
        recombine_w = _LUMA_REC2020
    else:
        recombine_w = _LUMA_REC709

    progress(0.5, "Recombining luminance…")
    replaced = recombine_luminance_linear_scale(
        rgb_pre,
        new_l,
        weights=recombine_w,
        eps=params.eps,
        blend=params.blend,
        highlight_soft_knee=params.highlight_soft_knee,
        pedestal=params.pedestal,
    )

    progress(0.9, "Blending…")
    result = apply_mask(base, replaced, mask)
    progress(1.0, "Recombine complete")
    return result.astype(np.float32, copy=False)
