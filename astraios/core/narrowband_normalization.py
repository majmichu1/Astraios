"""Narrowband Normalization — normalize Ha/OIII/SII channels relative to each
other before palette combination (SHO/HSO/HOS/HOO mapping).

Ported from Seti Astro Suite Pro (setiastrosuitepro)
`imageops/narrowband_normalization.py`, Copyright Franklin Marek,
GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

This reproduces SASpro's PixelMath-derived normalization exactly: per-channel
blackpoint/median/adev statistics feed a PixInsight-style Midtones Transfer
Function (MTF) rescale of the "other" narrowband channels against a chosen
reference channel per scenario, followed by an optional SCNR-style green-cast
reduction, an optional Lab-lightness replacement pass (CIE L*a*b* built from
the PixelMath script's exact D65-ish matrices), and a shared highlight
reduction / brightness / highlight-recover finishing stage. SASpro's own
credit line for the PixelMath concept and SHO/HOS/HSO/HOO formulas: Bill
Blanshan and Mike Cranfield (cosmicphotons.com).

This port drops SASpro's UI-side manual tiling (`ThreadPoolExecutor` over
1024px tiles) — that tiling only existed to interleave progress-bar updates
and background-thread execution; every tile ran the exact same elementwise
function independently (no cross-tile state), so processing the array in one
vectorized pass is numerically identical.

GPU note: this stays numpy/CPU-only, by design — mirroring
`astraios/core/pedestal.py`'s precedent for algorithms that are already
fully vectorized elementwise numpy (no naive per-pixel Python loops to
migrate). The PixelMath-derived math here also does several full-array
reductions (min/median/mean/adev) and MTF evaluations that, while GPU-portable
in principle, are dominated in practice by a handful of scalar per-channel
statistics rather than a hot elementwise kernel; correctness parity with
SASpro's exact formulas mattered more here than the modest speedup GPU
placement would add for this tool.

Inputs are mono (H, W) float32 arrays in [0, 1] for Ha/OIII/SII. Output is
(3, H, W) channels-first float32 RGB in [0, 1] (SASpro's original produces
(H, W, 3); the math is axis-order-independent, only the channel axis moved).

Known upstream quirk (preserved verbatim): in non-linear mode's Lab-lightness
replacement, SASpro's ``_normalize_hso`` and ``_normalize_hos`` reuse
``_normalize_sho``'s literal channel-slot indices for the "Ha (2)"/"SII (3)"
lightness options instead of adjusting them for their own T0/T1/T2 layout —
this looks like a copy/paste artifact (their "OIII" option is correct in HSO,
but wrong in HOS). It only affects `mode=1` (non-linear) combined with
`lightness` in `{2, 3}` (and 4 for HOS). Preserved as-is for exact behavioral
parity with SASpro; see the inline `NOTE` comments at each occurrence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


def _noop_progress(fraction: float, message: str) -> None:
    pass


class MissingChannelsError(ValueError):
    """Raised when a scenario's required channels are not all provided."""


@dataclass
class NBNParams:
    """Parameters for narrowband channel normalization.

    Attributes
    ----------
    scenario : str
        Palette mapping: "SHO", "HSO", "HOS", or "HOO".
    mode : int
        0 = linear (RGB assembled directly from normalized channels),
        1 = non-linear (Lab lightness replacement pass applied afterward).
    lightness : int
        Which lightness source drives the mode==1 Lab replacement. For HOO:
        0=Off/Original-CIE-lightness-of-RGB is not used (see below), 1=CIE
        lightness of raw RGB, 2=Ha, 3=OIII. For SHO/HSO/HOS: 0=CIE lightness
        via full Lab of the assembled RGB, 1=CIE lightness of raw RGB,
        2=Ha, 3=SII, 4=OIII. (Matches SASpro's combo-box indices exactly.)
    blackpoint : float
        0..1 — where between each channel's min and median the normalization
        blackpoint M sits (0 = min, 1 = median).
    hlrecover : float
        >= 0.25 — highlight recovery scale in the finishing stage.
    hlreduct : float
        >= 0.25 — highlight reduction strength in the finishing stage.
    brightness : float
        >= 0.25 — overall brightness in the finishing stage.
    blendmode : int
        HOO only — 0=Screen-like, 1=Add-like, 2=Linear-Dodge-like blend of
        Ha into the normalized OIII channel.
    hablend : float
        HOO only — 0..1 mix ratio for `blendmode`.
    oiiiboost : float
        HOO OIII normalization boost divisor.
    siiboost : float
        SHO/HSO/HOS SII normalization boost divisor.
    oiiiboost2 : float
        SHO/HSO/HOS OIII normalization boost divisor.
    scnr : bool
        SHO/HSO/HOS only — apply a green-channel SCNR-style cap
        (G = min((R+B)/2, G_original)) to suppress a green color cast.
    """

    scenario: str
    mode: int
    lightness: int
    blackpoint: float
    hlrecover: float
    hlreduct: float
    brightness: float
    blendmode: int = 0
    hablend: float = 0.6
    oiiiboost: float = 1.0
    siiboost: float = 1.0
    oiiiboost2: float = 1.0
    scnr: bool = False


__all__ = ["NBNParams", "MissingChannelsError", "normalize_narrowband"]

_EPS = 1e-12


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _inv01(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 - x


def _rescale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map x from [lo, hi] -> [0, 1] (clipped)."""
    denom = max(float(hi) - float(lo), _EPS)
    return _clip01((x - float(lo)) / denom)


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    """PixInsight Midtones Transfer Function. `m` is the midtone (pivot) value."""
    m = float(np.clip(m, _EPS, 1.0 - _EPS))
    x = _clip01(np.asarray(x, dtype=np.float32))

    num = (m - 1.0) * x
    den = (2.0 * m - 1.0) * x - m

    safe_den = np.where(
        np.abs(den) < _EPS,
        np.where(den >= 0.0, _EPS, -_EPS).astype(np.float32),
        den,
    )
    return _clip01(num / safe_den)


def _adev(x: np.ndarray) -> float:
    """Approximate absolute deviation (PixelMath adev())."""
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))


def _stats_min_med_mean(chs: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(chs)
    mins = np.empty((n,), dtype=np.float32)
    meds = np.empty((n,), dtype=np.float32)
    means = np.empty((n,), dtype=np.float32)
    for i, ch in enumerate(chs):
        mins[i] = float(np.nanmin(ch))
        meds[i] = float(np.nanmedian(ch))
        means[i] = float(np.nanmean(ch))
    return mins, meds, means


def _stats_adev_vec(chs: tuple[np.ndarray, ...]) -> np.ndarray:
    v = np.empty((len(chs),), dtype=np.float32)
    for i, ch in enumerate(chs):
        v[i] = _adev(ch)
    return v


# ---------------- Color space helpers (PixelMath script's exact matrices) ----------------


def _srgb_to_linear(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    return np.where(u > 0.04045, ((u + 0.055) / 1.055) ** 2.4, u / 12.92)


def _linear_to_srgb(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    u = np.clip(u, 0.0, 1.0)
    u = np.where(np.isfinite(u), u, 0.0)
    u = np.maximum(u, 0.0)
    return np.where(u > 0.0031308, 1.055 * (u ** (1.0 / 2.4)) - 0.055, 12.92 * u)


def _rgb_to_xyz_pi(
    r: np.ndarray, g: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = _srgb_to_linear(_clip01(r))
    g1 = _srgb_to_linear(_clip01(g))
    b1 = _srgb_to_linear(_clip01(b))

    x = (r1 * 0.4360747) + (g1 * 0.3850649) + (b1 * 0.1430804)
    y = (r1 * 0.2225045) + (g1 * 0.7168786) + (b1 * 0.0606169)
    z = (r1 * 0.0139322) + (g1 * 0.0971045) + (b1 * 0.7141733)
    return x, y, z


def _xyz_to_lab_pi(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, t ** (1.0 / 3.0), (7.787 * t) + (16.0 / 116.0))

    x1, y1, z1 = f(x), f(y), f(z)
    lightness = 116.0 * y1 - 16.0
    a = 500.0 * (x1 - y1)
    b = 200.0 * (y1 - z1)
    return lightness, a, b


def _xyz_to_rgb_pi(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r2 = (x * 3.1338561) + (y * -1.6168667) + (z * -0.4906146)
    g2 = (x * -0.9787684) + (y * 1.9161415) + (z * 0.0334540)
    b2 = (x * 0.0719453) + (y * -0.2289914) + (z * 1.4052427)

    return (
        _clip01(_linear_to_srgb(r2)),
        _clip01(_linear_to_srgb(g2)),
        _clip01(_linear_to_srgb(b2)),
    )


def _ciel_lightness_from_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    x, y, z = _rgb_to_xyz_pi(r, g, b)
    lightness, _, _ = _xyz_to_lab_pi(x, y, z)
    return lightness / 100.0


def _lab_lightness_replace(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, y2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the script's Lab lightness replacement path.

    Convert RGB -> XYZ -> Lab, replace the Y-like term with `y2` (already in
    the script's ``(L+16)/116`` space), rebuild XYZ using a/b and `y2`
    exactly as the script does, then convert XYZ -> RGB.
    """
    x, y, z = _rgb_to_xyz_pi(r, g, b)
    _lightness, a, b_ = _xyz_to_lab_pi(x, y, z)

    x2 = (a / 500.0) + y2
    z2 = y2 - (b_ / 200.0)

    def finv(t: np.ndarray) -> np.ndarray:
        return np.where(t > 0.008856, t**3, (t - 16.0 / 116.0) / 7.787)

    return _xyz_to_rgb_pi(finv(x2), finv(y2), finv(z2))


# ---------------- Common finishing steps ----------------


def _apply_hl_reduction_and_brightness_and_recover(
    e10: np.ndarray, params: NBNParams
) -> np.ndarray:
    hlr = max(float(params.hlreduct), 0.25)
    br = max(float(params.brightness), 0.25)
    hrec = max(float(params.hlrecover), 0.25)

    m_hlr = float(np.clip(1.0 - (0.5 / hlr), _EPS, 1.0 - _EPS))
    e11 = (_mtf(m_hlr, e10) * e10) + (e10 * _inv01(e10))

    m_b = float(np.clip(0.5 / br, _EPS, 1.0 - _EPS))
    e12 = _mtf(m_b, e11)

    e13 = _rescale(e12, 0.0, hrec)
    return _clip01(e13)


def _compute_m_e0(chs: tuple[np.ndarray, ...], blackpoint: float) -> tuple[np.ndarray, np.ndarray]:
    """M = min(T) + Blackpoint*(med(T)-min(T)); E0 = adev(T)/1.2533 + mean(T) - M."""
    mins, meds, means = _stats_min_med_mean(chs)
    m = mins + float(blackpoint) * (meds - mins)
    adevs = _stats_adev_vec(chs)
    e0 = (adevs / 1.2533) + means - m
    return m.astype(np.float32), e0.astype(np.float32)


def _e1_for(a_ref: float, a_other: float, boost: float) -> float:
    denom = a_ref - 2.0 * a_ref * a_other + a_other
    e1 = (a_ref * (1.0 - a_other)) / max(float(denom), _EPS)
    return e1 / max(float(boost), _EPS)


def _normalize_channel(t: np.ndarray, m_val: float, e1: float) -> np.ndarray:
    e2 = _rescale(t, m_val, 1.0)
    min_t_m = np.minimum(t, m_val)
    e3 = _inv01(_inv01(_mtf(e1, e2)) * _inv01(min_t_m))
    return _clip01(e3)


# ---------------- Scenario cores ----------------
#
# The four functions below are direct, literal translations of SASpro's
# `_normalize_hoo` / `_normalize_sho` / `_normalize_hso` / `_normalize_hos`
# (only the output channel axis moved from last to first). They are kept as
# separate functions rather than merged into one parameterized helper so
# that each scenario's exact, individually-verified channel-slot arithmetic
# — including the upstream HSO/HOS lightness quirk documented in the module
# docstring — is preserved without any risk of "fixing" it via a shared
# abstraction.


def _normalize_hoo(
    ha: np.ndarray, oiii: np.ndarray, params: NBNParams, progress: ProgressCallback
) -> np.ndarray:
    t0, t1 = ha, oiii  # R=Ha, G/B derive from OIII

    progress(0.15, "Computing global stats")
    m, e0 = _compute_m_e0((t0, t1, t1), params.blackpoint)

    inv_m1 = max(float(_inv01(m[1])), _EPS)
    a0 = e0 / inv_m1
    e1 = _e1_for(float(a0[1]), float(a0[0]), params.oiiiboost)

    hb = float(np.clip(params.hablend, 0.0, 1.0))
    inv_hb = 1.0 - hb

    progress(0.35, "Normalizing channels")
    e3 = _normalize_channel(t1, float(m[1]), e1)

    if params.blendmode == 0:
        e4 = (t0 * hb) + (e3 * inv_hb)
    elif params.blendmode == 1:
        e4 = (e3 * hb) + (t1 * inv_hb)
    else:
        e4 = (t0 * hb) + (t1 * inv_hb)

    r, g, b = t0, e4, e3

    if params.mode == 0:
        out = np.stack([r, g, b], axis=0)
    else:
        progress(0.55, "Lab lightness replacement")
        if params.lightness == 0:
            x, y, z = _rgb_to_xyz_pi(r, g, b)
            lightness, _, _ = _xyz_to_lab_pi(x, y, z)
            y2 = (lightness + 16.0) / 116.0
        elif params.lightness == 1:
            ciel = _ciel_lightness_from_rgb(t0, t1, t1)
            y2 = (ciel + 0.16) / 1.16
        elif params.lightness == 2:
            y2 = (t0 + 0.16) / 1.16  # Ha
        else:
            y2 = (t1 + 0.16) / 1.16  # OIII
        r3, g3, b3 = _lab_lightness_replace(r, g, b, y2.astype(np.float32))
        out = np.stack([r3, g3, b3], axis=0)

    progress(0.8, "Finishing (HL reduction / brightness / recover)")
    return _apply_hl_reduction_and_brightness_and_recover(out, params)


def _normalize_sho(
    ha: np.ndarray, oiii: np.ndarray, sii: np.ndarray, params: NBNParams, progress: ProgressCallback
) -> np.ndarray:
    t0, t1, t2 = sii, ha, oiii  # R=SII, G=Ha, B=OIII

    progress(0.15, "Computing global stats")
    m, e0 = _compute_m_e0((t0, t1, t2), params.blackpoint)

    inv_m0 = max(float(_inv01(m[0])), _EPS)
    a = e0 / inv_m0
    e1_sii = _e1_for(float(a[0]), float(a[1]), params.siiboost)

    inv_m2 = max(float(_inv01(m[2])), _EPS)
    a = e0 / inv_m2
    e1_oiii = _e1_for(float(a[2]), float(a[1]), params.oiiiboost2)

    progress(0.35, "Normalizing channels")
    e3 = _normalize_channel(t0, float(m[0]), e1_sii)
    e6 = _normalize_channel(t2, float(m[2]), e1_oiii)

    r = e3
    g = t1 if not params.scnr else np.minimum((r + e6) * 0.5, t1)
    b = e6

    if params.mode == 0:
        out = np.stack([r, g, b], axis=0)
    else:
        progress(0.55, "Lab lightness replacement")
        if params.lightness == 0:
            x, y, z = _rgb_to_xyz_pi(r, g, b)
            lightness, _, _ = _xyz_to_lab_pi(x, y, z)
            y2 = (lightness + 16.0) / 116.0
        elif params.lightness == 1:
            ciel = _ciel_lightness_from_rgb(t0, t1, t2)
            y2 = (ciel + 0.16) / 1.16
        elif params.lightness == 2:
            y2 = (t1 + 0.16) / 1.16  # Ha
        elif params.lightness == 3:
            y2 = (t0 + 0.16) / 1.16  # SII
        else:
            y2 = (t2 + 0.16) / 1.16  # OIII
        r3, g3, b3 = _lab_lightness_replace(r, g, b, y2.astype(np.float32))
        out = np.stack([r3, g3, b3], axis=0)

    progress(0.8, "Finishing (HL reduction / brightness / recover)")
    return _apply_hl_reduction_and_brightness_and_recover(out, params)


def _normalize_hso(
    ha: np.ndarray, oiii: np.ndarray, sii: np.ndarray, params: NBNParams, progress: ProgressCallback
) -> np.ndarray:
    t0, t1, t2 = ha, sii, oiii  # R=Ha, G=SII, B=OIII

    progress(0.15, "Computing global stats")
    m, e0 = _compute_m_e0((t0, t1, t2), params.blackpoint)

    inv_m1 = max(float(_inv01(m[1])), _EPS)
    a = e0 / inv_m1
    e1_sii = _e1_for(float(a[1]), float(a[0]), params.siiboost)

    inv_m2 = max(float(_inv01(m[2])), _EPS)
    a = e0 / inv_m2
    e1_oiii = _e1_for(float(a[2]), float(a[0]), params.oiiiboost2)

    progress(0.35, "Normalizing channels")
    e3 = _normalize_channel(t1, float(m[1]), e1_sii)
    e6 = _normalize_channel(t2, float(m[2]), e1_oiii)

    r = t0
    g = e3 if not params.scnr else np.minimum((r + e6) * 0.5, e3)
    b = e6

    if params.mode == 0:
        out = np.stack([r, g, b], axis=0)
    else:
        progress(0.55, "Lab lightness replacement")
        if params.lightness == 0:
            x, y, z = _rgb_to_xyz_pi(r, g, b)
            lightness, _, _ = _xyz_to_lab_pi(x, y, z)
            y2 = (lightness + 16.0) / 116.0
        elif params.lightness == 1:
            ciel = _ciel_lightness_from_rgb(t0, t1, t2)
            y2 = (ciel + 0.16) / 1.16
        elif params.lightness == 2:
            # NOTE (preserved SASpro quirk): labeled "Ha" in the UI, but the
            # original reuses `_normalize_sho`'s slot (t1) instead of this
            # scenario's own Ha slot (t0). See module docstring.
            y2 = (t1 + 0.16) / 1.16
        elif params.lightness == 3:
            # NOTE: labeled "SII" but uses t0 (Ha) verbatim from SASpro.
            y2 = (t0 + 0.16) / 1.16
        else:
            y2 = (t2 + 0.16) / 1.16  # OIII (t2 is genuinely OIII here — correct)
        r3, g3, b3 = _lab_lightness_replace(r, g, b, y2.astype(np.float32))
        out = np.stack([r3, g3, b3], axis=0)

    progress(0.8, "Finishing (HL reduction / brightness / recover)")
    return _apply_hl_reduction_and_brightness_and_recover(out, params)


def _normalize_hos(
    ha: np.ndarray, oiii: np.ndarray, sii: np.ndarray, params: NBNParams, progress: ProgressCallback
) -> np.ndarray:
    t0, t1, t2 = ha, oiii, sii  # R=Ha, G=OIII, B=SII

    progress(0.15, "Computing global stats")
    m, e0 = _compute_m_e0((t0, t1, t2), params.blackpoint)

    inv_m1 = max(float(_inv01(m[1])), _EPS)
    a = e0 / inv_m1
    e1_oiii = _e1_for(float(a[1]), float(a[0]), params.oiiiboost2)

    inv_m2 = max(float(_inv01(m[2])), _EPS)
    a = e0 / inv_m2
    e1_sii = _e1_for(float(a[2]), float(a[0]), params.siiboost)

    progress(0.35, "Normalizing channels")
    e3 = _normalize_channel(t1, float(m[1]), e1_oiii)
    e6 = _normalize_channel(t2, float(m[2]), e1_sii)

    r = t0
    g = e3 if not params.scnr else np.minimum((r + e6) * 0.5, e3)
    b = e6

    if params.mode == 0:
        out = np.stack([r, g, b], axis=0)
    else:
        progress(0.55, "Lab lightness replacement")
        if params.lightness == 0:
            x, y, z = _rgb_to_xyz_pi(r, g, b)
            lightness, _, _ = _xyz_to_lab_pi(x, y, z)
            y2 = (lightness + 16.0) / 116.0
        elif params.lightness == 1:
            ciel = _ciel_lightness_from_rgb(t0, t1, t2)
            y2 = (ciel + 0.16) / 1.16
        elif params.lightness == 2:
            # NOTE (preserved SASpro quirk, see module docstring): labeled
            # "Ha" but uses t1 (OIII) verbatim from SASpro's `_normalize_hos`.
            y2 = (t1 + 0.16) / 1.16
        elif params.lightness == 3:
            # NOTE: labeled "SII" but uses t0 (Ha) verbatim from SASpro.
            y2 = (t0 + 0.16) / 1.16
        else:
            # NOTE: labeled "OIII" but uses t2 (SII) verbatim from SASpro.
            y2 = (t2 + 0.16) / 1.16
        r3, g3, b3 = _lab_lightness_replace(r, g, b, y2.astype(np.float32))
        out = np.stack([r3, g3, b3], axis=0)

    progress(0.8, "Finishing (HL reduction / brightness / recover)")
    return _apply_hl_reduction_and_brightness_and_recover(out, params)


def normalize_narrowband(
    ha: np.ndarray | None,
    oiii: np.ndarray | None,
    sii: np.ndarray | None,
    params: NBNParams,
    progress: ProgressCallback = _noop_progress,
) -> np.ndarray:
    """Normalize Ha/OIII/SII channels relative to each other and map to RGB.

    Parameters
    ----------
    ha, oiii, sii : ndarray or None
        Mono (H, W) float32 [0, 1] channels. `sii` is ignored (may be None)
        for scenario "HOO"; all three are required otherwise.
    params : NBNParams
        Scenario and normalization/finishing settings.
    progress : callable, optional
        `progress(fraction, message)` callback.

    Returns
    -------
    ndarray
        (3, H, W) float32 RGB in [0, 1].

    Raises
    ------
    MissingChannelsError
        If a required channel for the scenario is None.
    ValueError
        If channel shapes disagree, or the scenario is unrecognized.
    """
    scen = (params.scenario or "").split()[0].strip().upper()
    progress(0.0, f"Starting {scen}")

    if scen == "HOO":
        if ha is None or oiii is None:
            raise MissingChannelsError("HOO requires Ha and OIII.")
        if ha.shape != oiii.shape:
            raise ValueError(f"Channel shape mismatch: Ha={ha.shape} OIII={oiii.shape}")
        out = _normalize_hoo(
            ha.astype(np.float32, copy=False), oiii.astype(np.float32, copy=False), params, progress
        )
    elif scen in ("SHO", "HSO", "HOS"):
        missing = [n for n, v in (("Ha", ha), ("OIII", oiii), ("SII", sii)) if v is None]
        if missing:
            raise MissingChannelsError(f"{scen} requires " + ", ".join(missing) + ".")
        assert ha is not None and oiii is not None and sii is not None  # narrowed by check above
        shapes = {ha.shape, oiii.shape, sii.shape}
        if len(shapes) > 1:
            raise ValueError(
                f"Channel shape mismatch: Ha={ha.shape} OIII={oiii.shape} SII={sii.shape}"
            )
        ha32 = ha.astype(np.float32, copy=False)
        oiii32 = oiii.astype(np.float32, copy=False)
        sii32 = sii.astype(np.float32, copy=False)
        scenario_fn = {"SHO": _normalize_sho, "HSO": _normalize_hso, "HOS": _normalize_hos}[scen]
        out = scenario_fn(ha32, oiii32, sii32, params, progress)
    else:
        raise ValueError(f"Unknown narrowband normalization scenario: {params.scenario!r}")

    progress(1.0, "Done")
    return _clip01(out).astype(np.float32, copy=False)
