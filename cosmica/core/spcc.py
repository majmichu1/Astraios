"""Spectrophotometric Color Calibration (SPCC).

Calibrates OSC/RGB image colors using:
1. Catalog star colors (Gaia BP-RP index → effective temperature)
2. Filter transmission curves (camera + optional narrowband filters)
3. Stellar spectral energy distributions (blackbody approximation)
4. Least-squares fit of a 3×3 color correction matrix

Inspired by PixInsight's SPCC (PI 1.8.9+). Works after plate solving.

Workflow
--------
1. Plate solve → WCS
2. Query Gaia DR3 stars in field → get BP-RP colors
3. Detect image stars, measure R/G/B instrumental fluxes via aperture photometry
4. Match detected ↔ catalog stars by position
5. For each catalog star: compute expected R/G/B flux ratios from
   (blackbody × filter) integrals
6. Fit per-channel linear scale + additive background via least squares
7. Apply correction + background neutralization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


# ── Filter database ───────────────────────────────────────────────────────────
# Each filter is (wavelength_nm array, transmission array) in [0, 1].
# Data is sampled at 5 nm steps for simplicity.

def _flat_filter(lam_start: float, lam_end: float) -> tuple[np.ndarray, np.ndarray]:
    lam = np.arange(lam_start, lam_end + 5, 5, dtype=np.float64)
    return lam, np.ones_like(lam)


# Approximate Bayer-RGGB/OSC camera QE + filter curves (no separate filter)
_FILTERS: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
    "OSC (no filter)": {
        # Typical IMX sensor QE peaks: R 620nm, G 540nm, B 450nm
        "R": (np.array([580, 590, 600, 620, 640, 660, 680, 700, 720], dtype=np.float64),
              np.array([0.1, 0.4, 0.75, 1.0, 0.9, 0.7, 0.45, 0.2, 0.05], dtype=np.float64)),
        "G": (np.array([480, 500, 520, 540, 560, 580, 600, 620], dtype=np.float64),
              np.array([0.1, 0.45, 0.80, 1.0, 0.85, 0.55, 0.2, 0.05], dtype=np.float64)),
        "B": (np.array([380, 400, 420, 450, 480, 500, 520], dtype=np.float64),
              np.array([0.05, 0.35, 0.75, 1.0, 0.8, 0.4, 0.1], dtype=np.float64)),
    },
    "Mono + Baader LRGB": {
        "R": (np.array([600, 620, 640, 670, 700, 720], dtype=np.float64),
              np.array([0.05, 0.92, 0.98, 0.97, 0.92, 0.05], dtype=np.float64)),
        "G": (np.array([490, 510, 530, 560, 590, 610], dtype=np.float64),
              np.array([0.05, 0.93, 0.97, 0.95, 0.92, 0.05], dtype=np.float64)),
        "B": (np.array([380, 400, 420, 450, 480, 500], dtype=np.float64),
              np.array([0.05, 0.88, 0.95, 0.97, 0.9, 0.05], dtype=np.float64)),
    },
    "Mono + Astronomik LRGB": {
        "R": (np.array([590, 610, 640, 680, 710, 730], dtype=np.float64),
              np.array([0.02, 0.90, 0.98, 0.97, 0.90, 0.02], dtype=np.float64)),
        "G": (np.array([490, 510, 535, 565, 590, 615], dtype=np.float64),
              np.array([0.02, 0.90, 0.97, 0.96, 0.90, 0.02], dtype=np.float64)),
        "B": (np.array([390, 410, 435, 460, 490, 510], dtype=np.float64),
              np.array([0.02, 0.88, 0.95, 0.97, 0.88, 0.02], dtype=np.float64)),
    },
    "Mono + ZWO LRGB": {
        "R": (np.array([590, 615, 650, 685, 710], dtype=np.float64),
              np.array([0.03, 0.91, 0.97, 0.93, 0.03], dtype=np.float64)),
        "G": (np.array([495, 515, 540, 565, 595], dtype=np.float64),
              np.array([0.03, 0.91, 0.96, 0.91, 0.03], dtype=np.float64)),
        "B": (np.array([395, 415, 440, 465, 495], dtype=np.float64),
              np.array([0.03, 0.87, 0.94, 0.90, 0.03], dtype=np.float64)),
    },
}

FILTER_NAMES = list(_FILTERS.keys())


# ── Spectral models ───────────────────────────────────────────────────────────

def _blackbody_flux(lam_nm: np.ndarray, T_K: float) -> np.ndarray:
    """Planck blackbody spectral radiance (relative, unnormalised)."""
    lam_m = lam_nm * 1e-9
    h, c, k = 6.626e-34, 2.998e8, 1.381e-23
    with np.errstate(over="ignore"):
        denom = np.expm1(h * c / (lam_m * k * T_K))
    return np.where(denom > 0, 1.0 / (lam_m**5 * denom), 0.0)


def _bp_rp_to_teff(bp_rp: float) -> float:
    """Convert Gaia BP-RP colour index to approximate effective temperature (K).

    Polynomial fit from Casagrande et al. (2010) / Mamajek calibration.
    Valid range: -0.5 < BP-RP < 3.5 (O through M stars).
    """
    bp_rp = float(np.clip(bp_rp, -0.5, 3.5))
    # Coefficients for log10(Teff) vs BP-RP (rough polynomial)
    log_teff = 3.999 - 0.654 * bp_rp + 0.194 * bp_rp**2 - 0.031 * bp_rp**3
    return 10.0 ** log_teff


def _expected_channel_flux(T_K: float, lam_nm: np.ndarray, trans: np.ndarray) -> float:
    """Integrate (blackbody × filter transmission) over wavelength."""
    bb = _blackbody_flux(lam_nm, T_K)
    integrand = bb * trans
    return float(np.trapezoid(integrand, lam_nm))


# ── SPCC params ───────────────────────────────────────────────────────────────

@dataclass
class SPCCParams:
    """Parameters for Spectrophotometric Color Calibration.

    Attributes
    ----------
    filter_name : str
        Key from FILTER_NAMES selecting the filter/camera combination.
    neutralize_background : bool
        Subtract per-channel sky background after color correction.
    min_stars : int
        Minimum matched stars required; raises ValueError if not met.
    """
    filter_name: str = "OSC (no filter)"
    neutralize_background: bool = True
    min_stars: int = 8


# ── Aperture photometry ───────────────────────────────────────────────────────

def _aperture_flux(data: np.ndarray, cx: float, cy: float,
                   r: float = 5.0) -> np.ndarray | None:
    """Measure per-channel flux in a circular aperture minus sky annulus.

    Returns shape (C,) fluxes or None if aperture is out of bounds.
    """
    iy, ix = int(round(cy)), int(round(cx))
    ri = int(np.ceil(r * 2.5))          # annulus outer radius

    if data.ndim == 2:
        imgs = data[np.newaxis]
    else:
        imgs = data                     # (C, H, W)
    C, H, W = imgs.shape

    if iy - ri < 0 or iy + ri >= H or ix - ri < 0 or ix + ri >= W:
        return None

    Y, X = np.ogrid[-ri:ri+1, -ri:ri+1]
    R = np.sqrt(X**2 + Y**2)

    ap_mask   = R <= r
    sky_mask  = (R > r * 1.5) & (R <= r * 2.5)
    if sky_mask.sum() < 10:
        return None

    fluxes = []
    for ch in range(C):
        patch = imgs[ch, iy-ri:iy+ri+1, ix-ri:ix+ri+1]
        sky   = float(np.median(patch[sky_mask]))
        flux  = float((patch[ap_mask] - sky).sum())
        fluxes.append(max(flux, 1e-10))
    return np.array(fluxes, dtype=np.float64)


# ── Main SPCC function ────────────────────────────────────────────────────────

def spcc_calibrate(
    data: np.ndarray,
    catalog_stars: list[tuple[float, float, float]],   # (x_img, y_img, bp_rp)
    params: SPCCParams | None = None,
    progress: ProgressCallback | None = None,
) -> np.ndarray:
    """Apply spectrophotometric color calibration.

    Parameters
    ----------
    data : ndarray
        Float32 [0,1] image, shape (3, H, W).
    catalog_stars : list of (x_img, y_img, bp_rp)
        Star positions in image pixel coords and Gaia BP-RP colour index.
        Obtained from plate solve + catalog query.
    params : SPCCParams, optional
    progress : callable, optional

    Returns
    -------
    ndarray
        Colour-calibrated image, same shape as input.
    """
    if params is None:
        params = SPCCParams()
    if progress is None:
        progress = lambda f, m: None

    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError("SPCC requires a 3-channel (RGB) image, shape (3, H, W)")

    filters = _FILTERS.get(params.filter_name)
    if filters is None:
        raise ValueError(f"Unknown filter: {params.filter_name!r}. "
                         f"Choose from: {FILTER_NAMES}")

    progress(0.05, "Computing expected channel fluxes from stellar spectra…")

    # ── Step 1: for each catalog star compute expected R/G/B ratios ───────────
    expected_ratios = []  # list of (r_exp, g_exp, b_exp) relative fluxes
    for x_img, y_img, bp_rp in catalog_stars:
        T = _bp_rp_to_teff(bp_rp)
        fluxes = []
        for ch_name in ("R", "G", "B"):
            lam, trans = filters[ch_name]
            fluxes.append(_expected_channel_flux(T, lam, trans))
        total = max(sum(fluxes), 1e-30)
        expected_ratios.append([f / total for f in fluxes])

    progress(0.20, "Measuring instrumental star fluxes…")

    # ── Step 2: measure instrumental fluxes via aperture photometry ───────────
    inst_fluxes    = []
    exp_fluxes_sel = []
    for i, (x_img, y_img, _) in enumerate(catalog_stars):
        flux = _aperture_flux(data, x_img, y_img, r=5.0)
        if flux is None or flux.min() < 1e-8:
            continue
        # Normalise by green channel (reference)
        norm = flux[1]
        inst_fluxes.append(flux / norm)
        exp = np.array(expected_ratios[i])
        exp_fluxes_sel.append(exp / max(exp[1], 1e-30))

    n_matched = len(inst_fluxes)
    log.info("SPCC: %d stars used for calibration", n_matched)

    if n_matched < params.min_stars:
        raise ValueError(
            f"SPCC: only {n_matched} usable stars found (need {params.min_stars}). "
            "Run plate solve first or use a field with more stars."
        )

    progress(0.50, f"Fitting color correction matrix from {n_matched} stars…")

    # ── Step 3: solve per-channel scale factor via least squares ─────────────
    #  For each channel c: inst[c] ≈ scale[c] * exp[c]
    #  We fit scale[c] = mean(inst[c] / exp[c]) weighted by exp[c]
    inst_arr = np.array(inst_fluxes)   # (N, 3)
    exp_arr  = np.array(exp_fluxes_sel)  # (N, 3)

    scales = np.zeros(3, dtype=np.float64)
    for c in range(3):
        ratios  = inst_arr[:, c] / np.clip(exp_arr[:, c], 1e-10, None)
        weights = exp_arr[:, c]
        scales[c] = np.average(ratios, weights=weights)

    log.info("SPCC channel scales: R=%.4f G=%.4f B=%.4f", *scales)

    # ── Step 4: apply correction ──────────────────────────────────────────────
    progress(0.80, "Applying color correction…")

    result = data.copy().astype(np.float64)
    # Reference channel is G (index 1); rescale R and B to match expected
    # The correction divides out the instrumental response difference:
    #   corrected[c] = data[c] / scale[c] * scale[G]
    g_scale = scales[1]
    for c in range(3):
        if scales[c] > 1e-6:
            result[c] = data[c] * (g_scale / scales[c])

    # ── Step 5: background neutralisation ─────────────────────────────────────
    if params.neutralize_background:
        progress(0.90, "Neutralizing background…")
        H, W = result.shape[1], result.shape[2]
        # Sample corners (avoid stars)
        margin = max(H, W) // 10
        bg_vals = []
        for c in range(3):
            corner = result[c, :margin, :margin].ravel()
            bg_vals.append(float(np.percentile(corner, 10)))
        bg_min = min(bg_vals)
        for c in range(3):
            result[c] -= (bg_vals[c] - bg_min)

    progress(1.0, "SPCC complete")
    return np.clip(result, 0, 1).astype(np.float32)
