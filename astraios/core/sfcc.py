"""Spectral Flux Color Calibration (SFCC).

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro), Copyright
Franklin Marek, GPL-3.0-or-later. https://github.com/setiastro/setiastrosuitepro

SFCC is SPCC's (:mod:`astraios.core.spcc`) physically-detailed sibling. Where
SPCC looks up one combined "filter+camera" response curve per channel, SFCC
builds the system response as an explicit product of *separate* factors, the
same way SASpro's ``sfcc.py`` does it (see its ``run_spcc()``/``_gaia_integrals
_for_source_ids()``)::

    T_sys_c(lambda) = T_filter_c(lambda) x QE_sensor(lambda) x T_LP1(lambda) x T_LP2(lambda)
    S_expected_c    = integral of flux_star(lambda) x T_sys_c(lambda) d(lambda)

for each channel c in {R, G, B}. Comparing ``S_expected`` to the instrumental
flux actually measured on each catalog star gives a per-channel color
correction, same as SPCC, but now the filter transmission and sensor QE are
independent, swappable inputs instead of one baked-in "OSC/LRGB" lookup —
you can mix e.g. an Antlia-like R/G/B filter set with a KAF-like CCD QE
curve, or add a light-pollution / UV-IR-cut filter on top.

What was ported faithfully vs. reduced
---------------------------------------
* **Faithful**: the integration math itself — ``_integrate_flux_times_T`` /
  ``T_sys = T_filter * QE * LP`` / ``S = integral(flux * T_sys) dlambda`` — is a
  line-for-line port of SASpro's construction (see :func:`build_system_response`
  and :func:`expected_channel_ratios` below, and the module comments pointing
  at the exact source lines).
* **Reused, not reimplemented**: the stellar flux model
  (:func:`astraios.core.spcc._blackbody_flux`, BP-RP -> Teff via
  :func:`astraios.core.spcc._bp_rp_to_teff`), the aperture photometry
  (:func:`astraios.core.spcc._aperture_flux`), and the final per-channel
  scale application + background neutralization are the *same* functions/
  approach SPCC already uses in this codebase. SFCC's whole point is the
  filter x QE integration; duplicating SPCC's already-working machinery
  around it would just be two copies of the same bug surface.
* **Reduced, documented honestly**: SASpro's real SFCC loads its filter
  transmission curves, sensor QE curves, and ~157 Pickles stellar SED
  templates from a bundled 4 MB FITS library (``SASP_data.fits``, ~176
  filter curves digitized from vendor datasheets, 16 sensor QE curves, plus
  Gaia XP spectra downloaded live per star). That library is a large,
  separately-licensed asset we do not have redistribution rights to bundle
  here, so it is **not** shipped. Instead:
    - The stellar spectrum is a blackbody at the Gaia BP-RP-derived Teff
      (:func:`astraios.core.spcc._bp_rp_to_teff`), exactly like SPCC —
      not a Pickles/Gaia-XP spectrum. This is a documented simplification,
      not a silent one: it means SFCC's flux model has the same fidelity
      as SPCC's; only the filter x QE *optical* model is more detailed.
    - A small set of *representative, approximate* filter transmission and
      sensor QE curves is embedded below (:data:`FILTER_CURVES`,
      :data:`SENSOR_QE_CURVES`) — hand-shaped to be physically plausible
      (peak wavelengths, rough FWHM/QE-curve shape) but they are **not**
      digitized vendor data. Do not treat the curve *values* as
      manufacturer-accurate; the *integration math* around them is what is
      faithful.
    - :class:`SFCCParams` accepts ``custom_filter_curves`` /
      ``custom_sensor_curves`` (name -> (wavelength_nm, throughput) arrays)
      so a user can supply real digitized curves (e.g. from a manufacturer
      datasheet or a CSV export) and get the same faithful integration
      against real data. :func:`load_curve_csv` reads SASpro's own
      2-column CSV convention (wavelength_nm, response) for this purpose.
  SASpro's DR4 roadmap notes at the top of its ``sfcc.py`` describe an even
  more ambitious "blindly solve the sensor response from the data" scheme
  that supersedes bundled QE curves entirely; that is future work, not
  ported here (it was a design note in the source, not shipped code).
  The optional 3x3 color-matrix refinement in SASpro's dialog is likewise
  not ported — it is hidden/unchecked-by-default even in the source
  (``self.color_matrix_chk.hide()``), which itself documents this exact
  DR4-era distrust of the per-star color matrix approach.

GPU / CPU
---------
The physical model (per-star blackbody + filter x QE integration) works on
tiny arrays (one spectrum per matched star, a few hundred wavelength
samples) — this is CPU/numpy work; a GPU dispatch here would be pure
overhead. The only per-pixel, full-image operation is the final elementwise
channel scale (:func:`_scale_channel`), which *is* gated through
``get_device_manager()`` above a size threshold, with a try/except CUDA-OOM
fallback to CPU. The GPU on this dev machine was contended by LM Studio
during development (as in other recent ports here) — no GPU benchmark is
claimed; the dispatch is a correctness-preserving fallback, not a measured
speedup.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Reuse, don't reimplement: the stellar flux model and aperture photometry
# are identical machinery to SPCC. SFCC's novelty is the filter x QE system
# response construction below, not the star flux model or photometry.
from astraios.core.spcc import _aperture_flux as aperture_flux
from astraios.core.spcc import _blackbody_flux as blackbody_flux
from astraios.core.spcc import _bp_rp_to_teff as bp_rp_to_teff

log = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]

Curve = tuple[np.ndarray, np.ndarray]  # (wavelength_nm, throughput [0,1])


def _noop_progress(_f: float, _m: str) -> None:
    pass


# ── Bundled representative filter / sensor / reference-star data ─────────────
# See the module docstring: these are hand-shaped representative curves
# (plausible peak/FWHM/roll-off), NOT digitized vendor measurements. Swap in
# real data via SFCCParams.custom_filter_curves / custom_sensor_curves.


def _gaussian_bandpass(center_nm: float, fwhm_nm: float, n_points: int = 41,
                        half_width_factor: float = 4.0) -> Curve:
    """A narrowband-filter-shaped Gaussian transmission curve."""
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = half_width_factor * sigma
    wl = np.linspace(center_nm - half, center_nm + half, n_points, dtype=np.float64)
    tp = np.exp(-0.5 * ((wl - center_nm) / sigma) ** 2)
    return wl, tp


def _curve(wl_nm: list[float], tp: list[float]) -> Curve:
    return np.asarray(wl_nm, dtype=np.float64), np.asarray(tp, dtype=np.float64)


FILTER_CURVES: dict[str, Curve] = {
    # Generic dye-based Bayer/OSC color filter array — broad, overlapping.
    "Bayer-R (generic OSC)": _curve(
        [400, 450, 500, 550, 580, 600, 620, 650, 680, 720, 760, 800, 850, 900, 950, 1000],
        [0.02, 0.03, 0.05, 0.10, 0.25, 0.55, 0.85, 0.97, 0.95, 0.85,
         0.65, 0.45, 0.30, 0.20, 0.12, 0.06],
    ),
    "Bayer-G (generic OSC)": _curve(
        [400, 450, 480, 500, 520, 540, 560, 580, 600, 620, 650, 700, 750, 800],
        [0.03, 0.15, 0.45, 0.75, 0.95, 1.00, 0.90, 0.65, 0.35, 0.15, 0.06, 0.03, 0.02, 0.01],
    ),
    "Bayer-B (generic OSC)": _curve(
        [350, 380, 400, 420, 440, 460, 480, 500, 520, 550, 600, 650, 700],
        [0.02, 0.15, 0.45, 0.80, 0.97, 1.00, 0.85, 0.55, 0.25, 0.10, 0.04, 0.02, 0.01],
    ),
    # Sharper-cutoff interference RGB set, like a generic LRGB filter wheel set.
    "Broadband-R (generic LRGB interference)": _curve(
        [580, 590, 600, 610, 650, 690, 700, 710],
        [0.02, 0.05, 0.85, 0.97, 0.98, 0.97, 0.85, 0.03],
    ),
    "Broadband-G (generic LRGB interference)": _curve(
        [480, 490, 500, 510, 550, 590, 600, 610],
        [0.02, 0.05, 0.85, 0.97, 0.96, 0.95, 0.85, 0.03],
    ),
    "Broadband-B (generic LRGB interference)": _curve(
        [380, 390, 400, 410, 450, 480, 490, 500],
        [0.02, 0.05, 0.83, 0.95, 0.97, 0.95, 0.85, 0.03],
    ),
    # Narrowband, ~7nm FWHM (illustrative — real filters range 3-12nm).
    "Ha (656nm, 7nm narrowband)": _gaussian_bandpass(656.3, 7.0),
    "OIII (501nm, 7nm narrowband)": _gaussian_bandpass(500.7, 7.0),
    "SII (672nm, 7nm narrowband)": _gaussian_bandpass(672.4, 7.0),
    # Usable as an LP/cut filter stacked on top of an R/G/B filter.
    "UV/IR Cut (generic, 400-700nm)": _curve(
        [350, 390, 400, 410, 690, 700, 710, 750],
        [0.0, 0.02, 0.9, 0.98, 0.98, 0.9, 0.02, 0.0],
    ),
    "Clear / No Filter": _curve([300, 1100], [1.0, 1.0]),
}
FILTER_CURVE_NAMES: list[str] = sorted(FILTER_CURVES)

SENSOR_QE_CURVES: dict[str, Curve] = {
    "Ideal (100% QE)": _curve([300, 1100], [1.0, 1.0]),
    "Generic CMOS back-illuminated (Sony IMX-class)": _curve(
        [350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
        [0.15, 0.35, 0.55, 0.68, 0.78, 0.82, 0.80, 0.72, 0.58, 0.42, 0.28, 0.16, 0.08, 0.03],
    ),
    "Generic CCD (Kodak KAF-class)": _curve(
        [350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000],
        [0.05, 0.20, 0.35, 0.45, 0.55, 0.62, 0.65, 0.60, 0.48, 0.35, 0.22, 0.12, 0.06, 0.02],
    ),
}
SENSOR_QE_NAMES: list[str] = sorted(SENSOR_QE_CURVES)

# Effective temperature (K) for each white-reference stellar type. Used only
# as a diagnostic reference ratio (SFCCResult.reference_ratios) — mirrors
# SASpro's S_ref_R/G/B, which is computed in run_spcc() but likewise not fed
# back into the fit there either (see module docstring).
STELLAR_REFERENCE_TEFF: dict[str, float] = {
    "G2V (Sun-like, default)": 5778.0,
    "A0V (Vega)": 9600.0,
    "K0V": 5250.0,
    "M0V": 3800.0,
    "B0V": 30000.0,
}
WHITE_REFERENCE_NAMES: list[str] = list(STELLAR_REFERENCE_TEFF)


def load_curve_csv(path: str | Path) -> Curve:
    """Load a 2-column ``(wavelength_nm, response)`` CSV as a :data:`Curve`.

    Mirrors SASpro's ``_import_curve_from_csv`` (sfcc.py) so users can plug
    in real vendor-measured filter/sensor curves. Whitespace and ``#``
    comment lines are ignored; anything not two floats per row raises
    ``ValueError``.
    """
    wl: list[float] = []
    tp: list[float] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: expected 'wavelength_nm response', got {raw!r}")
            try:
                wl.append(float(parts[0]))
                tp.append(float(parts[1]))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: could not parse {raw!r} as two floats") from exc
    if len(wl) < 2:
        raise ValueError(f"{path}: need at least 2 data rows, found {len(wl)}")
    return np.asarray(wl, dtype=np.float64), np.asarray(tp, dtype=np.float64)


# ── Wavelength grid + system response construction (the ported physics) ─────

def wavelength_grid(lam_min_nm: float = 350.0, lam_max_nm: float = 1000.0,
                     step_nm: float = 2.0) -> np.ndarray:
    """Common integration grid, analogous to SASpro's ``wl_grid`` (3000-11000A)."""
    return np.arange(lam_min_nm, lam_max_nm + step_nm, step_nm, dtype=np.float64)


def _resample(curve: Curve, grid_nm: np.ndarray) -> np.ndarray:
    wl, tp = curve
    return np.interp(grid_nm, wl, tp, left=0.0, right=0.0)


def build_system_response(
    grid_nm: np.ndarray,
    filter_curve: Curve,
    sensor_curve: Curve,
    lp_curves: Iterable[Curve] = (),
) -> np.ndarray:
    """``T_sys(lambda) = T_filter(lambda) x QE_sensor(lambda) x prod(T_LP_i(lambda))``.

    Ported from SASpro's ``sfcc.py`` system-response construction
    (``T_sys_R = T_R * QE * LP`` in ``run_spcc()``, and identically in
    ``SaspViewer.update_plot()``): each curve is resampled (linear
    interpolation, zero outside its native range — same as SASpro's
    ``np.interp(wl_grid, wl_o, tp_o, left=0.0, right=0.0)``) onto a common
    grid, then multiplied together.
    """
    T = _resample(filter_curve, grid_nm) * _resample(sensor_curve, grid_nm)
    for lp in lp_curves:
        T = T * _resample(lp, grid_nm)
    return T


def integrate_flux_times_transmission(flux: np.ndarray, wl_nm: np.ndarray,
                                       transmission: np.ndarray) -> float:
    """``integral of flux(lambda) x T(lambda) d(lambda)``.

    Direct port of SASpro's ``_integrate_flux_times_T`` (sfcc.py line ~310):
    "Units don't matter for colors/ratios; we just need consistency."
    """
    f = np.asarray(flux, dtype=np.float64).reshape(-1)
    t = np.asarray(transmission, dtype=np.float64).reshape(-1)
    w = np.asarray(wl_nm, dtype=np.float64).reshape(-1)
    if f.size != w.size or t.size != w.size:
        raise ValueError("Integration arrays must be the same length")
    return float(np.trapezoid(f * t, w))


def expected_channel_ratios(
    teff_k: float,
    grid_nm: np.ndarray,
    t_sys_r: np.ndarray,
    t_sys_g: np.ndarray,
    t_sys_b: np.ndarray,
) -> tuple[float, float, float]:
    """``S_expected_c = integral(flux_star(lambda, Teff) x T_sys_c(lambda)) d(lambda)``, c in R,G,B.

    This is SFCC's ported physical core (SASpro ``sfcc.py``'s
    ``measured_c(i) = ... integral flux_star_i(lambda) x T_filter_c(lambda) x QE(lambda)``,
    background section + ``_gaia_integrals_for_source_ids``). The stellar
    flux model is the blackbody reused from :mod:`astraios.core.spcc`
    (documented reduction — see module docstring); the integration against
    the filter x QE system response is faithful.
    """
    flux = blackbody_flux(grid_nm, teff_k)
    s_r = integrate_flux_times_transmission(flux, grid_nm, t_sys_r)
    s_g = integrate_flux_times_transmission(flux, grid_nm, t_sys_g)
    s_b = integrate_flux_times_transmission(flux, grid_nm, t_sys_b)
    return s_r, s_g, s_b


# ── Params / result ───────────────────────────────────────────────────────────

@dataclass
class SFCCParams:
    """Settings for Spectral Flux Color Calibration.

    Filter/sensor/white-reference names are looked up in :data:`FILTER_CURVES`
    / :data:`SENSOR_QE_CURVES` / :data:`STELLAR_REFERENCE_TEFF` first, then in
    the matching ``custom_*`` override dict, so user-supplied curves (e.g.
    from :func:`load_curve_csv`) can be named and selected the same way.
    """

    # Filter transmission per channel — independent per channel, like SASpro's
    # r_filter_combo / g_filter_combo / b_filter_combo (any curve, any channel).
    filter_r: str = "Broadband-R (generic LRGB interference)"
    filter_g: str = "Broadband-G (generic LRGB interference)"
    filter_b: str = "Broadband-B (generic LRGB interference)"
    # Optional extra "LP/cut" filters stacked into every channel's T_sys,
    # like SASpro's lp_filter_combo / lp_filter_combo2. None = "(None)".
    lp_filter_1: str | None = None
    lp_filter_2: str | None = None
    # Sensor QE profile, like SASpro's sens_combo.
    sensor: str = "Generic CMOS back-illuminated (Sony IMX-class)"
    # White-reference spectral type for the diagnostic reference ratio only
    # (SFCCResult.reference_ratios) — like SASpro's star_combo (default G2V).
    white_reference: str = "G2V (Sun-like, default)"

    # name -> (wavelength_nm, throughput) overrides/additions, e.g. loaded via
    # load_curve_csv(). Same name-lookup applies to filter_r/g/b, lp_filter_1/2.
    custom_filter_curves: dict[str, Curve] | None = None
    # name -> (wavelength_nm, qe) overrides/additions for `sensor`.
    custom_sensor_curves: dict[str, Curve] | None = None

    # Catalog source for reference stars, used by apply_sfcc() when
    # catalog_stars isn't supplied directly. "vizier_gaia_dr3" queries Gaia
    # DR3 live via astraios.core.star_catalog (has BP/RP color). The locally
    # cached offline catalog (astraios.core.gaia_catalog) only stores G
    # magnitude, not BP/RP, so it cannot drive the color model yet — selecting
    # "offline_gaia" raises a clear error rather than silently mis-computing.
    catalog: str = "vizier_gaia_dr3"
    search_radius_deg: float = 0.5
    mag_limit: float = 16.0

    # Star detection (astraios.core.star_detection.detect_stars). SASpro's
    # SEP-based default threshold is 15 sigma; our simpler contour detector
    # is tuned differently, so the default here is lower (8 sigma) — tune per
    # image.
    detection_sigma: float = 8.0
    max_stars_detect: int = 300
    # Matching tolerance between a detected star centroid and a WCS-projected
    # catalog position, in pixels (SASpro: 3px, run_spcc() raw_matches).
    match_radius_px: float = 3.0
    # Reject stars whose peak pixel (any channel, 3x3 patch) exceeds this —
    # a saturated star's color is not physically meaningful.
    saturation_threshold: float = 0.98
    # Cap to the brightest N catalog-matched stars (SASpro: MAX_PHOT_STARS=500).
    max_phot_stars: int = 500
    # Minimum usable stars required, else apply_sfcc raises ValueError.
    min_stars: int = 8

    aperture_radius_px: float = 5.0

    # SASpro's default is unchecked (self.neutralize_chk.setChecked(False)).
    neutralize_background: bool = False


@dataclass
class SFCCResult:
    """Diagnostics from a completed SFCC run (see :func:`sfcc_calibrate`)."""

    scales: tuple[float, float, float]  # applied per-channel gain (R, G, B; G is the reference)
    n_detected: int
    n_matched: int
    n_used: int
    model: str
    rms_residual: float
    reference_ratios: tuple[float, float, float]  # diagnostic-only, see STELLAR_REFERENCE_TEFF doc
    used_gpu: bool = False


# ── Curve lookup ──────────────────────────────────────────────────────────────

def _lookup(name: str | None, bundled: dict[str, Curve],
            custom: dict[str, Curve] | None, kind: str) -> Curve:
    if name is None:
        return _curve([300, 1100], [1.0, 1.0])  # "(None)" -> transparent
    if custom and name in custom:
        return custom[name]
    if name in bundled:
        return bundled[name]
    available = sorted(set(bundled) | set(custom or {}))
    raise ValueError(f"Unknown {kind} curve {name!r}. Available: {available}")


# ── GPU/CPU channel scale (the only per-pixel, full-image op in this module) ─

_GPU_PIXEL_THRESHOLD = 4_000_000  # ~2000x2000; below this, CPU numpy is already instant


def _scale_channel(channel: np.ndarray, factor: float) -> np.ndarray:
    """``channel * factor``, dispatched to GPU only above a size threshold.

    The physical model above (per-star spectra) is CPU-only by design (small
    arrays, no benefit from GPU). This is the one full-image elementwise op,
    so it is the one that goes through ``get_device_manager()`` — gated
    behind a pixel-count threshold and wrapped in try/except so a CUDA OOM
    (the GPU here was contended by LM Studio during development; no
    benchmark is claimed) falls back to plain CPU numpy rather than crashing.
    """
    if channel.size >= _GPU_PIXEL_THRESHOLD:
        try:
            from astraios.core.device_manager import get_device_manager

            dm = get_device_manager()
            if dm.is_gpu:
                t = dm.from_numpy(np.ascontiguousarray(channel, dtype=np.float32))
                t = t * factor
                return dm.to_cpu(t).numpy()
        except Exception as exc:  # noqa: BLE001 - any GPU failure falls back to CPU
            log.debug("SFCC: GPU channel scale unavailable (%s), using CPU", exc)
    return (channel * factor).astype(np.float32)


# ── Core calibration (reuses SPCC's photometry + correction-application) ────

def sfcc_calibrate(
    data: np.ndarray,
    catalog_stars: list[tuple[float, float, float]],
    params: SFCCParams | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, SFCCResult]:
    """Apply Spectral Flux Color Calibration given pre-resolved catalog stars.

    Parameters
    ----------
    data : ndarray
        Float32 image, shape ``(3, H, W)``, channels-first RGB in [0, 1].
        Raises ``ValueError`` for mono/other-shaped input.
    catalog_stars : list of (x_img, y_img, bp_rp)
        Star pixel positions + Gaia BP-RP color index, same convention as
        :func:`astraios.core.spcc.spcc_calibrate` (obtained from plate solve
        + catalog query — see :func:`apply_sfcc` for the WCS-driven version).
    params : SFCCParams, optional
    progress : callable, optional

    Returns
    -------
    (ndarray, SFCCResult)
        Calibrated image (same shape/dtype as input) and fit diagnostics.
    """
    if params is None:
        params = SFCCParams()
    if progress is None:
        progress = _noop_progress

    if data.ndim != 3 or data.shape[0] != 3:
        raise ValueError(
            "SFCC requires a 3-channel (RGB) image, shape (3, H, W); got "
            f"shape {data.shape!r}. A mono/luminance image has no color to calibrate."
        )

    filter_r = _lookup(params.filter_r, FILTER_CURVES, params.custom_filter_curves, "filter")
    filter_g = _lookup(params.filter_g, FILTER_CURVES, params.custom_filter_curves, "filter")
    filter_b = _lookup(params.filter_b, FILTER_CURVES, params.custom_filter_curves, "filter")
    sensor = _lookup(params.sensor, SENSOR_QE_CURVES, params.custom_sensor_curves, "sensor")
    lp_curves = [
        _lookup(name, FILTER_CURVES, params.custom_filter_curves, "LP/cut filter")
        for name in (params.lp_filter_1, params.lp_filter_2)
        if name is not None
    ]

    grid = wavelength_grid()
    t_sys_r = build_system_response(grid, filter_r, sensor, lp_curves)
    t_sys_g = build_system_response(grid, filter_g, sensor, lp_curves)
    t_sys_b = build_system_response(grid, filter_b, sensor, lp_curves)

    progress(0.05, "Computing white-reference ratio…")
    ref_teff = STELLAR_REFERENCE_TEFF.get(params.white_reference, 5778.0)
    ref_r, ref_g, ref_b = expected_channel_ratios(ref_teff, grid, t_sys_r, t_sys_g, t_sys_b)
    ref_g_safe = max(ref_g, 1e-30)
    reference_ratios = (ref_r / ref_g_safe, 1.0, ref_b / ref_g_safe)

    progress(0.15, "Computing expected channel fluxes from stellar spectra (filter x QE)…")

    expected_ratios: list[tuple[float, float, float]] = []
    for _x, _y, bp_rp in catalog_stars:
        teff = bp_rp_to_teff(bp_rp)
        s_r, s_g, s_b = expected_channel_ratios(teff, grid, t_sys_r, t_sys_g, t_sys_b)
        total = max(s_r + s_g + s_b, 1e-30)
        expected_ratios.append((s_r / total, s_g / total, s_b / total))

    progress(0.35, "Measuring instrumental star fluxes…")

    n_detected = len(catalog_stars)
    inst_fluxes: list[np.ndarray] = []
    exp_fluxes_sel: list[np.ndarray] = []
    h, w = data.shape[1], data.shape[2]

    for i, (x_img, y_img, _bp_rp) in enumerate(catalog_stars):
        if params.saturation_threshold < 1.0:
            iy, ix = int(round(y_img)), int(round(x_img))
            y0, y1 = max(0, iy - 1), min(h, iy + 2)
            x0, x1 = max(0, ix - 1), min(w, ix + 2)
            patch_sat = (
                y1 > y0 and x1 > x0
                and float(data[:, y0:y1, x0:x1].max()) >= params.saturation_threshold
            )
            if patch_sat:
                continue

        flux = aperture_flux(data, x_img, y_img, r=params.aperture_radius_px)
        if flux is None or flux.min() < 1e-8:
            continue

        norm = flux[1]
        inst_fluxes.append(flux / norm)
        exp = np.array(expected_ratios[i])
        exp_fluxes_sel.append(exp / max(exp[1], 1e-30))

    n_matched = n_detected
    n_used = len(inst_fluxes)
    log.info("SFCC: %d/%d catalog stars usable for calibration", n_used, n_matched)

    if n_used < params.min_stars:
        raise ValueError(
            f"SFCC: only {n_used} usable stars found (need {params.min_stars}). "
            "Run plate solve first, widen the catalog search radius, or use a "
            "field with more stars."
        )

    progress(0.55, f"Fitting per-channel color scale from {n_used} stars…")

    inst_arr = np.array(inst_fluxes)
    exp_arr = np.array(exp_fluxes_sel)

    scales = np.zeros(3, dtype=np.float64)
    for c in range(3):
        ratios = inst_arr[:, c] / np.clip(exp_arr[:, c], 1e-10, None)
        weights = exp_arr[:, c]
        scales[c] = np.average(ratios, weights=weights)

    # Fractional RMS residual after correction (diagnostic, mirrors SASpro's
    # rms_frac() in run_spcc() Step D): corrected/expected should be ~1.
    g_scale = scales[1]
    corrected = inst_arr * (g_scale / np.clip(scales, 1e-10, None))[np.newaxis, :]
    resid = (corrected / np.clip(exp_arr, 1e-10, None)) - 1.0
    rms_residual = float(np.sqrt(np.mean(resid[:, [0, 2]] ** 2)))  # R,B only (G is the reference)

    log.info("SFCC channel scales: R=%.4f G=%.4f B=%.4f (rms=%.4f)", *scales, rms_residual)

    progress(0.80, "Applying color correction…")

    result = data.astype(np.float32, copy=True)
    used_gpu = False
    for c in range(3):
        if scales[c] > 1e-6:
            before = data[c].size >= _GPU_PIXEL_THRESHOLD
            result[c] = _scale_channel(data[c], g_scale / scales[c])
            used_gpu = used_gpu or before

    if params.neutralize_background:
        progress(0.92, "Neutralizing background…")
        hh, ww = result.shape[1], result.shape[2]
        margin = max(1, max(hh, ww) // 10)
        bg_vals = []
        for c in range(3):
            corner = result[c, :margin, :margin].ravel()
            bg_vals.append(float(np.percentile(corner, 10)))
        bg_min = min(bg_vals)
        for c in range(3):
            result[c] -= (bg_vals[c] - bg_min)

    progress(1.0, "SFCC complete")

    sfcc_result = SFCCResult(
        scales=(float(scales[0]), float(scales[1]), float(scales[2])),
        n_detected=n_detected,
        n_matched=n_matched,
        n_used=n_used,
        model="weighted-mean-ratio",
        rms_residual=rms_residual,
        reference_ratios=reference_ratios,
        used_gpu=used_gpu,
    )
    return np.clip(result, 0, 1).astype(np.float32), sfcc_result


def _snap_to_detections(
    projected: list[tuple[float, float, float]],
    positions: np.ndarray,
    max_dist_px: float,
) -> list[tuple[float, float, float]]:
    """Snap each WCS-projected catalog position to the nearest detected star.

    Mirrors SASpro's ``run_spcc()`` nearest-neighbor match (``dx,dy`` within
    3px, sfcc.py line ~3154): only catalog stars with a real nearby detection
    are kept, and photometry runs at the detector's centroid, not the
    (slightly less accurate) raw WCS-projected position.
    """
    if positions.size == 0:
        return []
    out = []
    max_d2 = max_dist_px * max_dist_px
    for px, py, bp_rp in projected:
        d2 = (positions[:, 0] - px) ** 2 + (positions[:, 1] - py) ** 2
        j = int(np.argmin(d2))
        if d2[j] <= max_d2:
            out.append((float(positions[j, 0]), float(positions[j, 1]), bp_rp))
    return out


def _resolve_catalog_stars(
    image: np.ndarray,
    wcs_header: dict | None,
    params: SFCCParams,
    progress: ProgressCallback,
) -> list[tuple[float, float, float]]:
    """Query a catalog + detect stars to build (x, y, bp_rp) triples.

    Reuses :func:`astraios.core.star_detection.detect_stars` (shared star
    detector) and the same WCS dict / tangent-plane projection convention
    :meth:`MainWindow._update_wcs_overlay` uses, so results are consistent
    with what SPCC/PCC already show as the WCS star overlay.
    """
    if params.catalog == "offline_gaia":
        raise ValueError(
            "SFCC catalog='offline_gaia' is not supported: the locally cached "
            "Gaia catalog (astraios.core.gaia_catalog) stores only G magnitude, "
            "not BP/RP color, so it cannot drive the spectral flux model. Use "
            "catalog='vizier_gaia_dr3' (the default), or pass catalog_stars=… "
            "directly if you already have (x, y, bp_rp) triples."
        )
    if params.catalog != "vizier_gaia_dr3":
        raise ValueError(f"Unknown catalog {params.catalog!r}. Use 'vizier_gaia_dr3'.")

    if not wcs_header:
        raise ValueError(
            "apply_sfcc needs either catalog_stars=[(x, y, bp_rp), …] or "
            "wcs_header=… (a plate-solved WCS dict with 'ra'/'dec', e.g. from "
            "astraios.core.star_catalog.plate_solve_auto) to locate reference stars."
        )

    from astraios.core.color_calibration import _make_pixel_to_sky
    from astraios.core.star_catalog import query_gaia_dr3
    from astraios.core.star_detection import detect_stars
    from astraios.core.wcs import normalise_wcs_dict

    wcs = normalise_wcs_dict(dict(wcs_header))
    ra, dec = wcs.get("ra"), wcs.get("dec")
    if ra is None or dec is None:
        raise ValueError(
            "wcs_header must resolve to 'ra'/'dec' (see astraios.core.wcs.normalise_wcs_dict)"
        )

    progress(0.02, f"Querying Gaia DR3 (r={params.search_radius_deg} deg)…")
    catalog = query_gaia_dr3(float(ra), float(dec), radius_deg=params.search_radius_deg)

    h, w = image.shape[1], image.shape[2]
    sky_fn = _make_pixel_to_sky(wcs, w, h)
    ra0, dec0 = sky_fn(w / 2.0, h / 2.0)
    cos_dec = float(np.cos(np.radians(dec0)))
    scale_deg = (wcs.get("scale") or 1.0) / 3600.0

    projected: list[tuple[float, float, float]] = []
    for star in catalog:
        if star.bp_mag is None or star.rp_mag is None:
            continue
        dra = (star.ra_deg - ra0) * cos_dec
        ddec = star.dec_deg - dec0
        px = w / 2.0 + dra / max(scale_deg, 1e-10)
        py = h / 2.0 - ddec / max(scale_deg, 1e-10)
        if 0 <= px < w and 0 <= py < h:
            projected.append((float(px), float(py), float(star.bp_mag) - float(star.rp_mag)))

    progress(0.08, f"Detecting stars ({len(projected)} catalog candidates in frame)…")
    detected = detect_stars(
        image, max_stars=params.max_stars_detect, sigma_threshold=params.detection_sigma
    )
    matched = _snap_to_detections(projected, detected.positions, params.match_radius_px)
    return matched[: params.max_phot_stars]


def apply_sfcc(
    image: np.ndarray,
    params: SFCCParams | None = None,
    wcs_header: dict | None = None,
    progress: ProgressCallback | None = None,
    catalog_stars: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Apply Spectral Flux Color Calibration to a plate-solved RGB image.

    Parameters
    ----------
    image : ndarray
        Float32 image, shape ``(3, H, W)``, channels-first RGB in [0, 1].
        Raises ``ValueError`` for mono/other-shaped input.
    params : SFCCParams, optional
    wcs_header : dict, optional
        A plate-solved WCS dict (``{'ra':, 'dec':, 'scale':, ...}`` — see
        :func:`astraios.core.star_catalog.plate_solve_auto`). Used to query
        Gaia DR3 and locate reference stars when ``catalog_stars`` isn't
        supplied directly. Required unless ``catalog_stars`` is given.
    progress : callable, optional
    catalog_stars : list of (x_img, y_img, bp_rp), optional
        Pre-resolved catalog star positions + color index (same convention
        as :func:`astraios.core.spcc.spcc_calibrate`). Bypasses the WCS/
        network query entirely — this is how the dialog reuses an
        already-computed WCS star overlay (see
        ``MainWindow._wcs_overlay_stars``), and how tests exercise SFCC
        deterministically/offline.

    Returns
    -------
    ndarray
        Colour-calibrated image, same shape/dtype as input.
    """
    if params is None:
        params = SFCCParams()
    if progress is None:
        progress = _noop_progress

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            "SFCC requires a 3-channel (RGB) image, shape (3, H, W); got "
            f"shape {image.shape!r}. A mono/luminance image has no color to calibrate."
        )

    if catalog_stars is None:
        catalog_stars = _resolve_catalog_stars(image, wcs_header, params, progress)

    result, sfcc_result = sfcc_calibrate(image, catalog_stars, params=params, progress=progress)
    log.info(
        "SFCC complete: scales R=%.4f G=%.4f B=%.4f, %d/%d stars used, rms=%.4f, model=%s",
        *sfcc_result.scales, sfcc_result.n_used, sfcc_result.n_matched,
        sfcc_result.rms_residual, sfcc_result.model,
    )
    return result
