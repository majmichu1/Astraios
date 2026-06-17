"""Aperture photometry — measure star brightness with configurable apertures."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


@dataclass
class PhotometryParams:
    aperture_radius: float = 10.0  # pixels
    annulus_inner: float = 15.0
    annulus_outer: float = 20.0
    detection_threshold: float = 5.0  # sigma
    max_sources: int = 500


@dataclass
class PhotometryResult:
    x: NDArray  # pixel coordinates
    y: NDArray
    flux: NDArray  # aperture sum flux
    flux_err: NDArray
    mag: NDArray  # instrumental magnitude (-2.5 * log10(flux))
    bg_median: NDArray  # local background per star
    n_sources: int


def _detect_sources(
    image: NDArray,
    threshold: float,
    max_sources: int,
) -> tuple[NDArray, NDArray]:
    """Detect sources above a threshold using connected-component labelling.

    Parameters
    ----------
    image : ndarray
        2D float array.
    threshold : float
        Pixel value threshold for detection.
    max_sources : int
        Maximum number of sources to return (brightest first).

    Returns
    -------
    x : ndarray
        X-centroids of detected sources.
    y : ndarray
        Y-centroids of detected sources.
    """
    from scipy.ndimage import center_of_mass, label

    binary = image > threshold
    labelled, n_labels = label(binary)

    sources_x: list[float] = []
    sources_y: list[float] = []
    fluxes: list[float] = []

    for label_id in range(1, n_labels + 1):
        mask = labelled == label_id
        cy, cx = center_of_mass(image, labelled, label_id)
        flux = float(np.sum(image[mask]))
        if np.isfinite(cy) and np.isfinite(cx):
            sources_y.append(float(cy))
            sources_x.append(float(cx))
            fluxes.append(flux)

    if len(sources_x) > max_sources:
        order = np.argsort(fluxes)[::-1][:max_sources]
        sources_x = [sources_x[i] for i in order]
        sources_y = [sources_y[i] for i in order]

    return np.array(sources_x, dtype=np.float64), np.array(sources_y, dtype=np.float64)


def _aperture_sum(
    image: NDArray,
    x: float,
    y: float,
    radius: float,
) -> float:
    """Compute the sum of pixel values within a circular aperture.

    Parameters
    ----------
    image : ndarray
        2D float array.
    x, y : float
        Centre coordinates (column, row).
    radius : float
        Aperture radius in pixels.

    Returns
    -------
    float
        Sum of pixel values inside the aperture.
    """
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - x) ** 2 + (yy - y) ** 2) <= radius ** 2
    return float(np.sum(image[mask]))


def _annulus_stats(
    image: NDArray,
    x: float,
    y: float,
    r_in: float,
    r_out: float,
) -> tuple[float, float]:
    """Compute median and std of pixel values in an annular sky region.

    Parameters
    ----------
    image : ndarray
        2D float array.
    x, y : float
        Centre coordinates.
    r_in : float
        Inner radius of annulus.
    r_out : float
        Outer radius of annulus.

    Returns
    -------
    median : float
        Median of annulus pixels (local background estimate).
    std : float
        Standard deviation of annulus pixels.
    """
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    mask = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)
    pixels = image[mask]
    if len(pixels) == 0:
        return 0.0, 0.0
    return float(np.median(pixels)), float(np.std(pixels))


def _photutils_photometry(
    image: NDArray,
    params: PhotometryParams,
) -> PhotometryResult:
    """Photometry using photutils."""
    from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
    from photutils.detection import DAOStarFinder

    median = float(np.median(image))
    std = float(np.std(image))
    threshold = median + params.detection_threshold * std

    daofind = DAOStarFinder(fwhm=params.aperture_radius / 2.0, threshold=threshold)
    sources = daofind(image)

    if sources is None or len(sources) == 0:
        return PhotometryResult(
            x=np.empty(0), y=np.empty(0), flux=np.empty(0),
            flux_err=np.empty(0), mag=np.empty(0), bg_median=np.empty(0),
            n_sources=0,
        )

    if len(sources) > params.max_sources:
        sources = sources[:params.max_sources]

    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
    aperture = CircularAperture(positions, r=params.aperture_radius)
    annulus = CircularAnnulus(
        positions, r_in=params.annulus_inner, r_out=params.annulus_outer,
    )

    phot = aperture_photometry(image, aperture)
    ann_phot = aperture_photometry(image, annulus)

    x = np.array(sources["xcentroid"], dtype=np.float64)
    y = np.array(sources["ycentroid"], dtype=np.float64)
    flux = np.array(phot["aperture_sum"], dtype=np.float64)
    ann_median = np.array(ann_phot["aperture_sum"], dtype=np.float64)
    ann_area = np.pi * (params.annulus_outer ** 2 - params.annulus_inner ** 2)
    bg_median = ann_median / max(ann_area, 1.0)

    flux_bg = bg_median * np.pi * params.aperture_radius ** 2
    flux_corrected = flux - flux_bg
    flux_corrected = np.clip(flux_corrected, 0, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(np.maximum(flux_corrected, 1e-10))

    flux_err = np.sqrt(np.maximum(flux_corrected, 0)) + ann_median

    return PhotometryResult(
        x=x, y=y, flux=flux_corrected, flux_err=flux_err,
        mag=mag, bg_median=bg_median, n_sources=len(x),
    )


def _sep_photometry(
    image: NDArray,
    params: PhotometryParams,
) -> PhotometryResult:
    """Photometry using sep (faster fallback)."""
    import sep

    bg = sep.Background(image)
    bg_sub = image - bg.back()
    bg_rms = bg.rms()

    objects = sep.extract(
        bg_sub, params.detection_threshold, err=bg_rms,
    )

    if len(objects) == 0:
        return PhotometryResult(
            x=np.empty(0), y=np.empty(0), flux=np.empty(0),
            flux_err=np.empty(0), mag=np.empty(0), bg_median=np.empty(0),
            n_sources=0,
        )

    if len(objects) > params.max_sources:
        order = np.argsort(objects["flux"])[::-1][:params.max_sources]
        objects = objects[order]

    flux, flux_err, flag = sep.sum_circle(
        bg_sub, objects["x"], objects["y"],
        params.aperture_radius, err=bg_rms,
    )

    bg_flux, bg_flux_err, bg_flag = sep.sum_circle(
        image, objects["x"], objects["y"],
        params.aperture_radius,
    )
    ann_flux, ann_flux_err, ann_flag = sep.sum_circle(
        image, objects["x"], objects["y"],
        params.annulus_outer, err=bg_rms,
    )

    ann_area = np.pi * (params.annulus_outer ** 2 - params.annulus_inner ** 2)
    bg_median = bg_flux / max(ann_area, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(np.maximum(flux, 1e-10))

    return PhotometryResult(
        x=np.array(objects["x"], dtype=np.float64),
        y=np.array(objects["y"], dtype=np.float64),
        flux=np.array(flux, dtype=np.float64),
        flux_err=np.array(flux_err, dtype=np.float64),
        mag=np.array(mag, dtype=np.float64),
        bg_median=np.array(bg_median, dtype=np.float64),
        n_sources=len(objects),
    )


def run_photometry(
    image: NDArray,
    params: PhotometryParams | None = None,
) -> PhotometryResult:
    """Run aperture photometry on an image.

    Uses photutils if available, then sep, then a pure-numpy fallback.

    Parameters
    ----------
    image : ndarray
        (H, W) float32 image.
    params : PhotometryParams, optional
        Photometry parameters. Uses defaults if *None*.

    Returns
    -------
    PhotometryResult
        Flux, magnitude for each detected source.
    """
    if params is None:
        params = PhotometryParams()

    if image.ndim == 3:
        gray = np.mean(image, axis=0).astype(np.float64)
    else:
        gray = image.astype(np.float64)

    # Try photutils first (most feature-complete)
    try:
        log.debug("Attempting photutils photometry")
        result = _photutils_photometry(gray, params)
        if result.n_sources > 0:
            log.info("photutils: %d sources detected", result.n_sources)
            return result
    except ImportError:
        log.debug("photutils not available")
    except Exception as exc:
        log.warning("photutils photometry failed: %s", exc)

    # Try sep as fallback
    try:
        log.debug("Attempting sep photometry")
        result = _sep_photometry(gray, params)
        if result.n_sources > 0:
            log.info("sep: %d sources detected", result.n_sources)
            return result
    except ImportError:
        log.debug("sep not available")
    except Exception as exc:
        log.warning("sep photometry failed: %s", exc)

    # Pure-numpy fallback
    log.debug("Falling back to pure-numpy photometry")
    median = float(np.median(gray))
    mad = float(np.median(np.abs(gray - median)))
    noise = max(mad * 1.4826, 1e-6)
    threshold = median + params.detection_threshold * noise

    x_arr, y_arr = _detect_sources(gray, threshold, params.max_sources)
    n = len(x_arr)

    if n == 0:
        return PhotometryResult(
            x=np.empty(0), y=np.empty(0), flux=np.empty(0),
            flux_err=np.empty(0), mag=np.empty(0), bg_median=np.empty(0),
            n_sources=0,
        )

    fluxes = np.zeros(n, dtype=np.float64)
    bgs = np.zeros(n, dtype=np.float64)
    flux_errs = np.zeros(n, dtype=np.float64)

    for i in range(n):
        fluxes[i] = _aperture_sum(gray, x_arr[i], y_arr[i], params.aperture_radius)
        bg, _ = _annulus_stats(gray, x_arr[i], y_arr[i], params.annulus_inner, params.annulus_outer)
        bgs[i] = bg
        bg_sum = bg * np.pi * params.aperture_radius ** 2
        fluxes[i] = max(fluxes[i] - bg_sum, 0.0)
        flux_errs[i] = np.sqrt(fluxes[i] + bg_sum)

    with np.errstate(divide="ignore", invalid="ignore"):
        mags = -2.5 * np.log10(np.maximum(fluxes, 1e-10))

    log.info("numpy photometry: %d sources detected", n)

    return PhotometryResult(
        x=x_arr, y=y_arr, flux=fluxes, flux_err=flux_errs,
        mag=mags, bg_median=bgs, n_sources=n,
    )
