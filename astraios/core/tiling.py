"""Memory-bounded tiled application of pixel-wise image operations.

Huge frames (tens of megapixels) blow past a constrained machine's RAM when a
processing stage allocates a full-resolution output *on top of* the working
image. For operations whose output pixel depends only on the input pixel —
colour adjustment, SCNR, gain — we can apply them one tile at a time, writing
each tile's result back into the working array in place, so the extra memory is
one tile instead of a whole second image.

Crucially this is only correct when the operation's *parameters* are global and
already fixed (saturation factor, SCNR amount, …): the apply is pixel-wise, so a
tiled result is bit-identical to the full-frame result — no seams. Do NOT use
this for stages whose parameters are derived from the frame as a whole
(background fit, stretch, colour balance); tile those by measuring once globally
and only then applying.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Default tile edge in pixels — a 2048x2048x3 float32 tile is ~50MB.
_DEFAULT_TILE = 2048
# Tile only when the frame is large enough to matter.
_TILE_THRESHOLD_MP = 24.0


def should_tile(image: np.ndarray, threshold_mp: float = _TILE_THRESHOLD_MP) -> bool:
    """True if ``image`` is large enough that tiled application is worthwhile."""
    if image.ndim < 2:
        return False
    h, w = image.shape[-2], image.shape[-1]
    return (h * w) / 1e6 >= threshold_mp


def iter_tiles(h: int, w: int, tile: int = _DEFAULT_TILE):
    """Yield ``(y0, y1, x0, x1)`` bounds covering an ``h x w`` grid in tiles."""
    for y0 in range(0, h, tile):
        y1 = min(h, y0 + tile)
        for x0 in range(0, w, tile):
            x1 = min(w, x0 + tile)
            yield y0, y1, x0, x1


def apply_pixelwise_tiled(
    image: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
    tile: int = _DEFAULT_TILE,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Apply a pixel-wise ``fn`` to ``image`` tile-by-tile, **in place**.

    ``fn(tile) -> tile`` must return an array of the same shape as the tile it is
    given and depend only on the tile's own pixels. ``image`` is mutated and
    returned, so the caller must own it (e.g. a throwaway working buffer). For a
    genuinely pixel-wise ``fn`` the result is identical to ``fn(image)``.

    Args:
        image: ``(H, W)`` or ``(C, H, W)`` array, mutated in place.
        fn: Pixel-wise operation applied to each tile.
        tile: Tile edge length in pixels.
        progress: Optional callback receiving a 0..1 fraction.

    Returns:
        ``image`` (the same object), mutated.
    """
    h, w = image.shape[-2], image.shape[-1]
    n_tiles = ((h + tile - 1) // tile) * ((w + tile - 1) // tile)
    for done, (y0, y1, x0, x1) in enumerate(iter_tiles(h, w, tile), start=1):
        if image.ndim == 3:
            sub = np.ascontiguousarray(image[:, y0:y1, x0:x1])
            image[:, y0:y1, x0:x1] = fn(sub)
        else:
            sub = np.ascontiguousarray(image[y0:y1, x0:x1])
            image[y0:y1, x0:x1] = fn(sub)
        if progress is not None:
            progress(done / n_tiles)
    return image
