"""Layer compositing system — Photoshop-style layer stack with blend modes.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.

Design notes (Astraios adaptation):
    - SASpro keeps a separate "base document" plus a list of ``ImageLayer``
      objects that composite on top of it. Astraios has no MDI/document
      model, so the base image is simply the bottom-most entry of
      :class:`LayerStack.layers` — everything is one ordered list.
    - ``LayerStack.layers[0]`` is the TOP of the visual stack (rendered
      last / on top); ``layers[-1]`` is the BOTTOM (rendered first). This
      matches a typical layers-panel UI where the first row is the topmost
      layer.
    - Compositing walks bottom-to-top, blending each layer against the
      accumulated result beneath it — identical math to SASpro's
      ``composite_stack``.
    - All blend-mode math runs through PyTorch tensors placed on the
      device chosen by :func:`astraios.core.device_manager.get_device_manager`,
      so a CUDA/MPS GPU is used automatically when present and the code
      transparently falls back to CPU tensors otherwise (per the project's
      "always go through DeviceManager" rule). Only :mod:`torch` ops that
      run identically on any device are used.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from astraios.core.device_manager import get_device_manager
from astraios.core.masks import Mask

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All blend modes ported from SASpro's ``pro/layers.py`` (Normal through
# Luminosity), plus "Subtract" — a standard blend mode SASpro did not name
# explicitly but which is trivial and commonly expected alongside "Add".
# ---------------------------------------------------------------------------
BLEND_MODES: list[str] = [
    "Normal",
    "Multiply",
    "Screen",
    "Overlay",
    "Soft Light",
    "Hard Light",
    "Color Dodge",
    "Color Burn",
    "Pin Light",
    "Add",
    "Subtract",
    "Lighten",
    "Darken",
    "Difference",
    "Difference (Squared)",
    "Relativistic Addition",
    "Sigmoid",
    "Luminosity",
]

_EPS = 1e-6


@dataclass
class Layer:
    """A single layer in a :class:`LayerStack`.

    Attributes
    ----------
    name : str
        User-visible layer name.
    data : ndarray
        float32 image data in [0, 1]. Mono: ``(H, W)``. Color: ``(C, H, W)``.
    opacity : float
        Overall layer opacity, 0..1.
    blend_mode : str
        One of :data:`BLEND_MODES`.
    visible : bool
        Whether the layer contributes to the composite.
    mask : Mask | None
        Optional (H, W) float32 [0, 1] mask restricting where the blend
        applies (1.0 = fully blended, 0.0 = layer has no effect there).
    sigmoid_center, sigmoid_strength : float
        Parameters for the "Sigmoid" blend mode (luminance-weighted mix).
    """

    name: str
    data: np.ndarray
    opacity: float = 1.0
    blend_mode: str = "Normal"
    visible: bool = True
    mask: Mask | None = None
    sigmoid_center: float = 0.5
    sigmoid_strength: float = 10.0

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.float32)
        if arr.ndim not in (2, 3):
            raise ValueError(f"Layer data must be (H,W) or (C,H,W), got shape {arr.shape}")
        self.data = np.clip(arr, 0.0, 1.0)
        if self.blend_mode not in BLEND_MODES:
            log.warning("Unknown blend mode %r, falling back to Normal", self.blend_mode)
            self.blend_mode = "Normal"
        self.opacity = float(np.clip(self.opacity, 0.0, 1.0))

    @property
    def is_color(self) -> bool:
        return self.data.ndim == 3

    @property
    def height(self) -> int:
        return int(self.data.shape[-2])

    @property
    def width(self) -> int:
        return int(self.data.shape[-1])

    def copy(self, name: str | None = None) -> Layer:
        """Return a deep copy (independent pixel/mask buffers)."""
        mask_copy = None
        if self.mask is not None:
            mask_copy = Mask(
                data=self.mask.data.copy(),
                name=self.mask.name,
                mask_type=self.mask.mask_type,
            )
        return Layer(
            name=name or f"{self.name} copy",
            data=self.data.copy(),
            opacity=self.opacity,
            blend_mode=self.blend_mode,
            visible=self.visible,
            mask=mask_copy,
            sigmoid_center=self.sigmoid_center,
            sigmoid_strength=self.sigmoid_strength,
        )


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def _luminance_chw(t: torch.Tensor) -> torch.Tensor:
    """Rec.709 luma of a (C,H,W) or (H,W) tensor -> (H,W)."""
    if t.ndim == 2:
        return t
    if t.shape[0] >= 3:
        return 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2]
    return t[0]


def _resize_chw(t: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear-resize a (C,H,W) or (H,W) tensor to ``hw`` if needed."""
    h, w = hw
    if t.shape[-2:] == (h, w):
        return t
    if t.ndim == 2:
        out = torch.nn.functional.interpolate(
            t[None, None], size=(h, w), mode="bilinear", align_corners=False
        )
        return out[0, 0]
    out = torch.nn.functional.interpolate(
        t[None], size=(h, w), mode="bilinear", align_corners=False
    )
    return out[0]


def _clip_color(c: torch.Tensor) -> torch.Tensor:
    """W3C compositing-spec ``ClipColor`` — pull an out-of-gamut RGB tensor
    back into [0, 1] while preserving its luminance. Used by the
    "Luminosity" blend mode's ``SetLum`` step.
    """
    lum = _luminance_chw(c).unsqueeze(0)
    n = c.amin(dim=0, keepdim=True)
    x = c.amax(dim=0, keepdim=True)

    out = c
    below = (n < 0.0).expand_as(out)
    denom_lo = torch.clamp(lum - n, min=_EPS)
    fixed_lo = lum + (out - lum) * lum / denom_lo
    out = torch.where(below, fixed_lo, out)

    above = (x > 1.0).expand_as(out)
    denom_hi = torch.clamp(x - lum, min=_EPS)
    fixed_hi = lum + (out - lum) * (1.0 - lum) / denom_hi
    out = torch.where(above, fixed_hi, out)

    return torch.clamp(out, 0.0, 1.0)


def _set_luminosity(base: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """W3C compositing-spec ``SetLum(base, Lum(src))`` — the standard
    "Luminosity" blend mode: keep base's hue/saturation, replace its
    luminance with src's. (SASpro's own "Luminosity" mode called out to an
    external ``luminancerecombine`` helper not present in this codebase;
    this is the standard formulation it approximates, and reduces to a
    plain ``src`` passthrough for mono data — there is no chrominance to
    preserve.)
    """
    if base.ndim == 2:
        return src
    d = (_luminance_chw(src) - _luminance_chw(base)).unsqueeze(0)
    return _clip_color(base + d)


def _apply_blend_mode(
    mode: str, base: torch.Tensor, src: torch.Tensor, layer: Layer
) -> torch.Tensor:
    """Core blend math. ``base``/``src`` are same-shape float32 tensors in
    [0, 1] (either (H,W) or (C,H,W)); returns the blended result, still in
    [0, 1] and unweighted by opacity/mask (that happens in the caller).
    """
    if mode == "Normal":
        return src
    if mode == "Multiply":
        return base * src
    if mode == "Screen":
        return 1.0 - (1.0 - base) * (1.0 - src)
    if mode == "Overlay":
        return torch.where(
            base <= 0.5,
            2.0 * base * src,
            1.0 - 2.0 * (1.0 - base) * (1.0 - src),
        )
    if mode == "Soft Light":
        return (1.0 - 2.0 * src) * (base * base) + 2.0 * src * base
    if mode == "Hard Light":
        return torch.where(
            src <= 0.5,
            2.0 * base * src,
            1.0 - 2.0 * (1.0 - base) * (1.0 - src),
        )
    if mode == "Color Dodge":
        denom = torch.clamp(1.0 - src, min=_EPS)
        return torch.clamp(base / denom, 0.0, 1.0)
    if mode == "Color Burn":
        denom = torch.clamp(src, min=_EPS)
        return torch.clamp(1.0 - (1.0 - base) / denom, 0.0, 1.0)
    if mode == "Pin Light":
        hi = torch.maximum(base, 2.0 * src - 1.0)
        lo = torch.minimum(base, 2.0 * src)
        return torch.where(src > 0.5, hi, lo)
    if mode == "Add":
        return torch.clamp(base + src, 0.0, 1.0)
    if mode == "Subtract":
        return torch.clamp(base - src, 0.0, 1.0)
    if mode == "Lighten":
        return torch.maximum(base, src)
    if mode == "Darken":
        return torch.minimum(base, src)
    if mode == "Difference":
        return torch.abs(base - src)
    if mode == "Difference (Squared)":
        d = base - src
        return torch.clamp(d * d, 0.0, 1.0)
    if mode == "Relativistic Addition":
        denom = torch.clamp(1.0 + base * src, min=_EPS)
        return torch.clamp((base + src) / denom, 0.0, 1.0)
    if mode == "Sigmoid":
        luma = _luminance_chw(base)
        w = torch.sigmoid(float(layer.sigmoid_strength) * (luma - float(layer.sigmoid_center)))
        if base.ndim == 3:
            w = w.unsqueeze(0)
        return base * (1.0 - w) + src * w
    if mode == "Luminosity":
        return _set_luminosity(base, src)
    # Unknown mode already normalized to "Normal" in Layer.__post_init__.
    return src


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(device=device)


def _promote_to_color(t: torch.Tensor) -> torch.Tensor:
    """Broadcast a mono (H,W) tensor to (3,H,W); pass color through."""
    if t.ndim == 2:
        return t.unsqueeze(0).expand(3, -1, -1).clone()
    if t.shape[0] == 1:
        return t.expand(3, -1, -1).clone()
    return t


# ---------------------------------------------------------------------------
# LayerStack
# ---------------------------------------------------------------------------


@dataclass
class LayerStack:
    """An ordered stack of :class:`Layer` objects.

    ``layers[0]`` is the TOP of the stack (rendered last, on top);
    ``layers[-1]`` is the BOTTOM (rendered first — the effective base
    image). :meth:`composite` flattens the whole stack top-down into a
    single float32 array.
    """

    layers: list[Layer] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.layers)

    # ---- stack operations -------------------------------------------------

    def add(self, layer: Layer, index: int = 0) -> None:
        """Insert ``layer`` at ``index`` (default: top of stack)."""
        index = max(0, min(index, len(self.layers)))
        self.layers.insert(index, layer)

    def remove(self, index: int) -> Layer:
        """Remove and return the layer at ``index``."""
        return self.layers.pop(index)

    def duplicate(self, index: int) -> Layer:
        """Duplicate the layer at ``index``, inserting the copy directly above it."""
        dup = self.layers[index].copy()
        self.layers.insert(index, dup)
        return dup

    def move(self, index: int, delta: int) -> int:
        """Move the layer at ``index`` by ``delta`` slots (negative = toward
        the top). Returns the layer's new index, or the original index if
        the move would go out of bounds.
        """
        new_index = index + delta
        if not (0 <= index < len(self.layers)) or not (0 <= new_index < len(self.layers)):
            return index
        self.layers[index], self.layers[new_index] = self.layers[new_index], self.layers[index]
        return new_index

    def merge_down(self, index: int) -> Layer:
        """Flatten the layer at ``index`` onto the layer directly below it
        (``index + 1``) into a single Normal-mode, full-opacity, unmasked
        layer that replaces both. Returns the merged layer.

        The pair is baked against whatever remains further down the stack
        (``layers[index + 2:]``) so the overall composite is unchanged by
        the merge — blend modes like Screen/Multiply depend on what's
        underneath, so baking against a blank canvas instead would silently
        alter the picture whenever anything real sits below the pair.
        """
        if not (0 <= index < len(self.layers) - 1):
            raise IndexError("No layer below to merge into")
        top = self.layers[index]
        bottom = self.layers[index + 1]
        # Merging should ignore visibility toggles on the two participants —
        # you're explicitly asking to combine them.
        top_v = top if top.visible else dataclasses.replace(top, visible=True)
        bottom_v = bottom if bottom.visible else dataclasses.replace(bottom, visible=True)
        below = self.layers[index + 2 :]
        under = self.composite(layers=below) if below else None
        merged_pixels = self.composite(layers=[top_v, bottom_v], base=under)
        assert merged_pixels is not None  # top_v/bottom_v are always visible
        merged = Layer(
            name=f"{bottom.name} + {top.name}",
            data=merged_pixels,
            opacity=1.0,
            blend_mode="Normal",
            visible=True,
        )
        self.layers[index + 1] = merged
        del self.layers[index]
        return merged

    def clear(self) -> None:
        self.layers.clear()

    # ---- compositing --------------------------------------------------

    def composite(
        self, layers: list[Layer] | None = None, base: np.ndarray | None = None
    ) -> np.ndarray | None:
        """Flatten the stack (or an explicit ``layers`` list) into one
        float32 [0, 1] array, honoring visibility, blend mode, opacity, and
        per-layer masks. Returns ``None`` if there is nothing visible to
        render and no ``base`` was supplied.

        The canvas size is that of the bottom-most (last) layer (or of
        ``base`` if given); other layers are resized to match. The canvas
        is promoted to 3-channel color if any visible layer (or ``base``)
        is color, otherwise it stays mono.

        ``base`` seeds the initial canvas instead of black/transparent —
        used by :meth:`merge_down` so a merged pair bakes against whatever
        is really underneath it in the stack rather than against nothing.
        """
        stack = self.layers if layers is None else layers
        visible = [layer for layer in stack if layer.visible]
        if not visible and base is None:
            return None

        dm = get_device_manager()
        device = dm.device

        is_color = any(layer.is_color for layer in visible) or (base is not None and base.ndim == 3)
        if visible:
            base_h, base_w = visible[-1].height, visible[-1].width
        else:
            assert base is not None  # guaranteed by the early-return above
            base_h, base_w = int(base.shape[-2]), int(base.shape[-1])

        out: torch.Tensor | None = None
        if base is not None:
            out = _to_tensor(base, device)
            if is_color:
                out = _promote_to_color(out)
            out = _resize_chw(out, (base_h, base_w))

        # Bottom -> top: each layer blends against everything beneath it.
        for layer in reversed(visible):
            src_t = _to_tensor(layer.data, device)
            if is_color:
                src_t = _promote_to_color(src_t)
            src_t = _resize_chw(src_t, (base_h, base_w))

            if out is None:
                out = torch.zeros_like(src_t)

            blended = _apply_blend_mode(layer.blend_mode, out, src_t, layer)

            alpha = float(layer.opacity)
            if layer.mask is not None:
                m = _to_tensor(layer.mask.data, device)
                m = _resize_chw(m, (base_h, base_w))
                alpha_map = torch.clamp(alpha * m, 0.0, 1.0)
                if out.ndim == 3:
                    alpha_map = alpha_map.unsqueeze(0)
                out = out * (1.0 - alpha_map) + blended * alpha_map
            else:
                out = out * (1.0 - alpha) + blended * alpha

        assert out is not None
        result = torch.clamp(out, 0.0, 1.0).to("cpu").contiguous().numpy()
        return result.astype(np.float32, copy=False)

    def flatten(self) -> np.ndarray | None:
        """Alias for :meth:`composite` over the whole stack."""
        return self.composite()
