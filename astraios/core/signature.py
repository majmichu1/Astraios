"""Signature / Insert — text watermark or image logo compositing.

Bakes a text signature or an image logo (PNG alpha preserved) onto an image
at a chosen corner/position with margin, scale, rotation, and opacity.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

# cv2 Hershey font families available for text signatures.
FONT_FACES: dict[str, int] = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "plain": cv2.FONT_HERSHEY_PLAIN,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "complex_small": cv2.FONT_HERSHEY_COMPLEX_SMALL,
    "script_simplex": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "script_complex": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
}

# Valid anchor positions for both text and image-logo overlays.
POSITIONS = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)


@dataclass
class SignatureParams:
    """All settings for one signature/insert bake."""

    mode: str = "text"  # "text" draws `text`; "image" composites `image_path`
    text: str = ""  # text string to draw when mode == "text"
    font_face: str = "simplex"  # cv2 Hershey family key, one of FONT_FACES
    font_size: int = 32  # approximate rendered glyph height in pixels
    bold: bool = False  # thicken the stroke to fake a bold weight
    italic: bool = False  # slant glyphs via cv2.FONT_ITALIC
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)  # fill RGB, each in [0,1]
    outline_color: tuple[float, float, float] | None = None  # stroke RGB, None disables outline
    outline_width: int = 0  # stroke width in pixels, 0 disables outline
    image_path: str | None = None  # PNG/JPG path drawn when mode == "image" (alpha preserved)
    position: str = "bottom_right"  # anchor corner, one of POSITIONS
    margin_x: int = 20  # horizontal margin from the anchor edge, in pixels
    margin_y: int = 20  # vertical margin from the anchor edge, in pixels
    scale: float = 100.0  # overlay scale, percent of natural rendered size
    rotation: float = 0.0  # rotation angle in degrees, clockwise
    opacity: float = 100.0  # overlay opacity, percent (0 = fully invisible / no-op)


# ---------------------------------------------------------------------------
# Position anchoring
# ---------------------------------------------------------------------------


def _anchor_point(
    base_w: int, base_h: int, ins_w: int, ins_h: int, key: str, mx: int, my: int
) -> tuple[float, float]:
    """Top-left (x, y) placement of an `ins_w`x`ins_h` overlay on a base canvas."""
    left = float(mx)
    right = float(base_w - ins_w - mx)
    top = float(my)
    bottom = float(base_h - ins_h - my)
    cx = (base_w - ins_w) / 2.0
    cy = (base_h - ins_h) / 2.0
    table = {
        "top_left": (left, top),
        "top_center": (cx, top),
        "top_right": (right, top),
        "middle_left": (left, cy),
        "center": (cx, cy),
        "middle_right": (right, cy),
        "bottom_left": (left, bottom),
        "bottom_center": (cx, bottom),
        "bottom_right": (right, bottom),
    }
    return table.get(key, table["bottom_right"])


# ---------------------------------------------------------------------------
# Overlay builders — both return HxWx4 float32 RGBA in [0,1]
# ---------------------------------------------------------------------------


def _load_image_overlay(path: str | None) -> np.ndarray:
    if not path:
        raise ValueError("SignatureParams.image_path is required when mode='image'.")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load signature logo image: {path}")

    if img.ndim == 2:
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = np.ones(img.shape[:2], dtype=np.float32)
    elif img.shape[2] == 4:
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = img[:, :, 3].astype(np.float32) / 255.0
    else:
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        alpha = np.ones(img.shape[:2], dtype=np.float32)

    rgb_f = rgb.astype(np.float32) / 255.0
    return np.dstack([rgb_f, alpha]).astype(np.float32)


def _font_scale_for_height(font_face: int, target_px: float, thickness: int) -> float:
    """Calibrate a cv2 fontScale so rendered glyph height approximates `target_px`."""
    (_, h), baseline = cv2.getTextSize("Hg", font_face, 1.0, thickness)
    total = h + baseline
    if total <= 0:
        return 1.0
    return max(0.05, float(target_px) / float(total))


def _render_text_overlay(params: SignatureParams) -> np.ndarray:
    text = params.text or ""
    if not text.strip():
        return np.zeros((0, 0, 4), dtype=np.float32)

    base_face = FONT_FACES.get(params.font_face, cv2.FONT_HERSHEY_SIMPLEX)
    font_face = base_face | cv2.FONT_ITALIC if params.italic else base_face
    thickness = max(1, int(round(params.font_size / 18.0))) + (2 if params.bold else 0)
    font_scale = _font_scale_for_height(base_face, params.font_size, thickness)

    has_outline = params.outline_color is not None and params.outline_width > 0
    outline_w = max(0, int(params.outline_width)) if has_outline else 0

    lines = text.splitlines() or [text]
    (_, base_h), baseline = cv2.getTextSize("Hg", font_face, font_scale, thickness + outline_w)
    line_h = base_h + baseline + 6
    line_sizes = [
        cv2.getTextSize(line, font_face, font_scale, thickness + outline_w)[0] for line in lines
    ]
    text_w = max((sz[0] for sz in line_sizes), default=1)

    pad = outline_w + 4
    canvas_w = max(1, text_w + 2 * pad)
    canvas_h = max(1, line_h * len(lines) + 2 * pad)

    color_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    alpha_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    def _to_bgr255(rgb: tuple[float, float, float]) -> tuple[int, int, int]:
        r, g, b = (int(round(float(np.clip(c, 0.0, 1.0)) * 255.0)) for c in rgb)
        return (b, g, r)

    fill_bgr = _to_bgr255(params.color)
    outline_bgr = (
        _to_bgr255(params.outline_color) if has_outline and params.outline_color else None
    )

    for i, line in enumerate(lines):
        org = (pad, pad + base_h + i * line_h)
        if outline_w > 0 and outline_bgr is not None:
            cv2.putText(
                color_canvas, line, org, font_face, font_scale, outline_bgr,
                thickness + 2 * outline_w, cv2.LINE_AA,
            )
            cv2.putText(
                alpha_canvas, line, org, font_face, font_scale, (255,),
                thickness + 2 * outline_w, cv2.LINE_AA,
            )
        cv2.putText(
            color_canvas, line, org, font_face, font_scale, fill_bgr, thickness, cv2.LINE_AA
        )
        cv2.putText(
            alpha_canvas, line, org, font_face, font_scale, (255,), thickness, cv2.LINE_AA
        )

    color_rgb = cv2.cvtColor(color_canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = alpha_canvas.astype(np.float32) / 255.0
    return np.dstack([color_rgb, alpha]).astype(np.float32)


# ---------------------------------------------------------------------------
# Overlay transforms
# ---------------------------------------------------------------------------


def _scale_overlay(overlay: np.ndarray, scale_pct: float) -> np.ndarray:
    scale = max(0.01, float(scale_pct) / 100.0)
    if abs(scale - 1.0) < 1e-6:
        return overlay
    h, w = overlay.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(overlay, (new_w, new_h), interpolation=interp)


def _rotate_rgba(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg % 360.0) < 1e-6 or img.shape[0] == 0 or img.shape[1] == 0:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    # Positive angle rotates clockwise (matches SASpro's QTransform.rotate convention).
    m = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = max(1, int(h * sin + w * cos))
    new_h = max(1, int(h * cos + w * sin))
    m[0, 2] += (new_w / 2.0) - cx
    m[1, 2] += (new_h / 2.0) - cy
    return cv2.warpAffine(
        img, m, (new_w, new_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def insert_signature(data: np.ndarray, params: SignatureParams) -> np.ndarray:
    """Composite a text or image-logo signature onto `data` and return the result.

    `data` is an Astraios-format float32 array: mono (H,W) or color (C,H,W).
    Only the pixels covered by the (scaled/rotated) overlay are modified;
    everything else is returned unchanged. `params.opacity == 0` is a no-op.
    """
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        _, h, w = arr.shape
    else:
        raise ValueError(f"Unsupported image shape for signature insert: {arr.shape}")

    result = arr.copy()
    opacity = float(np.clip(params.opacity / 100.0, 0.0, 1.0))
    if opacity <= 0.0 or w <= 0 or h <= 0:
        return result

    overlay_rgba = (
        _load_image_overlay(params.image_path)
        if params.mode == "image"
        else _render_text_overlay(params)
    )
    if overlay_rgba.size == 0:
        return result

    overlay_rgba = _scale_overlay(overlay_rgba, params.scale)
    overlay_rgba = _rotate_rgba(overlay_rgba, params.rotation)
    ov_h, ov_w = overlay_rgba.shape[:2]
    if ov_h <= 0 or ov_w <= 0:
        return result

    x0f, y0f = _anchor_point(w, h, ov_w, ov_h, params.position, params.margin_x, params.margin_y)
    x0, y0 = int(round(x0f)), int(round(y0f))
    x1, y1 = x0 + ov_w, y0 + ov_h

    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x1), min(h, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return result  # overlay fully off-canvas

    ox0, oy0 = cx0 - x0, cy0 - y0
    ox1, oy1 = ox0 + (cx1 - cx0), oy0 + (cy1 - cy0)

    ov_rgb = overlay_rgba[oy0:oy1, ox0:ox1, :3]
    ov_alpha = overlay_rgba[oy0:oy1, ox0:ox1, 3] * opacity  # (h, w)

    if arr.ndim == 2:
        ov_luma = (0.299 * ov_rgb[..., 0] + 0.587 * ov_rgb[..., 1] + 0.114 * ov_rgb[..., 2]).astype(
            np.float32
        )
        region = result[cy0:cy1, cx0:cx1]
        result[cy0:cy1, cx0:cx1] = region * (1.0 - ov_alpha) + ov_luma * ov_alpha
    else:
        c = result.shape[0]
        if c >= 3:
            alpha3 = ov_alpha[None, :, :]
            ov_chw = ov_rgb.transpose(2, 0, 1)
            region = result[:3, cy0:cy1, cx0:cx1]
            result[:3, cy0:cy1, cx0:cx1] = region * (1.0 - alpha3) + ov_chw * alpha3
        else:
            ov_luma = (
                0.299 * ov_rgb[..., 0] + 0.587 * ov_rgb[..., 1] + 0.114 * ov_rgb[..., 2]
            ).astype(np.float32)
            for ch in range(c):
                region = result[ch, cy0:cy1, cx0:cx1]
                result[ch, cy0:cy1, cx0:cx1] = region * (1.0 - ov_alpha) + ov_luma * ov_alpha

    return np.clip(result, 0.0, 1.0).astype(np.float32)
