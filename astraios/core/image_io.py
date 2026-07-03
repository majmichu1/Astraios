"""Image I/O — FITS, XISF, and common image format loading/saving."""

from __future__ import annotations

import logging
import struct
import xml.etree.ElementTree as ET
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from PIL import Image

log = logging.getLogger(__name__)


class FrameType(Enum):
    LIGHT = auto()
    DARK = auto()
    FLAT = auto()
    BIAS = auto()
    MASTER_DARK = auto()
    MASTER_FLAT = auto()
    MASTER_BIAS = auto()
    CALIBRATED = auto()
    ALIGNED = auto()
    RESULT = auto()
    UNKNOWN = auto()


class ImageData:
    """Container for astronomical image data with metadata."""

    def __init__(
        self,
        data: np.ndarray,
        header: dict[str, Any] | None = None,
        file_path: Path | None = None,
        frame_type: FrameType = FrameType.UNKNOWN,
    ):
        self.data = data  # float32 normalized to [0, 1], shape: (H, W) or (C, H, W)
        self.header = header or {}
        self.file_path = file_path
        self.frame_type = frame_type

    @property
    def is_color(self) -> bool:
        return self.data.ndim == 3

    @property
    def channels(self) -> int:
        return self.data.shape[0] if self.is_color else 1

    @property
    def height(self) -> int:
        return self.data.shape[-2]

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def shape_str(self) -> str:
        if self.is_color:
            return f"{self.width}x{self.height} ({self.channels}ch)"
        return f"{self.width}x{self.height} (mono)"

    @property
    def exposure(self) -> float | None:
        return self.header.get("EXPTIME") or self.header.get("EXPOSURE")

    @property
    def temperature(self) -> float | None:
        return self.header.get("CCD-TEMP") or self.header.get("SET-TEMP")

    @property
    def filter_name(self) -> str | None:
        return self.header.get("FILTER")

    @property
    def bayer_pattern(self) -> str | None:
        return self.header.get("BAYERPAT")

    def to_display(self, stretch: bool = True) -> np.ndarray:
        """Convert to uint8 RGB array for display (H, W, 3)."""
        if self.is_color:
            # (C, H, W) -> (H, W, C)
            img = np.transpose(self.data, (1, 2, 0))
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
        else:
            img = np.stack([self.data, self.data, self.data], axis=-1)

        if stretch:
            img = auto_stretch_for_display(img)

        return np.clip(img * 255, 0, 255).astype(np.uint8)


def auto_stretch_for_display(img: np.ndarray) -> np.ndarray:
    """Quick auto-stretch for preview display using median + MAD.

    Stats are computed on a thumbnail (≤1024px longest side) for speed,
    then applied to the full-resolution image.
    """
    h, w = img.shape[0], img.shape[1]
    max_side = 1024
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        sr = max(1, h // max(1, int(h * scale)))
        sc = max(1, w // max(1, int(w * scale)))
        thumb = img[::sr, ::sc, :]
    else:
        thumb = img

    result = np.empty_like(img)
    for ch in range(img.shape[-1]):
        t_ch = thumb[..., ch]
        med = float(np.median(t_ch))
        mad = float(np.median(np.abs(t_ch - med)))
        if mad < 1e-10:
            mad = float(np.std(t_ch))
        if mad < 1e-10:
            result[..., ch] = img[..., ch]
            continue
        shadow_clip = max(0.0, med - 2.8 * mad)
        scale_factor = 1.0 / max(1e-10, 1.0 - shadow_clip)
        src = img[..., ch]
        stretched = np.clip((src - shadow_clip) * scale_factor, 0.0, 1.0)
        midtone = 0.25
        mask = stretched > 0
        mtf = np.zeros_like(stretched)
        mtf[mask] = (
            (midtone - 1.0) * stretched[mask]
            / ((2.0 * midtone - 1.0) * stretched[mask] - midtone)
        )
        result[..., ch] = np.clip(mtf, 0.0, 1.0)
    return result


def auto_stretch_for_display_ref(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Stretch img using parameters computed from ref (same channel count).

    Use this for live preview: compute stretch params from the unprocessed
    downscaled image (ref) and apply the same params to the tool result (img).
    This ensures before/after brightness matches.

    Stats are computed on a thumbnail of ref (≤1024px) for speed.
    """
    # Downsample ref for fast stat computation only
    rh, rw = ref.shape[0], ref.shape[1]
    max_side = 1024
    if max(rh, rw) > max_side:
        sr = max(1, rh // max_side)
        sc = max(1, rw // max_side)
        ref_thumb = ref[::sr, ::sc, :]
    else:
        ref_thumb = ref

    result = np.empty_like(img)
    for ch in range(img.shape[-1]):
        ref_ch = ref_thumb[..., ch]
        src_ch = img[..., ch]
        med = np.median(ref_ch)
        mad = np.median(np.abs(ref_ch - med))
        if mad < 1e-10:
            mad = np.std(ref_ch)
        if mad < 1e-10:
            result[..., ch] = src_ch
            continue
        shadow_clip = max(0.0, med - 2.8 * mad)
        scale = 1.0 / max(1e-10, 1.0 - shadow_clip)
        stretched = (src_ch - shadow_clip) * scale
        midtone = 0.25
        stretched = np.clip(stretched, 0, 1)
        mask = stretched > 0
        mtf = np.zeros_like(stretched)
        mtf[mask] = (
            (midtone - 1) * stretched[mask] / ((2 * midtone - 1) * stretched[mask] - midtone)
        )
        result[..., ch] = np.clip(mtf, 0, 1)
    return result


def _normalize_fits_tile(tile: np.ndarray, header: dict | None = None) -> np.ndarray:
    """Normalize a FITS tile slice to float32 [0, 1].

    astropy already applies BZERO/BSCALE when ``hdu.data`` is accessed (every
    caller reads ``hdu.data``), so the array is already in physical units. We
    must NOT re-apply it, and must NOT pre-cast integer data to float here:
    either one forced ``_normalize_fits_data`` down its float branch, which
    contrast-*stretches* via min-max instead of dividing by the dtype max. That
    distorted the absolute scale of any narrow-range frame — calibration
    darks/flats, short subs — (e.g. uint16 10000-30000 became 0..1 instead of
    0.15..0.46). Dispatch on the real (integer/float) dtype instead.

    Float data written by Astraios itself (CREATOR header) is already in our
    normalized convention and passes through untouched: calibrated frames
    legitimately hold small negatives (bias/dark residue, preserved on purpose
    for correct stacking statistics) and values above 1.0 (flat division on
    bright stars). Min-max stretching those on reload would silently destroy
    the linear scale — and for tile reads would stretch each tile by its own
    local range, corrupting the frame.
    """
    tile = np.asarray(tile)
    if (
        header is not None
        and tile.dtype.kind == "f"
        and str(header.get("CREATOR", "")).startswith("Astraios")
    ):
        return tile.astype(np.float32)
    return _normalize_fits_data(tile)


def _normalize_fits_data(data: np.ndarray) -> np.ndarray:
    """Normalize FITS data to float32 in [0, 1]."""
    if data.dtype.kind == "u":
        max_val = np.iinfo(data.dtype).max
        return (data / max_val).astype(np.float32)
    elif data.dtype.kind == "i":
        info = np.iinfo(data.dtype)
        return ((data.astype(np.float32) - info.min) / (info.max - info.min))
    else:
        data = data.astype(np.float32)
        dmin, dmax = float(np.min(data)), float(np.max(data))
        if dmax <= 1.0 and dmin >= 0.0:
            return data
        if dmax - dmin > 0:
            return (data - dmin) / (dmax - dmin)
        return data


def load_fits(path: Path, debayer: bool = True) -> ImageData:
    """Load a FITS file and return normalized ImageData."""
    path = Path(path)
    with fits.open(str(path), memmap=False) as hdul:
        # Find the image HDU
        img_hdu = None
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                img_hdu = hdu
                break
        if img_hdu is None:
            raise ValueError(f"No image data found in {path}")

        data = img_hdu.data.copy()
        header = dict(img_hdu.header)

    data = _normalize_fits_tile(data, header)

    # Handle axes: FITS stores data as (NAXIS3, NAXIS2, NAXIS1) for color
    if data.ndim == 2:
        pass  # mono: (H, W)
    elif data.ndim == 3:
        if data.shape[0] in (1, 3, 4):
            pass  # already (C, H, W)
        elif data.shape[2] in (1, 3, 4):
            data = np.transpose(data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    else:
        raise ValueError(f"Unexpected FITS data shape: {data.shape}")

    frame_type = _guess_frame_type(header, path)

    # Auto-debayer raw OSC frames when loaded individually
    # (stacking pipeline keeps Bayer for best SNR and debayers the final stack)
    # Pass debayer=False to skip (e.g. during alignment where mono is faster)
    if debayer and data.ndim == 2 and frame_type == FrameType.LIGHT:
        from astraios.core.debayer import debayer as _debayer
        from astraios.core.debayer import detect_bayer_pattern
        bayer = detect_bayer_pattern(header)
        if bayer:
            data = _debayer(data, pattern=bayer, method="vng")
            log.debug("Auto-debayered light frame: %s → %s", path.name, data.shape)

    return ImageData(data=data, header=header, file_path=path, frame_type=frame_type)


def save_fits(image: ImageData, path: Path, overwrite: bool = True) -> None:
    """Save ImageData to a FITS file."""
    path = Path(path)
    data = image.data.copy()

    # Scale to 32-bit float FITS
    hdu = fits.PrimaryHDU(data.astype(np.float32))

    # Copy useful header keys
    for key in (
        "EXPTIME",
        "EXPOSURE",
        "CCD-TEMP",
        "FILTER",
        "BAYERPAT",
        "DATE-OBS",
        "OBJECT",
        "TELESCOP",
        "INSTRUME",
    ):
        if key in image.header:
            try:
                hdu.header[key] = image.header[key]
            except (ValueError, KeyError):
                pass

    hdu.header["CREATOR"] = "Astraios"
    hdu.header["HISTORY"] = "Processed with Astraios"
    hdu.writeto(str(path), overwrite=overwrite)
    log.info("Saved FITS: %s", path)


def load_xisf(path: Path) -> ImageData:
    """Load an XISF file (PixInsight format).

    XISF is an XML-header + binary data format. We parse the XML header
    to find image geometry, sample format, and data location, then read
    the raw pixel block.
    """
    path = Path(path)
    with open(path, "rb") as f:
        # XISF signature: "XISF0100"
        sig = f.read(8)
        if sig != b"XISF0100":
            raise ValueError(f"Not a valid XISF file: {path}")

        # Header length and reserved
        header_len = struct.unpack("<I", f.read(4))[0]
        _reserved = f.read(4)  # reserved

        xml_bytes = f.read(header_len)
        xml_str = xml_bytes.rstrip(b"\x00").decode("utf-8")
        root = ET.fromstring(xml_str)

        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        img_elem = root.find(f".//{ns}Image")
        if img_elem is None:
            raise ValueError(f"No Image element in XISF header: {path}")

        geometry = img_elem.get("geometry", "").split(":")
        width, height = int(geometry[0]), int(geometry[1])
        n_channels = int(geometry[2]) if len(geometry) > 2 else 1

        sample_format = img_elem.get("sampleFormat", "Float32")
        dtype_map = {
            "UInt8": np.uint8,
            "UInt16": np.uint16,
            "UInt32": np.uint32,
            "Float32": np.float32,
            "Float64": np.float64,
        }
        dtype = dtype_map.get(sample_format, np.float32)

        # Find data block
        data_elem = img_elem.find(f"{ns}Data")
        if data_elem is not None:
            loc = data_elem.get("location", "")
        else:
            loc = img_elem.get("location", "")

        if loc.startswith("attachment:"):
            parts = loc.split(":")
            offset = int(parts[1])
            size = int(parts[2])
            f.seek(offset)
            raw = f.read(size)
        else:
            # inline / embedded — read from current position
            n_pixels = width * height * n_channels
            item_size = np.dtype(dtype).itemsize
            raw = f.read(n_pixels * item_size)

        data = np.frombuffer(raw, dtype=dtype).copy()

    if n_channels == 1:
        data = data.reshape(height, width)
    else:
        # XISF stores planar: channel-first
        data = data.reshape(n_channels, height, width)

    data = _normalize_fits_data(data)

    header = {}
    for prop in root.iter(f"{ns}FITSKeyword"):
        name = prop.get("name", "")
        value = prop.get("value", "")
        if name:
            header[name] = value

    return ImageData(data=data, header=header, file_path=path, frame_type=FrameType.UNKNOWN)


def _build_xisf_xml(image: ImageData) -> tuple[bytes, int]:
    """Build XISF XML header with full metadata.

    Returns:
        (xml_bytes, data_offset) — properly padded header and data offset.
    """
    data = image.data.astype(np.float32)
    if data.ndim == 2:
        h, w = data.shape
        n_ch = 1
    else:
        n_ch, h, w = data.shape

    geometry = f"{w}:{h}:{n_ch}"
    color_space = "Gray" if n_ch == 1 else "RGB"

    # Build FITS keywords
    fits_keywords = ""
    known_keys = (
        "EXPTIME",
        "EXPOSURE",
        "CCD-TEMP",
        "FILTER",
        "BAYERPAT",
        "DATE-OBS",
        "OBJECT",
        "TELESCOP",
        "INSTRUME",
        "XBINNING",
        "YBINNING",
        "FOCALLEN",
        "APTDIAM",
        "ROTATANG",
    )
    for key in known_keys:
        if key in image.header:
            val = image.header[key]
            fits_keywords += f'<FITSKeyword name="{key}" value="{val}"/>\n'

    # Always add creator
    fits_keywords += '<FITSKeyword name="CREATOR" value="Astraios"/>\n'

    # Processing history from header
    history = image.header.get("ASTRAIOS_HISTORY", "")
    history_xml = ""
    if history:
        for step in history.split("||"):
            step = step.strip()
            if step:
                history_xml += f'<Process identifier="astraios">{step}</Process>\n'

    # Color space properties
    color_props = ""
    if n_ch >= 3:
        color_props = (
            '<ColorSpace id="RGB">'
            '<Channel name="Red" sampleFormat="Float32"/>\n'
            '<Channel name="Green" sampleFormat="Float32"/>\n'
            '<Channel name="Blue" sampleFormat="Float32"/>\n'
            "</ColorSpace>"
        )

    # Image properties (resolution, etc.)
    properties = ""
    if image.exposure:
        properties += f'<Property name="Exposure Time">{image.exposure}</Property>\n'
    if image.temperature:
        properties += f'<Property name="CCD Temperature">{image.temperature}</Property>\n'
    if image.filter_name:
        properties += f'<Property name="Filter">{image.filter_name}</Property>\n'

    location = f"attachment:{{OFFSET}}:{data.nbytes}"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf">\n'
        f'  <Image geometry="{geometry}" sampleFormat="Float32" '
        f'colorSpace="{color_space}" location="{location}">\n'
        f"    {properties}"
        f"    {fits_keywords}"
        f"    {color_props}"
        f"    {history_xml}"
        "  </Image>\n"
        "</xisf>"
    )

    xml_bytes = xml.encode("utf-8")
    # Pad header to 16-byte boundary
    pad_len = (16 - (len(xml_bytes) % 16)) % 16
    xml_bytes += b"\x00" * pad_len
    data_offset = 16 + len(xml_bytes)

    # Replace offset placeholder
    offset_str = str(data_offset).encode("utf-8")
    xml_bytes = xml_bytes.replace(b"{OFFSET}", offset_str)
    # If replacement changed length, re-pad and update offset in XML
    if len(xml_bytes) % 16 != 0:
        pad_len = (16 - (len(xml_bytes) % 16)) % 16
        xml_bytes += b"\x00" * pad_len
        data_offset = 16 + len(xml_bytes)
        # Update the offset in the XML to match the new header size
        new_offset_str = str(data_offset).encode("utf-8")
        xml_bytes = xml_bytes.replace(offset_str, new_offset_str, 1)

    return xml_bytes, data_offset


def save_xisf(image: ImageData, path: Path) -> None:
    """Save ImageData to XISF format with full metadata."""
    path = Path(path)
    data = image.data.astype(np.float32)

    xml_bytes, _data_offset = _build_xisf_xml(image)

    with open(path, "wb") as f:
        f.write(b"XISF0100")
        f.write(struct.pack("<I", len(xml_bytes)))
        f.write(b"\x00" * 4)  # reserved
        f.write(xml_bytes)
        f.write(data.tobytes())

    log.info("Saved XISF: %s", path)


def load_image(path: str | Path, debayer: bool = True) -> ImageData:
    """Auto-detect format and load an image."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".fit", ".fits", ".fts"):
        return load_fits(path, debayer=debayer)
    elif suffix in (".xisf",):
        return load_xisf(path)
    elif suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        return _load_common_image(path)
    else:
        raise ValueError(f"Unsupported image format: {suffix}")


def _load_common_image(path: Path) -> ImageData:
    """Load common image formats via Pillow, with tifffile fallback for 32-bit TIFFs."""
    suffix = path.suffix.lower()

    # tifffile handles 32-bit float TIFFs (e.g. Siril stacks) that Pillow rejects
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile
            raw = tifffile.imread(str(path))
            # asarray avoids a second full-size copy when the data is already
            # float32 (e.g. Siril stacks); for integer TIFFs it converts once.
            data = np.asarray(raw, dtype=np.float32)
            if data is not raw:
                del raw  # free the source buffer immediately
            if data.ndim == 3 and data.shape[2] in (3, 4):
                # HWC → CHW, drop alpha — one contiguous copy
                data = np.ascontiguousarray(np.transpose(data[:, :, :3], (2, 0, 1)))
            # Normalize to [0, 1] IN PLACE — no extra full-size temporaries.
            data = _normalize01_inplace(data)
            return ImageData(data=data, file_path=path, frame_type=FrameType.UNKNOWN)
        except Exception:
            pass  # fall through to Pillow

    from PIL import Image

    img = Image.open(path)
    if img.mode in ("I;16", "I;16B"):
        data = np.asarray(img, dtype=np.uint16).astype(np.float32)
        data *= 1.0 / 65535.0
    elif img.mode == "I":
        data = np.asarray(img, dtype=np.int32).astype(np.float32)
        data = _normalize01_inplace(data)
    elif img.mode == "F":
        data = np.array(img, dtype=np.float32)
        data = _normalize01_inplace(data)
    else:
        img = img.convert("RGB")
        data = np.asarray(img, dtype=np.float32) * (1.0 / 255.0)
        data = np.ascontiguousarray(np.transpose(data, (2, 0, 1)))  # (H,W,C) -> (C,H,W)

    return ImageData(data=data, file_path=path, frame_type=FrameType.UNKNOWN)


def _normalize01_inplace(data: np.ndarray) -> np.ndarray:
    """Scale a float32 array to [0, 1] with no full-size temporaries.

    Operates in place when possible; copies only a read-only buffer. Returns the
    array to use (callers must assign the result).
    """
    dmin = float(data.min())
    dmax = float(data.max())
    if dmax > dmin:
        if not data.flags.writeable:
            data = data.copy()
        data -= dmin
        data *= 1.0 / (dmax - dmin)
    return data


def save_image(
    image: ImageData,
    path: str | Path,
    overwrite: bool = True,
    bit_depth: int = 16,
    jpeg_quality: int = 95,
) -> None:
    """Save ImageData to FITS, XISF, TIFF, PNG, or JPEG.

    Args:
        image: ImageData to save (normalized float32 [0, 1]).
        path: Output file path (extension determines format).
        overwrite: Whether to overwrite existing file (FITS only).
        bit_depth: Output bit depth for TIFF/PNG (8 or 16).
        jpeg_quality: Quality for JPEG (1-100, only used for .jpg/.jpeg).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".fit", ".fits", ".fts"):
        save_fits(image, path, overwrite=overwrite)
    elif suffix == ".xisf":
        save_xisf(image, path)
    elif suffix in (".tif", ".tiff"):
        _save_tiff(image, path, bit_depth=bit_depth)
    elif suffix == ".png":
        _save_png(image, path, bit_depth=bit_depth)
    elif suffix in (".jpg", ".jpeg"):
        _save_jpeg(image, path, quality=jpeg_quality)
    else:
        raise ValueError(f"Unsupported export format: {suffix}")


def _save_tiff(image: ImageData, path: Path, bit_depth: int = 16) -> None:
    """Save ImageData to 16-bit or 8-bit TIFF."""
    data = _to_display_array(image)

    if bit_depth == 16:
        img = _array_to_pil_16(data)
        img.save(str(path), compression="tiff_lzw")
    else:
        data_u8 = (data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(data_u8, mode=_tiff_mode(data_u8))
        img.save(str(path), compression="tiff_lzw")

    log.info("Saved TIFF (%d-bit): %s", bit_depth, path)


def _save_png(image: ImageData, path: Path, bit_depth: int = 16) -> None:
    """Save ImageData to PNG (8-bit or 16-bit if Pillow supports it)."""
    data = _to_display_array(image)

    if bit_depth == 16:
        img = _array_to_pil_16(data)
        try:
            img.save(str(path))
        except (OSError, ValueError):
            # Fallback to 8-bit if 16-bit not supported
            log.warning("16-bit PNG not supported, falling back to 8-bit")
            data_u8 = (data * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(data_u8, mode=_png_mode(data_u8))
            img.save(str(path))
    else:
        data_u8 = (data * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(data_u8, mode=_png_mode(data_u8))
        img.save(str(path))

    log.info("Saved PNG (%d-bit): %s", bit_depth, path)


def _array_to_pil_16(data: np.ndarray) -> Image.Image:
    """Convert a float [0,1] array to a 16-bit PIL Image.

    Handles mono and RGBA. RGB 16-bit is not well supported by Pillow,
    so we fall back to 8-bit for RGB with a warning.
    """
    data_u16 = (data * 65535).clip(0, 65535).astype(np.uint16)

    if data_u16.ndim == 2:
        # Mono
        return Image.fromarray(data_u16)
    elif data_u16.shape[2] == 2:
        # Grayscale + alpha -> save as RGBA with G in all channels
        gray = data_u16[:, :, 0]
        alpha = data_u16[:, :, 1]
        rgba = np.stack([gray, gray, gray, alpha], axis=-1)
        return Image.fromarray(rgba, mode="RGBA;16")
    else:
        # RGB or RGBA 16-bit — Pillow doesn't support this well
        # Fall back to 8-bit
        log.warning(
            "16-bit multi-channel TIFF/PNG not fully supported by Pillow, "
            "falling back to 8-bit for RGB/RGBA"
        )
        data_u8 = (data * 255).clip(0, 255).astype(np.uint8)
        if data_u8.shape[2] == 4:
            return Image.fromarray(data_u8, mode="RGBA")
        return Image.fromarray(data_u8, mode="RGB")


def _save_jpeg(image: ImageData, path: Path, quality: int = 95) -> None:
    """Save ImageData to JPEG (always 8-bit RGB)."""
    data = _to_display_array(image)
    data_u8 = (data * 255).clip(0, 255).astype(np.uint8)

    # JPEG requires RGB or L (grayscale)
    if data_u8.ndim == 2:
        # Grayscale
        img = Image.fromarray(data_u8, mode="L")
    elif data_u8.shape[2] == 1:
        img = Image.fromarray(data_u8.squeeze(-1), mode="L")
    elif data_u8.shape[2] == 2:
        # Grayscale + alpha -> drop alpha
        img = Image.fromarray(data_u8[:, :, 0], mode="L")
    elif data_u8.shape[2] >= 3:
        img = Image.fromarray(data_u8[:, :, :3], mode="RGB")
    else:
        raise ValueError(f"Cannot save {data_u8.shape} as JPEG")

    img.save(str(path), quality=quality, subsampling=0)
    log.info("Saved JPEG (quality=%d): %s", quality, path)


def _to_display_array(image: ImageData) -> np.ndarray:
    """Convert ImageData to display-ready array (H, W) or (H, W, C) uint8 [0,1]."""
    if image.is_color:
        # (C, H, W) -> (H, W, C)
        arr = np.transpose(image.data, (1, 2, 0))
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        elif arr.shape[2] == 2:
            return arr  # grayscale + alpha
        elif arr.shape[2] == 3:
            return arr
        else:
            return arr[:, :, :3]  # truncate to RGB
    else:
        return image.data


def _tiff_mode(arr: np.ndarray) -> str:
    """Return appropriate PIL mode for TIFF."""
    if arr.ndim == 2:
        return "I;16" if arr.dtype == np.uint16 else "L"
    elif arr.shape[2] == 3:
        return "RGB;16" if arr.dtype == np.uint16 else "RGB"
    elif arr.shape[2] == 4:
        return "RGBA;16" if arr.dtype == np.uint16 else "RGBA"
    return "RGB"


def _png_mode(arr: np.ndarray) -> str:
    """Return appropriate PIL mode for PNG."""
    if arr.ndim == 2:
        return "I;16" if arr.dtype == np.uint16 else "L"
    elif arr.shape[2] == 3:
        return "RGB;16" if arr.dtype == np.uint16 else "RGB"
    elif arr.shape[2] == 4:
        return "RGBA;16" if arr.dtype == np.uint16 else "RGBA"
    return "RGB"


def _guess_frame_type(header: dict, path: Path) -> FrameType:
    """Try to guess the frame type from FITS header or filename.

    Checks (in order):
    1. IMAGETYP / FRAME header keyword (most reliable)
    2. OBJECT header keyword patterns
    3. Filename patterns
    4. Exposure + readout patterns (0s exposure → bias, very short → bias/dark)
    """
    type_map = {
        "light": FrameType.LIGHT,
        "science": FrameType.LIGHT,
        "object": FrameType.LIGHT,
        "dark": FrameType.DARK,
        "flat": FrameType.FLAT,
        "bias": FrameType.BIAS,
        "offset": FrameType.BIAS,
        "calibration": FrameType.BIAS,  # rare but seen
    }

    # 1. IMAGETYP or FRAME keyword (SharpCap, SGP, NINA, MaxIm, etc.)
    for kw in ("IMAGETYP", "FRAME", "EXPTYPE", "OBSTYPE"):
        val = str(header.get(kw, "")).lower().strip()
        if val:
            for key, ft in type_map.items():
                if key in val:
                    return ft

    # 2. OBJECT keyword — sometimes set to "Dark" or "Flat" by capture software
    obj = str(header.get("OBJECT", "")).lower().strip()
    for key, ft in [("dark", FrameType.DARK), ("flat", FrameType.FLAT),
                    ("bias", FrameType.BIAS), ("offset", FrameType.BIAS)]:
        if obj == key:
            return ft

    # 3. Filename patterns (handles most common naming conventions)
    name = path.stem.lower()
    # Check longer patterns first to avoid partial matches
    for pattern, ft in [
        ("masterflat", FrameType.FLAT), ("master_flat", FrameType.FLAT),
        ("masterdark", FrameType.DARK), ("master_dark", FrameType.DARK),
        ("masterbias", FrameType.BIAS), ("master_bias", FrameType.BIAS),
        ("_flat_", FrameType.FLAT), ("_dark_", FrameType.DARK),
        ("_bias_", FrameType.BIAS), ("_light_", FrameType.LIGHT),
        ("flat", FrameType.FLAT), ("dark", FrameType.DARK),
        ("bias", FrameType.BIAS), ("offset", FrameType.BIAS),
        ("light", FrameType.LIGHT),
    ]:
        if pattern in name:
            return ft

    # 4. Heuristics from EXPTIME: 0s or very short → likely bias
    try:
        exptime = float(header.get("EXPTIME", header.get("EXPOSURE", -1)))
        if exptime == 0.0:
            return FrameType.BIAS
    except (ValueError, TypeError):
        pass

    return FrameType.UNKNOWN
