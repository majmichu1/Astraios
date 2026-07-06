"""SER (v3) planetary/lunar/solar video container reader.

Parses the SER file format used by FireCapture, SharpCap, and other planetary
capture tools, and streams frames one at a time so multi-gigabyte SER files
never need to be loaded fully into RAM.

SER v3 header (178 bytes, little-endian)::

    FileID[14]              ASCII, typically "LUCAM-RECORDER"
    LuID                    int32
    ColorID                 int32  (see SER_COLOR_NAMES)
    LittleEndian            int32  (nonzero = pixel samples are little-endian)
    ImageWidth              int32
    ImageHeight             int32
    PixelDepthPerPlane      int32  (bits per sample: 8, 10, 12, 14, 16, ...)
    FrameCount              int32
    Observer[40]            ASCII
    Instrument[40]          ASCII
    Telescope[40]           ASCII
    DateTime                int64  (.NET ticks, local time)
    DateTime_UTC            int64  (.NET ticks, UTC)

Frame data follows immediately after the header. An optional trailer of
``FrameCount`` little-endian int64 timestamps (one per frame) may follow the
frame data.

Ported and adapted from Seti Astro Suite Pro (setiastrosuitepro),
Copyright Franklin Marek, GPL-3.0-or-later.
https://github.com/setiastro/setiastrosuitepro
"""

from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from astraios.core.debayer import debayer as debayer_cfa

log = logging.getLogger(__name__)

__all__ = [
    "SER_HEADER_SIZE",
    "SER_COLOR_NAMES",
    "SERHeader",
    "SERFrameReader",
    "read_ser_header",
    "iter_ser_frames",
]

SER_HEADER_SIZE = 178
SER_SIGNATURE_LEN = 14

#: SER v3 spec ColorID values. 24-27 are non-standard but appear in the wild
#: from a handful of writers; accepted as aliases of the RGB/RGBA/BGR/BGRA IDs.
SER_COLOR_NAMES: dict[int, str] = {
    0: "MONO",
    8: "BAYER_RGGB",
    9: "BAYER_GRBG",
    10: "BAYER_GBRG",
    11: "BAYER_BGGR",
    16: "BAYER_CYYM",
    17: "BAYER_YCMY",
    18: "BAYER_YMCY",
    19: "BAYER_MYYC",
    24: "RGB",
    25: "BGR",
    100: "RGB",
    101: "BGR",
}

_STANDARD_BAYER = {"BAYER_RGGB", "BAYER_GRBG", "BAYER_GBRG", "BAYER_BGGR"}
_CMY_BAYER = {"BAYER_CYYM", "BAYER_YCMY", "BAYER_YMCY", "BAYER_MYYC"}


def _decode_cstr(b: bytes) -> str:
    try:
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class SERHeader:
    """Parsed SER v3 header plus derived layout fields."""

    file_id: str
    lu_id: int
    color_id: int
    color_name: str
    little_endian: bool
    width: int
    height: int
    pixel_depth: int
    frame_count: int
    observer: str
    instrument: str
    telescope: str
    date_time: int
    date_time_utc: int
    bytes_per_sample: int
    channels: int
    frame_bytes: int
    data_offset: int
    has_trailer: bool

    @property
    def is_bayer(self) -> bool:
        return self.color_name in _STANDARD_BAYER or self.color_name in _CMY_BAYER


def _parse_header(fh: BinaryIO) -> SERHeader:
    fh.seek(0)
    hdr = fh.read(SER_HEADER_SIZE)
    if len(hdr) < SER_HEADER_SIZE:
        raise ValueError("Not a valid SER file (header shorter than 178 bytes).")

    file_id = _decode_cstr(hdr[:SER_SIGNATURE_LEN])
    lu_id, color_id, little_endian_flag, width, height, pixel_depth, frame_count = (
        struct.unpack_from("<7i", hdr, SER_SIGNATURE_LEN)
    )
    observer = _decode_cstr(hdr[42:82])
    instrument = _decode_cstr(hdr[82:122])
    telescope = _decode_cstr(hdr[122:162])
    date_time, date_time_utc = struct.unpack_from("<2q", hdr, 162)

    if width <= 0 or height <= 0 or frame_count < 0:
        raise ValueError(
            f"Implausible SER header: width={width} height={height} frames={frame_count}"
        )

    color_name = SER_COLOR_NAMES.get(int(color_id), f"UNKNOWN({color_id})")
    channels = 3 if color_name in ("RGB", "BGR") else 1
    bytes_per_sample = 1 if pixel_depth <= 8 else 2
    frame_bytes = int(width) * int(height) * channels * bytes_per_sample

    data_offset = SER_HEADER_SIZE
    try:
        file_size = os.fstat(fh.fileno()).st_size
    except (OSError, AttributeError):
        file_size = data_offset + frame_count * frame_bytes

    data_size = frame_count * frame_bytes
    expected_with_trailer = data_offset + data_size + frame_count * 8
    has_trailer = file_size >= expected_with_trailer > data_offset

    return SERHeader(
        file_id=file_id,
        lu_id=int(lu_id),
        color_id=int(color_id),
        color_name=color_name,
        little_endian=bool(little_endian_flag),
        width=int(width),
        height=int(height),
        pixel_depth=int(pixel_depth),
        frame_count=int(frame_count),
        observer=observer,
        instrument=instrument,
        telescope=telescope,
        date_time=int(date_time),
        date_time_utc=int(date_time_utc),
        bytes_per_sample=bytes_per_sample,
        channels=channels,
        frame_bytes=frame_bytes,
        data_offset=data_offset,
        has_trailer=has_trailer,
    )


def read_ser_header(path: str | os.PathLike) -> SERHeader:
    """Parse and return the header of a SER file without reading any frames."""
    with open(os.fspath(path), "rb") as f:
        return _parse_header(f)


class SERFrameReader:
    """Streams SER frames one at a time via plain file seek+read.

    Memory-bounded: only the header (178 bytes) and one frame at a time are
    ever held in memory, regardless of file size — safe for multi-gigabyte
    planetary SER captures.

    Frames are returned float32 normalized to [0, 1], channels-first per
    Astraios convention: mono ``(H, W)``, color ``(3, H, W)`` in R, G, B
    order. Bayer-mosaic frames are demosaiced via
    :func:`astraios.core.debayer.debayer`.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._fh = open(self.path, "rb")
        self.header = _parse_header(self._fh)

    def __enter__(self) -> SERFrameReader:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            log.debug("SERFrameReader.close: file handle already closed", exc_info=True)

    def __len__(self) -> int:
        return self.header.frame_count

    def _read_raw(self, index: int) -> np.ndarray:
        h = self.header
        if index < 0 or index >= h.frame_count:
            raise IndexError(f"Frame index {index} out of range (0..{h.frame_count - 1})")

        offset = h.data_offset + index * h.frame_bytes
        self._fh.seek(offset)
        buf = self._fh.read(h.frame_bytes)
        if len(buf) != h.frame_bytes:
            raise OSError(
                f"Truncated SER frame {index}: expected {h.frame_bytes} bytes, got {len(buf)}"
            )

        dtype: np.dtype | type = np.uint8 if h.bytes_per_sample == 1 else np.dtype("<u2")
        arr = np.frombuffer(buf, dtype=dtype)
        if h.bytes_per_sample == 2 and not h.little_endian:
            arr = arr.byteswap()

        if h.channels == 1:
            arr = arr.reshape(h.height, h.width)
        else:
            arr = arr.reshape(h.height, h.width, h.channels)
        return arr

    def read_frame(
        self,
        index: int,
        *,
        debayer: bool = True,
        bayer_pattern: str | None = None,
        debayer_method: str = "bilinear",
    ) -> np.ndarray:
        """Read and normalize frame ``index``.

        Returns float32 [0, 1]; mono ``(H, W)`` or color ``(3, H, W)``.
        Bayer frames are demosaiced unless ``debayer=False`` (in which case
        the raw mosaic is returned as mono).
        """
        h = self.header
        raw = self._read_raw(index)

        if raw.dtype == np.uint8:
            arr = raw.astype(np.float32) / 255.0
        else:
            if 8 < h.pixel_depth < 16:
                denom = float((1 << h.pixel_depth) - 1)
            else:
                denom = 65535.0
            arr = raw.astype(np.float32) / denom

        if h.color_name == "RGB":
            return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)), dtype=np.float32)
        if h.color_name == "BGR":
            arr = arr[..., ::-1]
            return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)), dtype=np.float32)

        # Mono or Bayer mosaic — arr is (H, W).
        if h.color_name in _STANDARD_BAYER and debayer:
            pattern = (bayer_pattern or h.color_name.split("_", 1)[1]).upper()
            if pattern not in ("RGGB", "BGGR", "GRBG", "GBRG"):
                log.warning(
                    "Frame %d: unrecognized Bayer override %r, returning raw mosaic as mono",
                    index, pattern,
                )
                return np.ascontiguousarray(arr, dtype=np.float32)
            return debayer_cfa(arr, pattern=pattern, method=debayer_method)

        if h.color_name in _CMY_BAYER and debayer:
            log.warning(
                "Frame %d: CMY-style Bayer pattern %s is not supported by the RGB "
                "demosaic path; returning raw mosaic as mono.",
                index, h.color_name,
            )

        return np.ascontiguousarray(arr, dtype=np.float32)

    def read_timestamp(self, index: int) -> int | None:
        """Return the trailer timestamp (.NET ticks) for ``index``, or None if absent."""
        h = self.header
        if not h.has_trailer:
            return None
        if index < 0 or index >= h.frame_count:
            raise IndexError(f"Frame index {index} out of range (0..{h.frame_count - 1})")
        ts_offset = h.data_offset + h.frame_count * h.frame_bytes + index * 8
        self._fh.seek(ts_offset)
        buf = self._fh.read(8)
        if len(buf) != 8:
            return None
        (value,) = struct.unpack("<q", buf)
        return int(value)

    def __iter__(self):
        for i in range(self.header.frame_count):
            yield self.read_frame(i)


def iter_ser_frames(
    path: str | os.PathLike,
    *,
    debayer: bool = True,
    bayer_pattern: str | None = None,
    debayer_method: str = "bilinear",
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
):
    """Stream frames ``[start, stop)`` from a SER file without loading it into RAM.

    Opens the file, yields normalized float32 frames one at a time, and
    closes it when the generator is exhausted or garbage-collected.
    """
    with SERFrameReader(path) as reader:
        n = len(reader)
        stop_i = n if stop is None else min(int(stop), n)
        for i in range(int(start), stop_i, int(step)):
            yield reader.read_frame(
                i, debayer=debayer, bayer_pattern=bayer_pattern, debayer_method=debayer_method
            )
