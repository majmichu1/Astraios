"""Tests for SER planetary/lunar/solar stacking (astraios/core/ser_reader.py,
astraios/core/ser_stacker.py).

Builds tiny synthetic SER v3 files in tmp_path (valid 178-byte header + raw
frame data) representing a "planetary disk" that jitters slightly frame to
frame, with some frames sharp and others Gaussian-blurred, so quality
ranking and lucky-imaging selection have something meaningful to select for.
"""

from __future__ import annotations

import inspect
import struct

import cv2
import numpy as np
import pytest

from astraios.core.ser_reader import (
    SER_HEADER_SIZE,
    SERFrameReader,
    iter_ser_frames,
    read_ser_header,
)
from astraios.core.ser_stacker import (
    SERStackParams,
    compute_ser_quality_scores,
    frame_quality_score,
    stack_ser,
)

# ---------------------------------------------------------------------------
# Synthetic SER fabrication helpers
# ---------------------------------------------------------------------------


def _render_disk(
    h: int,
    w: int,
    cx: float,
    cy: float,
    radius: float,
    *,
    blur_sigma: float = 0.0,
    noise_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A synthetic "planetary disk" with a few surface features, float32 [0, 1]."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = np.where(r <= radius, 0.75, 0.04).astype(np.float32)

    # Fine surface texture (moves rigidly with the disk) so a Laplacian/gradient
    # sharpness metric has real structure to measure.
    features = ((-0.3, 0.2, 0.35), (0.25, -0.15, -0.30), (0.0, 0.4, 0.25), (-0.35, -0.35, 0.20))
    for fx, fy, amp in features:
        fr = np.sqrt((xx - (cx + fx * radius)) ** 2 + (yy - (cy + fy * radius)) ** 2)
        blob = amp * np.exp(-(fr**2) / (2.0 * (radius * 0.12) ** 2))
        img = np.where(r <= radius, img + blob, img)

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    if blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    if noise_sigma > 0 and rng is not None:
        img = img + rng.normal(0.0, noise_sigma, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _pack_mono_frame(img01: np.ndarray, pixel_depth: int, little_endian: bool) -> bytes:
    if pixel_depth <= 8:
        arr = np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)
        return arr.tobytes()
    maxval = (1 << pixel_depth) - 1
    arr = np.clip(np.round(img01 * maxval), 0, maxval).astype(np.uint16)
    dtype = "<u2" if little_endian else ">u2"
    return arr.astype(dtype).tobytes()


def _write_ser(
    path,
    *,
    width: int,
    height: int,
    color_id: int,
    pixel_depth: int,
    frames_data: list[bytes],
    little_endian: bool = True,
) -> None:
    header = bytearray(SER_HEADER_SIZE)
    header[0:14] = b"LUCAM-RECORDER"
    struct.pack_into(
        "<7i",
        header,
        14,
        0,
        color_id,
        1 if little_endian else 0,
        width,
        height,
        pixel_depth,
        len(frames_data),
    )
    struct.pack_into("<2q", header, 162, 0, 0)
    with open(path, "wb") as f:
        f.write(header)
        for fb in frames_data:
            f.write(fb)


def _make_test_clip(
    tmp_path,
    name: str,
    *,
    n_sharp: int = 8,
    n_blurred: int = 16,
    size: int = 72,
    pixel_depth: int = 8,
    little_endian: bool = True,
    jitter: float = 3.0,
    seed: int = 0,
):
    """Mono SER: ``n_sharp`` sharp frames followed by ``n_blurred`` blurred
    ones, each with a small random position jitter (simulated seeing wobble).

    Returns (path, sharp_indices, blurred_indices).
    """
    rng = np.random.default_rng(seed)
    h = w = size
    cx0, cy0 = w / 2.0, h / 2.0
    radius = size * 0.3

    sharp_indices = list(range(n_sharp))
    blurred_indices = list(range(n_sharp, n_sharp + n_blurred))

    frames = []
    for i in range(n_sharp + n_blurred):
        dx, dy = rng.uniform(-jitter, jitter, size=2)
        blur = 0.0 if i in sharp_indices else 3.0
        img = _render_disk(
            h, w, cx0 + dx, cy0 + dy, radius, blur_sigma=blur, noise_sigma=0.004, rng=rng
        )
        frames.append(_pack_mono_frame(img, pixel_depth, little_endian))

    path = tmp_path / name
    _write_ser(
        path,
        width=w,
        height=h,
        color_id=0,
        pixel_depth=pixel_depth,
        frames_data=frames,
        little_endian=little_endian,
    )
    return str(path), sharp_indices, blurred_indices


def _make_bayer_clip(
    tmp_path,
    name: str,
    *,
    n_frames: int = 4,
    size: int = 64,
    pixel_depth: int = 8,
    color_id: int = 8,
    seed: int = 1,
) -> str:
    """A Bayer-tagged SER whose R=G=B for every pixel — the mosaic sampled
    from it equals the plain mono render regardless of CFA phase, which
    keeps frame synthesis simple while still exercising the debayer path."""
    rng = np.random.default_rng(seed)
    h = w = size
    cx0, cy0 = w / 2.0, h / 2.0
    radius = size * 0.3

    frames = []
    for _ in range(n_frames):
        dx, dy = rng.uniform(-2.0, 2.0, size=2)
        img = _render_disk(h, w, cx0 + dx, cy0 + dy, radius, noise_sigma=0.003, rng=rng)
        frames.append(_pack_mono_frame(img, pixel_depth, True))

    path = tmp_path / name
    _write_ser(
        path,
        width=w,
        height=h,
        color_id=color_id,
        pixel_depth=pixel_depth,
        frames_data=frames,
        little_endian=True,
    )
    return str(path)


# ---------------------------------------------------------------------------
# SER header parsing
# ---------------------------------------------------------------------------


class TestReadSerHeader:
    def test_mono8(self, tmp_path):
        path, _sharp, _blurred = _make_test_clip(
            tmp_path, "mono8.ser", n_sharp=2, n_blurred=2, size=48
        )
        header = read_ser_header(path)
        assert (header.width, header.height) == (48, 48)
        assert header.frame_count == 4
        assert header.color_id == 0
        assert header.color_name == "MONO"
        assert header.pixel_depth == 8
        assert header.bytes_per_sample == 1
        assert header.channels == 1
        assert header.data_offset == SER_HEADER_SIZE

    def test_mono16(self, tmp_path):
        path, *_ = _make_test_clip(
            tmp_path, "mono16.ser", n_sharp=2, n_blurred=2, size=40, pixel_depth=16
        )
        header = read_ser_header(path)
        assert header.pixel_depth == 16
        assert header.bytes_per_sample == 2
        assert header.frame_count == 4

    def test_bayer_color_id(self, tmp_path):
        path = _make_bayer_clip(tmp_path, "bayer_hdr.ser", n_frames=3, size=32, color_id=8)
        header = read_ser_header(path)
        assert header.color_id == 8
        assert header.color_name == "BAYER_RGGB"
        assert header.channels == 1  # mosaic stored as a single plane
        assert header.frame_count == 3
        assert header.is_bayer

    def test_big_endian_16bit_flag(self, tmp_path):
        path, *_ = _make_test_clip(
            tmp_path,
            "be16.ser",
            n_sharp=2,
            n_blurred=2,
            size=32,
            pixel_depth=16,
            little_endian=False,
        )
        header = read_ser_header(path)
        assert header.little_endian is False
        assert header.pixel_depth == 16

    def test_rejects_truncated_file(self, tmp_path):
        path = tmp_path / "bad.ser"
        path.write_bytes(b"\x00" * 10)
        with pytest.raises(ValueError):
            read_ser_header(path)


# ---------------------------------------------------------------------------
# Frame streaming
# ---------------------------------------------------------------------------


class TestIterSerFrames:
    def test_mono_shapes_count_and_range(self, tmp_path):
        path, _sharp, _blurred = _make_test_clip(
            tmp_path, "iter_mono.ser", n_sharp=3, n_blurred=3, size=48
        )
        frames = list(iter_ser_frames(path))
        assert len(frames) == 6
        for f in frames:
            assert f.shape == (48, 48)
            assert f.dtype == np.float32
            assert f.min() >= 0.0
            assert f.max() <= 1.0

    def test_bayer_debayers_to_chw_color(self, tmp_path):
        path = _make_bayer_clip(tmp_path, "iter_bayer.ser", n_frames=4, size=64, color_id=9)  # GRBG
        frames = list(iter_ser_frames(path))
        assert len(frames) == 4
        for f in frames:
            assert f.shape == (3, 64, 64)
            assert f.dtype == np.float32
            assert f.min() >= 0.0
            assert f.max() <= 1.0

    def test_16bit_roundtrip_values(self, tmp_path):
        path, *_ = _make_test_clip(
            tmp_path, "iter16.ser", n_sharp=2, n_blurred=2, size=32, pixel_depth=16
        )
        frames = list(iter_ser_frames(path))
        assert len(frames) == 4
        assert all(f.shape == (32, 32) for f in frames)

    def test_is_a_generator_function(self):
        # Streaming contract: frames are produced lazily, not materialized as a list.
        assert inspect.isgeneratorfunction(iter_ser_frames)

    def test_reader_reads_single_frames_without_loading_whole_file(self, tmp_path):
        path, *_ = _make_test_clip(tmp_path, "bounded.ser", n_sharp=4, n_blurred=4, size=48)
        with SERFrameReader(path) as reader:
            assert len(reader) == 8
            one = reader.read_frame(0)
            # One frame is a small fraction of the whole clip's frame data —
            # read_frame only ever touches header.frame_bytes worth of disk.
            # float32 upconvert of 1-byte samples is at most a 4x size increase.
            assert one.nbytes <= reader.header.frame_bytes * 4
            assert one.nbytes < reader.header.frame_bytes * len(reader)


# ---------------------------------------------------------------------------
# Quality ranking
# ---------------------------------------------------------------------------


class TestQualityScore:
    def test_sharp_scores_higher_than_blurred(self):
        rng = np.random.default_rng(0)
        sharp = _render_disk(64, 64, 32, 32, radius=20, noise_sigma=0.003, rng=rng)
        blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)
        assert frame_quality_score(sharp) > frame_quality_score(blurred)

    def test_empty_frame_does_not_crash(self):
        empty = np.zeros((32, 32), dtype=np.float32)
        score = frame_quality_score(empty)
        assert np.isfinite(score)


class TestQualityRanking:
    def test_sharp_frames_rank_above_blurred(self, tmp_path):
        path, sharp_idx, blurred_idx = _make_test_clip(
            tmp_path, "rank.ser", n_sharp=8, n_blurred=16, size=72, seed=3
        )
        scores = compute_ser_quality_scores(path)
        assert scores.shape == (24,)

        top8 = set(np.argsort(-scores)[:8].tolist())
        assert top8 == set(sharp_idx)
        assert scores[sharp_idx].mean() > scores[blurred_idx].mean()


# ---------------------------------------------------------------------------
# Full stacking pipeline
# ---------------------------------------------------------------------------


class TestStackSer:
    def test_keep_percent_selects_good_frames_and_beats_naive_average(self, tmp_path):
        n_sharp, n_blurred = 8, 16
        path, sharp_idx, blurred_idx = _make_test_clip(
            tmp_path, "stack.ser", n_sharp=n_sharp, n_blurred=n_blurred, size=80, seed=5
        )

        keep_percent = 100.0 * n_sharp / (n_sharp + n_blurred)
        params = SERStackParams(keep_percent=keep_percent, integration_method="average")
        result = stack_ser(path, params)

        assert result.shape == (80, 80)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Naive baseline: unaligned mean of every frame (sharp + blurred, jittered).
        naive = np.mean(np.stack(list(iter_ser_frames(path)), axis=0), axis=0).astype(np.float32)

        assert frame_quality_score(result) > frame_quality_score(naive)

    def test_median_integration(self, tmp_path):
        path, *_ = _make_test_clip(tmp_path, "median.ser", n_sharp=4, n_blurred=4, size=48, seed=7)
        result = stack_ser(path, SERStackParams(keep_percent=50.0, integration_method="median"))
        assert result.shape == (48, 48)
        assert np.isfinite(result).all()

    def test_sigma_clip_integration(self, tmp_path):
        path, *_ = _make_test_clip(tmp_path, "sigclip.ser", n_sharp=4, n_blurred=4, size=48, seed=9)
        result = stack_ser(path, SERStackParams(keep_percent=50.0, integration_method="sigma_clip"))
        assert result.shape == (48, 48)
        assert np.isfinite(result).all()

    def test_centroid_alignment(self, tmp_path):
        path, *_ = _make_test_clip(
            tmp_path, "centroid.ser", n_sharp=4, n_blurred=4, size=48, seed=11
        )
        result = stack_ser(path, SERStackParams(keep_percent=50.0, alignment_method="centroid"))
        assert result.shape == (48, 48)

    def test_no_alignment(self, tmp_path):
        path, *_ = _make_test_clip(
            tmp_path, "noalign.ser", n_sharp=4, n_blurred=4, size=48, seed=12
        )
        result = stack_ser(path, SERStackParams(keep_percent=50.0, alignment_method="none"))
        assert result.shape == (48, 48)

    def test_16bit_mono_stacks(self, tmp_path):
        path, *_ = _make_test_clip(
            tmp_path, "s16.ser", n_sharp=3, n_blurred=3, size=40, pixel_depth=16, seed=13
        )
        result = stack_ser(path, SERStackParams(keep_percent=50.0))
        assert result.shape == (40, 40)

    def test_bayer_color_stacks(self, tmp_path):
        path = _make_bayer_clip(tmp_path, "sbayer.ser", n_frames=6, size=48, color_id=10)  # GBRG
        result = stack_ser(path, SERStackParams(keep_percent=50.0))
        assert result.shape == (3, 48, 48)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_roi_crop_applied(self, tmp_path):
        path, *_ = _make_test_clip(tmp_path, "roi.ser", n_sharp=3, n_blurred=3, size=64, seed=19)
        result = stack_ser(path, SERStackParams(keep_percent=50.0, roi=(8, 8, 32, 32)))
        assert result.shape == (32, 32)

    def test_drizzle_scale_upsamples_canvas(self, tmp_path):
        path, *_ = _make_test_clip(
            tmp_path, "drizzle.ser", n_sharp=3, n_blurred=3, size=40, seed=21
        )
        result = stack_ser(path, SERStackParams(keep_percent=50.0, drizzle_scale=2.0))
        assert result.shape == (80, 80)

    def test_best_stack_reference_mode(self, tmp_path):
        path, *_ = _make_test_clip(
            tmp_path, "refstack.ser", n_sharp=6, n_blurred=6, size=48, seed=23
        )
        result = stack_ser(
            path,
            SERStackParams(keep_percent=50.0, reference_mode="best_stack", reference_count=3),
        )
        assert result.shape == (48, 48)

    def test_color_mode_force_mono(self, tmp_path):
        path = _make_bayer_clip(tmp_path, "forcemono.ser", n_frames=4, size=32, color_id=11)  # BGGR
        result = stack_ser(path, SERStackParams(keep_percent=50.0, color_mode="mono"))
        assert result.shape == (32, 32)

    def test_invalid_alignment_method_rejected(self):
        with pytest.raises(ValueError):
            SERStackParams(alignment_method="bogus")

    def test_max_frames_caps_analysis(self, tmp_path):
        path, *_ = _make_test_clip(tmp_path, "capped.ser", n_sharp=4, n_blurred=4, size=32, seed=29)
        scores = compute_ser_quality_scores(path, SERStackParams(max_frames=3))
        assert scores.shape == (3,)
