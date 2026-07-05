"""Tests for the Nebula Flythrough video renderer."""

import cv2
import numpy as np
import pytest

from astraios.core.flythrough import (
    FlythroughParams,
    LayerFxParams,
    LayerTrajectoryParams,
    OverlayLayerParams,
    render_flythrough,
)

WIDTH, HEIGHT = 320, 240
FPS = 10
DURATION = 2.0
EXPECTED_FRAMES = int(round(FPS * DURATION))


def _synthetic_image(w: int = WIDTH, h: int = HEIGHT) -> np.ndarray:
    """A small color (C,H,W) image with a bright square in the center."""
    img = np.zeros((3, h, w), dtype=np.float32)
    img[:, h // 2 - 30 : h // 2 + 30, w // 2 - 30 : w // 2 + 30] = 0.7
    img[0, h // 2 - 10 : h // 2 + 10, w // 2 - 10 : w // 2 + 10] = 1.0
    return img


class TestFlythroughParams:
    def test_defaults(self):
        p = FlythroughParams()
        assert p.fps == 30
        assert p.duration == 10.0
        assert p.starless.zoom_start == 1.0
        assert p.starless.zoom_end == 6.0
        assert p.stars.blend_mode == "Screen"
        assert p.mid.zoom_start == 1.0 and p.mid.zoom_end == 1.0

    def test_fx_defaults_disabled(self):
        fx = LayerFxParams()
        assert fx.depth_warp == 0.0
        assert fx.radial_stretch == 0.0
        assert fx.zoom_blur == 0.0
        assert fx.chroma == 0.0
        assert fx.animate_effects is True


class TestBasicRender:
    def test_render_creates_readable_video(self, tmp_path):
        img = _synthetic_image()
        params = FlythroughParams(
            fps=FPS,
            duration=DURATION,
            out_width=WIDTH,
            out_height=HEIGHT,
            starless=LayerTrajectoryParams(zoom_start=1.0, zoom_end=2.5),
        )
        out_path = tmp_path / "flythrough.mp4"

        result_path = render_flythrough(img, out_path, params)

        assert result_path == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 1000  # non-trivial size

        cap = cv2.VideoCapture(str(out_path))
        try:
            assert cap.isOpened()
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert n_frames == EXPECTED_FRAMES
            assert (frame_w, frame_h) == (WIDTH, HEIGHT)

            ok, first = cap.read()
            assert ok
            assert first.shape == (HEIGHT, WIDTH, 3)
        finally:
            cap.release()

    def test_zoom_changes_framing_between_first_and_last_frame(self, tmp_path):
        img = _synthetic_image()
        params = FlythroughParams(
            fps=FPS,
            duration=DURATION,
            out_width=WIDTH,
            out_height=HEIGHT,
            starless=LayerTrajectoryParams(zoom_start=1.0, zoom_end=4.0),
        )
        out_path = tmp_path / "zoom.mp4"
        render_flythrough(img, out_path, params)

        cap = cv2.VideoCapture(str(out_path))
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok1, first = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames - 1)
            ok2, last = cap.read()
        finally:
            cap.release()

        assert ok1 and ok2
        # A deeper zoom on the last frame must produce a visibly different framing.
        assert not np.array_equal(first, last)
        diff = np.abs(first.astype(np.int16) - last.astype(np.int16))
        assert diff.mean() > 1.0

    def test_mono_input_accepted(self, tmp_path):
        img = _synthetic_image()[0]  # (H, W) mono
        params = FlythroughParams(fps=5, duration=1.0, out_width=160, out_height=120)
        out_path = tmp_path / "mono.mp4"
        result_path = render_flythrough(img, out_path, params)
        assert result_path.exists()

        cap = cv2.VideoCapture(str(out_path))
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        assert n_frames == 5

    def test_codec_falls_back_to_mp4v(self, tmp_path):
        img = _synthetic_image()
        params = FlythroughParams(
            fps=5, duration=1.0, out_width=160, out_height=120, codec="totally_bogus_codec"
        )
        out_path = tmp_path / "fallback.mp4"
        result_path = render_flythrough(img, out_path, params)
        assert result_path.exists()
        assert result_path.stat().st_size > 0


class TestParallaxCompositing:
    def test_stars_over_starless_parallax(self, tmp_path):
        h, w = HEIGHT, WIDTH
        starless = np.zeros((3, h, w), dtype=np.float32)
        starless[:, h // 2 - 40 : h // 2 + 40, w // 2 - 40 : w // 2 + 40] = 0.35

        stars = np.zeros((3, h, w), dtype=np.float32)
        stars[:, 20, 20] = 1.0
        stars[:, h - 20, w - 20] = 1.0
        stars[:, h // 2, w // 2] = 1.0

        params = FlythroughParams(
            fps=FPS,
            duration=DURATION,
            out_width=WIDTH,
            out_height=HEIGHT,
            starless=LayerTrajectoryParams(zoom_start=1.0, zoom_end=2.0),
            stars=OverlayLayerParams(
                zoom_start=1.0, zoom_end=1.3, blend_mode="Screen", opacity=1.0
            ),
        )
        out_path = tmp_path / "parallax.mp4"
        placeholder = np.zeros((3, h, w), dtype=np.float32)

        result_path = render_flythrough(
            placeholder, out_path, params, stars_layer=stars, starless_layer=starless
        )

        assert result_path.exists()
        cap = cv2.VideoCapture(str(out_path))
        try:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert n_frames == EXPECTED_FRAMES
            ok, frame = cap.read()
        finally:
            cap.release()
        assert ok
        # Stars screened on top must raise brightness above the starless-only background.
        assert frame.max() > 100

    def test_depth_warp_path_does_not_crash(self, tmp_path):
        h, w = 150, 200
        starless = np.zeros((3, h, w), dtype=np.float32)
        starless[:, h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20] = 0.6
        params = FlythroughParams(
            fps=5,
            duration=1.0,
            out_width=w,
            out_height=h,
            starless=LayerTrajectoryParams(
                zoom_start=1.5, zoom_end=2.5, fx=LayerFxParams(depth_warp=3.0)
            ),
        )
        out_path = tmp_path / "depth_warp.mp4"
        result_path = render_flythrough(starless, out_path, params)
        assert result_path.exists()
        assert result_path.stat().st_size > 0


class TestErrorHandling:
    def test_unsupported_shape_raises(self, tmp_path):
        bad = np.zeros((2, 3, 4, 5), dtype=np.float32)
        params = FlythroughParams(fps=5, duration=0.5, out_width=64, out_height=64)
        with pytest.raises(ValueError):
            render_flythrough(bad, tmp_path / "bad.mp4", params)
