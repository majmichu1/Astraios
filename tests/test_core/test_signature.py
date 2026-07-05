"""Tests for signature/watermark insertion."""

import cv2
import numpy as np
import pytest

from astraios.core.signature import POSITIONS, SignatureParams, insert_signature

WIDTH, HEIGHT = 150, 100


def _color_image(w: int = WIDTH, h: int = HEIGHT) -> np.ndarray:
    return np.full((3, h, w), 0.2, dtype=np.float32)


def _mono_image(w: int = WIDTH, h: int = HEIGHT) -> np.ndarray:
    return np.full((h, w), 0.2, dtype=np.float32)


def _changed_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    if before.ndim == 2:
        return np.abs(after - before) > 1e-6
    return np.abs(after - before).sum(axis=0) > 1e-6


class TestSignatureParams:
    def test_defaults(self):
        p = SignatureParams()
        assert p.mode == "text"
        assert p.position == "bottom_right"
        assert p.scale == 100.0
        assert p.opacity == 100.0
        assert p.rotation == 0.0
        assert p.outline_color is None


class TestTextSignature:
    def test_text_changes_pixels_only_near_chosen_corner(self):
        img = _color_image()
        h, w = img.shape[1], img.shape[2]

        tl_params = SignatureParams(
            mode="text", text="Astraios", font_size=18, position="top_left",
            margin_x=5, margin_y=5,
        )
        br_params = SignatureParams(
            mode="text", text="Astraios", font_size=18, position="bottom_right",
            margin_x=5, margin_y=5,
        )
        out_tl = insert_signature(img, tl_params)
        out_br = insert_signature(img, br_params)

        changed_tl = _changed_mask(img, out_tl)
        changed_br = _changed_mask(img, out_br)
        assert changed_tl.any() and changed_br.any(), "expected some pixels to change"

        ys_tl, xs_tl = np.where(changed_tl)
        ys_br, xs_br = np.where(changed_br)

        # The bottom-right placement must sit well below and to the right of top-left.
        assert ys_br.mean() > ys_tl.mean() + h * 0.2
        assert xs_br.mean() > xs_tl.mean() + w * 0.2
        # Neither placement should touch the opposite corner region.
        assert ys_tl.max() < h * 0.6
        assert xs_tl.max() < w * 0.9

    def test_opacity_zero_is_noop(self):
        img = _color_image()
        params = SignatureParams(mode="text", text="Watermark", opacity=0.0)
        out = insert_signature(img, params)
        assert np.array_equal(out, img)

    def test_empty_text_is_noop(self):
        img = _color_image()
        params = SignatureParams(mode="text", text="   ")
        out = insert_signature(img, params)
        assert np.array_equal(out, img)

    def test_mono_image_supported(self):
        img = _mono_image()
        params = SignatureParams(mode="text", text="M", font_size=16, position="top_left")
        out = insert_signature(img, params)
        assert out.shape == img.shape
        assert out.ndim == 2
        assert (out != img).any()

    def test_color_image_supported(self):
        img = _color_image()
        params = SignatureParams(mode="text", text="C", font_size=16, color=(1.0, 0.0, 0.0))
        out = insert_signature(img, params)
        assert out.shape == img.shape
        # Red channel should rise more than green/blue somewhere under the text.
        assert out[0].max() > img[0].max()

    def test_bold_and_outline_do_not_crash(self):
        img = _color_image()
        params = SignatureParams(
            mode="text", text="Bold", font_size=24, bold=True, italic=True,
            outline_color=(0.0, 0.0, 0.0), outline_width=2,
        )
        out = insert_signature(img, params)
        assert out.shape == img.shape
        assert (out != img).any()

    def test_rotation_does_not_crash(self):
        img = _color_image()
        params = SignatureParams(mode="text", text="Spin", rotation=30.0, position="center")
        out = insert_signature(img, params)
        assert out.shape == img.shape

    def test_scale_increases_affected_area(self):
        img = _color_image()
        small = SignatureParams(mode="text", text="Astraios", font_size=12, scale=50.0)
        big = SignatureParams(mode="text", text="Astraios", font_size=12, scale=200.0)

        out_small = insert_signature(img, small)
        out_big = insert_signature(img, big)

        area_small = _changed_mask(img, out_small).sum()
        area_big = _changed_mask(img, out_big).sum()
        assert area_big > area_small

    @pytest.mark.parametrize("position", POSITIONS)
    def test_all_positions_valid(self, position):
        img = _color_image()
        params = SignatureParams(
            mode="text", text="P", font_size=14, position=position, margin_x=3, margin_y=3
        )
        out = insert_signature(img, params)
        assert out.shape == img.shape
        assert np.all(np.isfinite(out))
        assert out.min() >= 0.0 and out.max() <= 1.0


class TestImageLogoSignature:
    def _make_logo(self, tmp_path, size=20, color_bgr=(0, 0, 255)):
        logo = np.zeros((size, size, 4), dtype=np.uint8)
        logo[:, :, :3] = color_bgr
        logo[:, :, 3] = 255
        path = tmp_path / "logo.png"
        cv2.imwrite(str(path), logo)
        return path

    def test_image_logo_composites(self, tmp_path):
        logo_path = self._make_logo(tmp_path, color_bgr=(0, 0, 255))  # BGR red
        img = _color_image()
        params = SignatureParams(
            mode="image", image_path=str(logo_path), position="top_left",
            margin_x=2, margin_y=2, scale=100.0,
        )
        out = insert_signature(img, params)
        # Red channel (index 0 in RGB CHW) should be raised near the logo.
        assert out[0, 5, 5] > img[0, 5, 5]
        assert out[0, 5, 5] > out[2, 5, 5]

    def test_image_logo_missing_path_raises(self):
        img = _color_image()
        params = SignatureParams(mode="image", image_path=None)
        with pytest.raises(ValueError):
            insert_signature(img, params)

    def test_image_logo_bad_path_raises(self):
        img = _color_image()
        params = SignatureParams(mode="image", image_path="/nonexistent/path/logo.png")
        with pytest.raises(ValueError):
            insert_signature(img, params)

    def test_image_logo_mono_target(self, tmp_path):
        logo_path = self._make_logo(tmp_path, color_bgr=(255, 255, 255))
        img = _mono_image()
        params = SignatureParams(mode="image", image_path=str(logo_path), position="bottom_right")
        out = insert_signature(img, params)
        assert out.shape == img.shape
        assert (out != img).any()

    def test_opacity_partial_blends(self, tmp_path):
        logo_path = self._make_logo(tmp_path, color_bgr=(255, 255, 255))
        img = _color_image()
        common = {
            "mode": "image",
            "image_path": str(logo_path),
            "position": "top_left",
            "margin_x": 5,
            "margin_y": 5,
        }
        params_full = SignatureParams(opacity=100.0, **common)
        params_half = SignatureParams(opacity=50.0, **common)

        out_full = insert_signature(img, params_full)
        out_half = insert_signature(img, params_half)

        # Logo lands at (5:25, 5:25) for a 20x20 logo with margin (5, 5).
        idx = (slice(None), slice(5, 25), slice(5, 25))
        assert out_half[idx].mean() < out_full[idx].mean()
        assert out_half[idx].mean() > img[idx].mean()


class TestUnsupportedShape:
    def test_unsupported_ndim_raises(self):
        bad = np.zeros((2, 3, 4, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            insert_signature(bad, SignatureParams(mode="text", text="x"))
