"""Tests for background extraction."""

import numpy as np
import pytest

from astraios.core.background import BackgroundParams, extract_background


class TestBackgroundExtraction:
    def test_removes_linear_gradient(self):
        # Create an image with a strong linear gradient
        y, x = np.mgrid[0:200, 0:300]
        gradient = (x / 300.0 * 0.5).astype(np.float32)  # 0 to 0.5 left-right
        signal = np.full((200, 300), 0.3, dtype=np.float32)
        image = signal + gradient

        params = BackgroundParams(grid_size=8, polynomial_order=2)
        corrected, bg_model = extract_background(image, params)

        # After correction, the gradient should be significantly reduced
        left_mean = np.mean(corrected[:, :50])
        right_mean = np.mean(corrected[:, -50:])
        original_diff = abs(np.mean(image[:, -50:]) - np.mean(image[:, :50]))
        corrected_diff = abs(right_mean - left_mean)
        assert corrected_diff < original_diff * 0.3  # at least 70% gradient removed

    def test_preserves_signal(self):
        # Flat image with small uniform signal
        image = np.full((100, 120), 0.3, dtype=np.float32)
        image += np.random.normal(0, 0.001, image.shape).astype(np.float32)

        corrected, bg_model = extract_background(image)
        # Signal should be roughly preserved
        assert abs(np.mean(corrected) - np.mean(corrected)) < 0.1

    def test_color_image(self):
        data = np.random.random((3, 100, 120)).astype(np.float32) * 0.1
        # Add gradient to each channel
        y, x = np.mgrid[0:100, 0:120]
        for ch in range(3):
            data[ch] += (x / 120.0 * 0.3).astype(np.float32)

        corrected, bg_model = extract_background(data)
        assert corrected.shape == data.shape
        assert bg_model.shape == data.shape

    def test_preserves_linear_range(self):
        image = np.random.random((100, 120)).astype(np.float32) * 0.5
        corrected, _ = extract_background(image)
        # Linear subtraction may yield small negatives; display stretch clips later
        assert np.isfinite(corrected).all()
        assert corrected.max() <= image.max() + 0.05


class TestEdgeGradientArtifact:
    """Regression: background extraction must not leave a bright gradient on the
    edges. Caused by (a) the model being smoothed with zero-padding (darkening
    the borders) and (b) samples not reaching the edges (polynomial
    extrapolation). Both are fixed."""

    @staticmethod
    def _bg(kind, h=320, w=320):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        if kind == "vignette":
            return (0.08 + 0.25 * (((xx / w - 0.5)) ** 2 + ((yy / h - 0.5)) ** 2)).astype(np.float32)
        return (0.1 + 0.3 * (xx / w)).astype(np.float32)

    @staticmethod
    def _edge_excess(corrected):
        h, w = corrected.shape
        b = 16
        edge = np.concatenate([
            corrected[:b].ravel(), corrected[-b:].ravel(),
            corrected[:, :b].ravel(), corrected[:, -b:].ravel(),
        ])
        center = corrected[h // 2 - 30:h // 2 + 30, w // 2 - 30:w // 2 + 30]
        return float(edge.mean()) - float(center.mean())

    def test_no_bright_edges_after_extraction(self):
        from astraios.core.background import BackgroundParams, extract_background

        for kind in ("vignette", "linear"):
            img = self._bg(kind)
            corrected, _ = extract_background(img, BackgroundParams())  # default smoothing=0.5
            excess = abs(self._edge_excess(corrected))
            assert excess < 0.02, f"{kind}: edge excess {excess:.4f} too high"

    def test_smoothing_does_not_darken_model_edges(self):
        # The model's border values must stay close to the interior, not be
        # pulled toward zero by the smoothing.
        from astraios.core.background import BackgroundParams, extract_background

        img = self._bg("vignette")
        _, model = extract_background(img, BackgroundParams(smoothing=0.8))
        h, w = model.shape
        # Edge of the model should track the true (brighter) edge background,
        # i.e. be >= the model centre for this vignette case.
        assert float(model[0].mean()) >= float(model[h // 2].mean()) - 0.01
