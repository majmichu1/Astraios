"""Tests for pixel math engine."""

import numpy as np
import pytest

from astraios.core.pixel_math import (
    PixelMathError,
    evaluate,
    prepare_variables,
    validate_expression,
)


class TestValidateExpression:
    def test_valid_arithmetic(self):
        assert validate_expression("T + 0.1") is None
        assert validate_expression("T * 2") is None
        assert validate_expression("R + G + B") is None
        assert validate_expression("max(R, G)") is None

    def test_invalid_syntax(self):
        result = validate_expression("T +")
        assert result is not None

    def test_disallowed_operations(self):
        result = validate_expression("__import__('os')")
        assert result is not None

    def test_unknown_function(self):
        result = validate_expression("eval(T)")
        assert result is not None


class TestEvaluate:
    def test_add_scalar(self):
        img = np.ones((50, 50), dtype=np.float32) * 0.3
        variables = prepare_variables(img)
        result = evaluate("T + 0.2", variables)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.01)

    def test_multiply(self):
        img = np.ones((50, 50), dtype=np.float32) * 0.4
        variables = prepare_variables(img)
        result = evaluate("T * 2", variables)
        np.testing.assert_allclose(result.mean(), 0.8, atol=0.01)

    def test_clipped_to_range(self):
        img = np.ones((50, 50), dtype=np.float32) * 0.8
        variables = prepare_variables(img)
        result = evaluate("T + 0.5", variables)
        assert result.max() <= 1.0

    def test_channel_access(self):
        img = np.zeros((3, 50, 50), dtype=np.float32)
        img[0] = 0.5  # R
        img[1] = 0.3  # G
        img[2] = 0.1  # B
        variables = prepare_variables(img)
        result = evaluate("R", variables)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.01)

    def test_luminance(self):
        img = np.ones((3, 50, 50), dtype=np.float32) * 0.5
        variables = prepare_variables(img)
        result = evaluate("L", variables)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.02)

    def test_functions(self):
        img = np.ones((50, 50), dtype=np.float32) * 0.25
        variables = prepare_variables(img)
        result = evaluate("sqrt(T)", variables)
        np.testing.assert_allclose(result.mean(), 0.5, atol=0.01)

    def test_max_function(self):
        img = np.zeros((3, 50, 50), dtype=np.float32)
        img[0] = 0.8
        img[1] = 0.3
        variables = prepare_variables(img)
        result = evaluate("max(R, G)", variables)
        np.testing.assert_allclose(result.mean(), 0.8, atol=0.01)

    def test_comparison(self):
        img = np.linspace(0, 1, 2500).reshape(50, 50).astype(np.float32)
        variables = prepare_variables(img)
        result = evaluate("T > 0.5", variables)
        # About half should be 1.0
        assert 0.4 < result.mean() < 0.6

    def test_unknown_variable_raises(self):
        variables = {"T": np.ones((10, 10), dtype=np.float32)}
        with pytest.raises(PixelMathError, match="Unknown variable"):
            evaluate("X + 1", variables)

    def test_complex_expression(self):
        img = np.ones((3, 50, 50), dtype=np.float32) * 0.5
        variables = prepare_variables(img)
        result = evaluate("clip(R * 2 - G, 0, 1)", variables)
        assert result.shape == (50, 50)


class TestPrepareVariables:
    def test_mono_variables(self):
        img = np.ones((50, 50), dtype=np.float32)
        v = prepare_variables(img)
        assert "T" in v
        assert "L" in v

    def test_color_variables(self):
        img = np.ones((3, 50, 50), dtype=np.float32)
        v = prepare_variables(img)
        assert all(k in v for k in ("T", "R", "G", "B", "L"))

    def test_additional_images(self):
        img = np.ones((50, 50), dtype=np.float32)
        extra = {"ref": np.zeros((50, 50), dtype=np.float32)}
        v = prepare_variables(img, extra)
        assert "ref" in v
