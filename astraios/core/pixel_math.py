"""Pixel Math Engine — safe expression evaluator for image arithmetic.

Uses Python's ast module to parse and evaluate mathematical expressions
on image data. Only whitelisted operations are allowed — no arbitrary
code execution.
"""

from __future__ import annotations

import ast
import logging
import math
import operator

import numpy as np

log = logging.getLogger(__name__)

# Whitelisted binary operators (numpy version)
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

# Whitelisted unary operators
_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted comparison operators
_CMPOPS = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

# Whitelisted functions that operate on arrays (numpy version)
_FUNCTIONS = {
    "min": np.minimum,
    "max": np.maximum,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "clip": np.clip,
    "normalize": lambda x: (x - np.min(x)) / max(np.max(x) - np.min(x), 1e-10),
    "mean": np.mean,
    "median": np.median,
    "sin": np.sin,
    "cos": np.cos,
    "pow": np.power,
}

# GPU versions (torch)
_TORCH_FUNCTIONS = None

def _get_torch_functions():
    global _TORCH_FUNCTIONS
    if _TORCH_FUNCTIONS is not None:
        return _TORCH_FUNCTIONS
    import torch
    _TORCH_FUNCTIONS = {
        "min": torch.minimum,
        "max": torch.maximum,
        "abs": torch.abs,
        "sqrt": torch.sqrt,
        "log": torch.log,
        "log10": torch.log10,
        "exp": torch.exp,
        "clip": torch.clamp,
        "normalize": lambda x: (x - torch.min(x)) / max(torch.max(x) - torch.min(x), 1e-10),
        "mean": torch.mean,
        # torch.median(x) with no dim returns the scalar median directly; the
        # `.values` accessor only exists on the (values, indices) tuple from the
        # dim form, so `.values` here returned a bound method, not the median.
        "median": lambda x: torch.median(x) if x.numel() > 0 else torch.tensor(0.0),
        "sin": torch.sin,
        "cos": torch.cos,
        "pow": torch.pow,
    }
    return _TORCH_FUNCTIONS

# Constants
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


class PixelMathError(Exception):
    """Error in pixel math expression."""
    pass


def validate_expression(expression: str) -> str | None:
    """Check if an expression is syntactically valid and safe.

    Returns None if valid, or an error message string if invalid.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        _check_node(tree.body)
        return None
    except PixelMathError as e:
        return str(e)
    except SyntaxError as e:
        return f"Syntax error: {e.msg}"


def evaluate(
    expression: str,
    variables: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate a pixel math expression (GPU-accelerated when available).

    Parameters
    ----------
    expression : str
        Mathematical expression. Variables:
        - $T : current image (all channels)
        - $1, $2, ... : additional loaded images
        - R, G, B : individual channels of $T
        - L : luminance of $T
    variables : dict
        Mapping of variable names to numpy arrays.

    Returns
    -------
    ndarray
        Result of evaluation.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        _check_node(tree.body)
    except SyntaxError as e:
        raise PixelMathError(f"Syntax error: {e.msg}") from e

    from astraios.core.device_manager import get_device_manager
    dm = get_device_manager()

    use_gpu = dm.is_gpu
    env = {}
    env.update(_CONSTANTS)

    if use_gpu:
        import torch as _torch
        for k, v in variables.items():
            env[k] = dm.from_numpy(v.astype(np.float32, copy=False)) if isinstance(v, np.ndarray) else v
        global _FUNCTIONS
        saved_funcs = _FUNCTIONS
        try:
            _FUNCTIONS = _get_torch_functions()
            result = _eval_node(tree.body, env)
        finally:
            _FUNCTIONS = saved_funcs
        if isinstance(result, (int, float)):
            for v in variables.values():
                if isinstance(v, np.ndarray):
                    result = np.full_like(v, float(result))
                    break
        elif isinstance(result, _torch.Tensor):
            result = result.cpu().numpy()
    else:
        env.update(variables)
        result = _eval_node(tree.body, env)
        if isinstance(result, (int, float)):
            for v in variables.values():
                if isinstance(v, np.ndarray):
                    result = np.full_like(v, result)
                    break

    return np.clip(result, 0, 1).astype(np.float32)


def prepare_variables(
    current_image: np.ndarray,
    additional_images: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Prepare variable dict from image data.

    Parameters
    ----------
    current_image : ndarray
        Current image, shape (H, W) or (C, H, W).
    additional_images : dict, optional
        Named additional images.

    Returns
    -------
    dict
        Variable mapping for evaluate().
    """
    variables: dict[str, np.ndarray] = {"T": current_image}

    # Channel extraction
    if current_image.ndim == 3:
        if current_image.shape[0] >= 1:
            variables["R"] = current_image[0]
        if current_image.shape[0] >= 2:
            variables["G"] = current_image[1]
        if current_image.shape[0] >= 3:
            variables["B"] = current_image[2]
            variables["L"] = (
                0.2126 * current_image[0]
                + 0.7152 * current_image[1]
                + 0.0722 * current_image[2]
            )
    else:
        variables["L"] = current_image

    if additional_images:
        variables.update(additional_images)

    return variables


def _check_node(node: ast.AST) -> None:
    """Recursively validate that an AST node only uses whitelisted operations."""
    if isinstance(node, ast.Expression):
        _check_node(node.body)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINOPS:
            raise PixelMathError(f"Unsupported operator: {type(node.op).__name__}")
        _check_node(node.left)
        _check_node(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARYOPS:
            raise PixelMathError(f"Unsupported unary operator: {type(node.op).__name__}")
        _check_node(node.operand)
    elif isinstance(node, ast.Compare):
        _check_node(node.left)
        for op in node.ops:
            if type(op) not in _CMPOPS:
                raise PixelMathError(f"Unsupported comparison: {type(op).__name__}")
        for comp in node.comparators:
            _check_node(comp)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id not in _FUNCTIONS:
                raise PixelMathError(f"Unknown function: {node.func.id}")
        else:
            raise PixelMathError("Only simple function calls are allowed")
        for arg in node.args:
            _check_node(arg)
    elif isinstance(node, ast.Name):
        pass  # variable reference, checked at eval time
    elif isinstance(node, (ast.Constant,)):
        if not isinstance(node.value, (int, float)):
            raise PixelMathError(f"Only numeric constants allowed, got {type(node.value).__name__}")
    elif isinstance(node, ast.IfExp):
        _check_node(node.test)
        _check_node(node.body)
        _check_node(node.orelse)
    else:
        raise PixelMathError(f"Unsupported expression type: {type(node).__name__}")


def _eval_node(node: ast.AST, env: dict) -> np.ndarray | float:
    """Recursively evaluate an AST node."""
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env)
        right = _eval_node(node.right, env)
        op = _BINOPS[type(node.op)]
        return op(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, env)
        op = _UNARYOPS[type(node.op)]
        return op(operand)
    elif isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        result = None
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, env)
            cmp = _CMPOPS[type(op)](left, right)
            if isinstance(cmp, np.ndarray):
                cmp = cmp.astype(np.float32)
            result = cmp if result is None else (result * cmp)
            left = right
        return result
    elif isinstance(node, ast.Call):
        func = _FUNCTIONS[node.func.id]
        args = [_eval_node(a, env) for a in node.args]
        return func(*args)
    elif isinstance(node, ast.Name):
        name = node.id
        if name in env:
            return env[name]
        raise PixelMathError(f"Unknown variable: {name}")
    elif isinstance(node, ast.Constant):
        return float(node.value)
    elif isinstance(node, ast.IfExp):
        test = _eval_node(node.test, env)
        body = _eval_node(node.body, env)
        orelse = _eval_node(node.orelse, env)
        if isinstance(test, np.ndarray):
            return np.where(test > 0.5, body, orelse)
        return body if test > 0.5 else orelse
    else:
        raise PixelMathError(f"Cannot evaluate: {type(node).__name__}")
