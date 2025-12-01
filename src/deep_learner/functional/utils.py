import cupy as cp
from numpy.typing import ArrayLike


def log_clamp(x: ArrayLike, clamp_value: float = -100) -> ArrayLike:
    backend = cp.get_array_module(x)
    return backend.maximum(safe_log(x), clamp_value)


def safe_div(x: ArrayLike, y: ArrayLike, eps: float = 1e-10) -> ArrayLike:
    return x / (y + eps)


def safe_log(x: ArrayLike, eps: float = 1e-100) -> ArrayLike:
    backend = cp.get_array_module(x)
    return backend.log(x + eps)
