import numpy as np
from numpy.typing import ArrayLike, NDArray


def log_clamp(x: ArrayLike, clamp_value: float = -100) -> NDArray:
    return np.maximum(safe_log(x), clamp_value)


def safe_div(x: ArrayLike, y: ArrayLike, eps: float = 1e-10) -> NDArray:
    return x / (y + eps)


def safe_log(x: ArrayLike, eps: float = 1e-100) -> NDArray:
    return np.log(x + eps)
