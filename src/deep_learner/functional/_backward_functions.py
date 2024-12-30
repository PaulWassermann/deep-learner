import numpy as np
import scipy.special

from deep_learner.functional.utils import safe_div, safe_log
from deep_learner.tensor import Tensor


# ---------- Unary functions ----------
def broadcast_backward(a: Tensor, shape: int | tuple[int]) -> dict[Tensor, Tensor]:
    # return {a: Tensor(data=...)}
    ...


def dropout_backward(
    a: Tensor, grad: Tensor, mask: Tensor, drop_proba: float
) -> dict[Tensor, Tensor]:
    return {a: Tensor(data=grad.data * mask / (1 - drop_proba))}


def relu_backward(a: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {a: Tensor(data=a.data.astype(bool) * grad.data)}


def sigmoid_backward(a: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {
        a: Tensor(
            data=scipy.special.expit(a.data)
            * (1 - scipy.special.expit(a.data))
            * grad.data
        )
    }


# TODO: possible optimization, cache softmax, faster but heavier memory constraint
def softmax_backward(a: Tensor, grad: Tensor, cache: Tensor) -> dict[Tensor, Tensor]:
    outer_product = np.matmul(cache[..., None], np.expand_dims(cache, axis=-2))
    diagonals = np.zeros(cache.data.shape + cache.data.shape[-1:])
    temp = np.diagonal(diagonals, axis1=-2, axis2=-1)
    temp.setflags(write=True)
    temp[:] = cache

    new_grad = np.squeeze(
        np.matmul(np.expand_dims(grad.data, axis=-2), diagonals - outer_product)
    )
    return {a: Tensor(new_grad)}


def softsign_backward(a: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {a: Tensor(data=(1 / (1 + np.abs(a.data) ** 2)) * grad.data)}


def sum_backward(a: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {a: Tensor(data=np.ones(a.data.shape) * grad.data)}


# ---------- Binary functions ----------
# TODO : fix issue when multiplying grad with np.ones wiht a batch dimension
def add_backward(a: Tensor, b: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {
        a: Tensor(data=np.ones(a.data.shape) * grad.data),
        b: Tensor(data=np.ones(b.data.shape) * grad.data),
    }


def binary_cross_entropy_backward(
    x: Tensor, y: Tensor, grad: Tensor
) -> dict[Tensor, Tensor]:
    return {
        x: Tensor(
            data=-(safe_div(y.data, x.data) - safe_div(1 - y.data, 1 - x.data))
            / len(x.data)
            * grad.data
        ),
        y: Tensor(
            data=-(safe_log(x.data) - safe_log(1 - x.data)) / len(x.data) * grad.data
        ),
    }


def cross_entropy_backward(x: Tensor, y: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {
        x: Tensor(data=-safe_div(y.data, x.data) / len(x.data) * grad.data),
        y: Tensor(data=-safe_log(x.data) / len(x.data) * grad.data),
    }


def matmul_backward(a: Tensor, b: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {
        a: Tensor(data=np.matmul(grad.data, np.swapaxes(b.data, -2, -1))),
        b: Tensor(data=np.matmul(np.swapaxes(a.data, -2, -1), grad.data)),
    }


def multiply_backward(a: Tensor, b: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {
        a: Tensor(data=b.data * grad.data),
        b: Tensor(data=a.data * grad.data),
    }


def subtract_backward(a: Tensor, b: Tensor, grad: Tensor) -> dict[Tensor, Tensor]:
    return {
        a: Tensor(data=np.ones(a.data.shape) * grad.data),
        b: Tensor(data=-np.ones(b.data.shape) * grad.data),
    }
