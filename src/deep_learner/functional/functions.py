from collections import defaultdict

import numpy as np
import scipy.special

from deep_learner._core.utils import partial_wrapper
from deep_learner.functional._backward_functions import (
    add_backward,
    binary_cross_entropy_backward,
    broadcast_backward,
    cross_entropy_backward,
    dropout_backward,
    matmul_backward,
    multiply_backward,
    relu_backward,
    sigmoid_backward,
    softmax_backward,
    softsign_backward,
    subtract_backward,
    sum_backward,
)
from deep_learner.functional.utils import log_clamp
from deep_learner.tensor import Tensor


# TODO : add input validation (shapes, types, ...)
# TODO : much later, add fused operations and their corresponding backward functions
# ---------- Unary functions ----------
def broadcast(a: Tensor, shape: int | tuple[int]) -> Tensor:
    return Tensor(
        data=np.broadcast_to(a.data, shape),
        grad_func=partial_wrapper(broadcast_backward, a, shape=shape),
    )


def dropout(a: Tensor, drop_proba: float) -> Tensor:
    mask = np.random.rand(*a.data.shape) > drop_proba
    return Tensor(
        data=mask * a.data / (1 - drop_proba),
        grad_func=partial_wrapper(
            dropout_backward, a, mask=mask, drop_proba=drop_proba
        ),
    )


def relu(a: Tensor) -> Tensor:
    return Tensor(
        data=np.maximum(a.data, 0, a.data), grad_func=partial_wrapper(relu_backward, a)
    )


def sigmoid(a: Tensor) -> Tensor:
    return Tensor(
        data=scipy.special.expit(a.data), grad_func=partial_wrapper(sigmoid_backward, a)
    )


def softmax(a: Tensor) -> Tensor:
    result = scipy.special.softmax(a.data, axis=-1)
    return Tensor(
        data=result,
        grad_func=partial_wrapper(softmax_backward, a, cache=result),
    )


def softsign(a: Tensor) -> Tensor:
    return Tensor(
        data=(a.data / (1 + np.abs(a.data))),
        grad_func=partial_wrapper(softsign_backward, a),
    )


# TODO : add axis and keepdim parameters
def sum(a: Tensor) -> Tensor:
    return Tensor(data=a.data.sum(), grad_func=partial_wrapper(sum_backward, a))


# ---------- Binary functions ----------
def add(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(data=a.data + b.data, grad_func=partial_wrapper(add_backward, a, b))


def binary_cross_entropy(x: Tensor, y: Tensor) -> Tensor:
    return Tensor(
        data=-np.mean(
            y.data * log_clamp(x.data) + (1 - y.data) * log_clamp(1 - x.data)
        ),
        grad_func=partial_wrapper(binary_cross_entropy_backward, x, y),
    )


def cross_entropy(x: Tensor, y: Tensor) -> Tensor:
    return Tensor(
        data=-np.sum(y.data * log_clamp(x.data)) / x.data.shape[0],
        grad_func=partial_wrapper(cross_entropy_backward, x, y),
    )


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(
        data=np.matmul(a.data, b.data), grad_func=partial_wrapper(matmul_backward, a, b)
    )


def multiply(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(
        data=a.data * b.data, grad_func=partial_wrapper(multiply_backward, a, b)
    )


def subtract(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(
        data=a.data - b.data, grad_func=partial_wrapper(subtract_backward, a, b)
    )
