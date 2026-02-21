from collections.abc import Callable

import numpy as np

from deep_learner import Tensor

EPSILON = 1e-7


def gradient_check_ss(func: Callable[..., Tensor], *inputs: Tensor) -> None:
    eps = Tensor(EPSILON)
    r = func(*inputs)
    r.backward()

    for i in range(len(inputs)):
        args = list(inputs)
        args[i] += eps
        r_pos = func(*args)

        args = list(inputs)
        args[i] -= eps
        r_neg = func(*args)

        t = (r_pos - r_neg) / (2 * eps)

        assert np.allclose(t.data, inputs[i].grad.data)


def gradient_check_vs(func: Callable[..., Tensor], *inputs: Tensor) -> None:
    r = func(*inputs)
    r.backward()

    for i in range(len(inputs)):
        shape = inputs[i].data.shape
        partials = np.zeros(shape)

        for index in np.ndindex(shape):
            eps = np.zeros(shape)
            eps[index] = EPSILON

            args = list(inputs)
            args[i] += eps
            r_pos = func(*args)

            args = list(inputs)
            args[i] -= eps
            r_neg = func(*args)

            partials[index] = ((r_pos - r_neg) / (2 * np.linalg.norm(eps))).data

        assert np.allclose(partials, inputs[i].grad.data)
