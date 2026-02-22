from collections.abc import Callable

import numpy as np

from deep_learner import Tensor, zeros

ATOL = 1e-5
EPSILON = 1e-7
RTOL = 1e-4


def gradient_check_ss(func: Callable[..., Tensor], *args: Tensor) -> None:
    """Compare the computed gradient of a scalar function with respect to its
    scalar parameters with the result of the central finite differences for each
    parameter.

    The gradient is computed through the backward method of the output Tensor.

    Parameters
    ----------
    func : Callable[..., Tensor]
        The function which backward function is to be checked

    Raises
    ------
    AssertionError:
        If the finite difference gives a different result than the computed
        gradient.
    """
    for arg in args:
        arg.data = arg.data.astype("float64")
        arg.grad = zeros(arg.grad.data.shape)

    r = func(*args)
    r.backward()

    for i in range(len(args)):
        initial_value = args[i].data.copy()

        args[i].data += EPSILON
        r_pos = func(*args)

        args[i].data = initial_value - EPSILON
        r_neg = func(*args)

        args[i].data = initial_value

        t = (r_pos - r_neg) / (2 * Tensor(EPSILON))

        assert np.allclose(t.data, args[i].grad.data), (
            f"Different gradients for arg n°{i}.\n"
            f"Analytical: {args[i].grad}\n"
            f"Numerical:  {t}\n"
        )


def gradient_check_vs(func: Callable[..., Tensor], *args: Tensor) -> None:
    """Compare the computed gradient of a scalar function with respect to its
    vector parameters with the result of the central finite differences for each
    parameter.

    The gradient is computed through the backward method of the output Tensor.

    Parameters
    ----------
    func : Callable[..., Tensor]
        The function which backward function is to be checked

    Raises
    ------
    AssertionError:
        If the finite difference gives a different result than the computed
        gradient.
    """
    for arg in args:
        arg.data = arg.data.astype("float64")
        arg.grad = zeros(arg.grad.data.shape)

    r = func(*args)
    r.backward()

    for i in range(len(args)):
        initial_value = args[i].data.copy()
        shape = args[i].data.shape
        partials = np.zeros(shape)

        for index in np.ndindex(shape):
            args[i].data[index] = initial_value[index] + EPSILON
            r_pos = func(*args)

            args[i].data[index] = initial_value[index] - EPSILON
            r_neg = func(*args)

            args[i].data[index] = initial_value[index]

            partials[index] = ((r_pos - r_neg) / (2 * Tensor(EPSILON))).data

        assert np.allclose(partials, args[i].grad.data), (
            f"Different gradients for arg n°{i}.\n"
            f"Analytical: {args[i].grad}\n"
            f"Numerical:  {partials}\n"
        )
