from collections.abc import Generator, Iterable, Sequence
from itertools import islice

import numpy as np

from deep_learner import Tensor, tensor


def batch(x, /, *iterables, batch_size: int, shuffle: bool = True) -> Generator[tuple]:
    indices = np.arange(len(x))
    rng = np.random.default_rng()

    if shuffle:
        rng.shuffle(indices, axis=0)

    yield from zip(
        *(_batched(arg[indices], batch_size) for arg in (x, *iterables)), strict=False
    )


def _batched(iterable: Iterable, n: int):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def constant(
    shape: int | Sequence[int], value: float, requires_grad: bool = False
) -> Tensor:
    """Create a new tensor, initializing all its values to the `constant` argument.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the new tensor
    value : float
        Value to
    requires_grad : bool, optional
        Indicates if gradients must be accumulated for the new tensor, by default False

    Returns
    -------
    Tensor
        A new tensor filled with `value`
    """
    return tensor(np.full(shape, value), requires_grad)


def ones(shape: int | Sequence[int], requires_grad: bool = False) -> Tensor:
    """Create a new tensor, initializing all its values to 1.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the new tensor
    requires_grad : bool, optional
        Indicates if gradients must be accumulated for the new tensor, by default False

    Returns
    -------
    Tensor
        A new tensor filled with ones
    """
    return tensor(np.ones(shape), requires_grad)


def rand_int(
    shape: int | Sequence[int], low: int, high: int, requires_grad: bool = False
) -> Tensor:
    """Create a new tensor and initalize its values sampling from a discrete uniform
    distribution bounded between `low` and `high`.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the new tensor
    low : float
        Inferior bound of the uniform distribution
    high : float
        Superior bound of the uniform distribution
    requires_grad : bool, optional
        Indicates if gradients must be accumulated for the new tensor, by default False

    Returns
    -------
    Tensor
        A new tensor filled with random values sampled from a discrete uniform
        distribution
    """
    return tensor(np.random.randint(low, high, shape), requires_grad)


def rand_uniform(
    shape: int | Sequence[int], low: float, high: float, requires_grad: bool = False
) -> Tensor:
    """Create a new tensor and intialize its values sampling from a continuous uniform
    distribution bounded between `low`and `high`.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the new tensor
    low : float
        Inferior bound of the uniform distribution
    high : float
        Superior bound of the uniform distribution
    requires_grad : bool, optional
        Indicates if gradients must be accumulated for the new tensor, by default False

    Returns
    -------
    Tensor
        A new tensor filled with random values sampled from a continuous uniform
        distribution
    """
    data = (high - low) * np.random.random(shape) + low
    return tensor(data, requires_grad)


def zeros(shape: int | Sequence[int], requires_grad: bool = False) -> Tensor:
    """Create a new tensor, initializing all its values to 0.

    Parameters
    ----------
    shape : int or sequence of int
        Shape of the new tensor
    requires_grad : bool, optional
        Indicates if gradients must be accumulated for the new tensor, by default False

    Returns
    -------
    Tensor
        A new tensor filled with zeros
    """
    return tensor(np.zeros(shape), requires_grad)
