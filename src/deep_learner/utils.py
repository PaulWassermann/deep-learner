from collections.abc import Sequence

import numpy as np

from deep_learner import Tensor, tensor


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
