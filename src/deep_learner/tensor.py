from typing import Callable, Optional, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

import deep_learner
from deep_learner._core.utils import format_grad_func_name, indent_text


# TODO: add direct access to data numpy attributes or override them
# TODO: add initialization functions (zeros, ones, constant, random, ...)
# TODO: add operators overloading and replace verbose tensor initializations
# TODO: create an alias for gradient function type hint ?
# TODO: change type parameter constraint to a literal !
# class Tensor[T: SupportedDataTypes]:
class Tensor:

    """
    Tensor class, generic over the data type the instance holds.

    This class mainly exists as a wrapper around the `data` attribute, to allow for
    operations to be registered on each instance. Registered operations allow to compute
    gradients through reverse-mode automatic differentiation.
    """

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        grad_func: Optional[Callable[[Self], dict[Self, Self]]] = None,
    ):
        self.data: NDArray = np.asarray(data)
        self.requires_grad: bool = requires_grad

        # TODO: Keep as tensor ?
        self.grad: Optional[Tensor]

        if self.requires_grad:
            self.grad = Tensor(data=np.zeros(self.data.shape))
        else:
            self.grad = None

        self.grad_func: Optional[Callable[[Self], dict[Self, Self]]] = grad_func

    def __add__(self, other: Self) -> Self:
        return deep_learner.add(self, other)

    def backward(self, gradient: Optional[Self] = None) -> None:
        if gradient is None:
            gradient = Tensor(1)

        if self.grad_func is not None:
            for tensor, gradient_update in self.grad_func(gradient).items():
                if tensor.requires_grad:
                    tensor.grad = accumulate_gradient(tensor.grad, gradient_update)

                tensor.backward(gradient_update)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __matmul__(self, other: Self) -> Self:
        return deep_learner.matmul(self, other)

    def __mul__(self, other: Self) -> Self:
        return deep_learner.multiply(self, other)

    # TODO: for __r*__ methods, see which dtype to use for the Tensor in which
    #  the ArrayLike is cast
    def __radd__(self, other: ArrayLike) -> Self:
        return deep_learner.add(Tensor(other), self)

    def __repr__(self):
        repr_str: str = "Tensor("

        numpy_str = str(self.data).splitlines()
        numpy_str = "\n".join(
            [indent_text(line, indent=len(repr_str)) for line in numpy_str]
        )

        repr_str += f"\n{numpy_str},\n"

        if self.requires_grad:
            repr_str += "requires_grad=True"

        elif self.grad_func is not None:
            repr_str += f"grad_func=<{format_grad_func_name(self.grad_func)}>"

        return repr_str + ")"

    def __rmatmul__(self, other: ArrayLike) -> Self:
        return deep_learner.matmul(Tensor(other), self)

    def __rmul__(self, other: ArrayLike) -> Self:
        return deep_learner.multiply(Tensor(other), self)

    def __rsub__(self, other: ArrayLike) -> Self:
        return deep_learner.subtract(Tensor(other), self)

    # TODO: update __str__ dunder method
    def __str__(self):
        return repr(self)

    def __sub__(self, other: Self) -> Self:
        return deep_learner.subtract(self, other)

    def zero_grad(self) -> None:
        self.grad.data = np.zeros_like(self.grad.data)


def accumulate_gradient(gradient: Tensor, gradient_update: Tensor) -> Tensor:
    accumulated_gradient = gradient.data + gradient_update.data
    if accumulated_gradient.ndim > gradient.data.ndim:
        accumulated_gradient = accumulated_gradient.sum(
            axis=tuple(
                dim for dim in range(gradient_update.data.ndim - gradient.data.ndim)
            ),
            keepdims=False,
        )

    if accumulated_gradient.shape[0] != gradient.data.shape[0]:
        accumulated_gradient = accumulated_gradient.sum(axis=0, keepdims=True)

    return Tensor(accumulated_gradient)


def tensor(data: ArrayLike, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad, grad_func=None)
