from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

import deep_learner
import deep_learner._core.types as types
import deep_learner._core.utils as utils

if TYPE_CHECKING:
    from deep_learner._core.types import Backend
    from deep_learner.functional._backward_functions import BackwardFunction


# TODO: add direct access to data numpy attributes or override them
# TODO: add operators overloading and replace verbose tensor initializations
# TODO: how to ensure data and grad_func are on the same device ?
# TODO: add a copy method
# TODO: add call to backward hooks
class Tensor:
    """
    Tensor class.

    This class mainly exists as a wrapper around the `data` attribute, to allow for
    operations to be registered on each instance. Registered operations allow to compute
    gradients through reverse-mode automatic differentiation.
    """

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        grad_func: BackwardFunction | None = None,
        device: types.Device = types.Device.CPU,
    ):
        backend = utils.get_backend(device)

        self.data: types.DeviceArray
        if isinstance(data, np.ndarray) and device == types.Device.CUDA:
            self.data = backend.array(data)
        else:
            self.data = backend.asarray(data)

        self.requires_grad: bool = requires_grad

        self.device: types.Device = device

        # TODO: Keep as tensor ?
        self.grad: Tensor | None
        if self.requires_grad:
            self.grad = Tensor(data=backend.zeros(self.data.shape), device=device)
        else:
            self.grad = None

        self._grad_func: BackwardFunction | None = grad_func

        self._backward_hooks: dict[int, types.BackwardHook] = {}

    def __add__(self, other: Tensor) -> Tensor:
        return deep_learner.add(self, other)

    def backward(self, gradient: Tensor | None = None) -> None:
        if gradient is None:
            gradient = Tensor(1, device=self.device)

        if self._grad_func is not None:
            for tensor, gradient_update in self._grad_func(gradient).items():
                if tensor.requires_grad:
                    tensor.grad = accumulate_gradient(tensor.grad, gradient_update)

                tensor.backward(gradient_update)

    def detach(self) -> Tensor:
        return Tensor(self.data, device=self.device)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for x in self.data:
            yield Tensor(x)

    def __matmul__(self, other: Tensor) -> Tensor:
        return deep_learner.matmul(self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        return deep_learner.multiply(self, other)

    def __pow__(self, other: float) -> Tensor:
        return deep_learner.power(self, other)

    def __radd__(self, other: ArrayLike) -> Tensor:
        return deep_learner.add(Tensor(other, device=self.device), self)

    def __rdiv__(self, other: ArrayLike) -> Tensor:
        return deep_learner.divide(Tensor(other, device=self.device), self)

    def register_backward_hook(self, hook: types.BackwardHook) -> utils.HookHandle:
        self._backward_hooks[utils.HookHandle._count] = hook
        return utils.HookHandle(self._backward_hooks)

    def __repr__(self) -> str:
        repr_str: str = "Tensor("

        numpy_str = str(self.data).splitlines()
        numpy_str = "\n".join(
            [utils.indent_text(line, indent=len(repr_str)) for line in numpy_str]
        )

        repr_str += f"\n{numpy_str},\n"

        if self.requires_grad:
            repr_str += "requires_grad=True, "

        if self._grad_func is not None:
            repr_str += f"grad_func=<{self._grad_func}>, "

        repr_str += f"device={self.device}"

        return repr_str + ")"

    def __rmatmul__(self, other: ArrayLike) -> Tensor:
        return deep_learner.matmul(Tensor(other, device=self.device), self)

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return deep_learner.multiply(Tensor(other, device=self.device), self)

    def __rsub__(self, other: ArrayLike) -> Tensor:
        return deep_learner.subtract(Tensor(other, device=self.device), self)

    # TODO: update __str__ dunder method
    def __str__(self):
        return repr(self)

    def __sub__(self, other: Tensor) -> Tensor:
        return deep_learner.subtract(self, other)

    def to(self, device: types.Device) -> Tensor:
        self.data = utils.convert_array(self.data, device)
        self.device = device

        if self.grad:
            self.grad = self.grad.to(device)

        if self._grad_func:
            self._grad_func.to(device)

        return self

    def __truediv__(self, other: Tensor) -> Tensor:
        return deep_learner.divide(self, other)

    def zero_grad(self) -> None:
        backend = utils.get_backend(self.device)
        self.grad.data = backend.zeros_like(self.grad.data)


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

    return Tensor(accumulated_gradient, device=gradient.device)


def tensor(
    data: ArrayLike,
    requires_grad: bool = False,
    device: types.Device = types.Device.CPU,
) -> Tensor:
    return Tensor(data, requires_grad, grad_func=None, device=device)
