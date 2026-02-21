from abc import ABC, abstractmethod
from typing import Self

import deep_learner._core.types as types
import deep_learner._core.utils as utils
import deep_learner._tensor as t
import deep_learner.functional.utils as f


class BackwardFunction(ABC):
    def __init__(self):
        self._tensors = []

    @abstractmethod
    def __call__(self, grad: t.Tensor, *args, **kwargs) -> dict[t.Tensor, t.Tensor]: ...

    def __setattr__(self, name, value):
        if isinstance(value, t.Tensor):
            self._tensors.append(value)
        return super().__setattr__(name, value)

    @classmethod
    def __str__(cls) -> str:
        return cls.__name__

    def to(self, device: types.Device) -> Self:
        for tensor in self._tensors:
            tensor.to(device)
        return self


# ---------- Unary functions ----------
class DropoutBackward(BackwardFunction):
    def __init__(self, a: t.Tensor, mask: t.Tensor, drop_proba: float):
        super().__init__()

        self.a = a
        self.mask = mask
        self.drop_proba = drop_proba

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        return {
            self.a: t.Tensor(
                data=grad.data * self.mask / (1 - self.drop_proba), device=grad.device
            )
        }


class ExponentialBackward(BackwardFunction):
    def __init__(self, a: t.Tensor):
        super().__init__()

        self.a = a

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_backend(grad.device)
        return {
            self.a: t.Tensor(
                data=backend.exp(self.a.data) * grad.data, device=grad.device
            )
        }


class MeanBackward(BackwardFunction):
    def __init__(self, a: t.Tensor):
        super().__init__()

        self.a = a

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        return {self.a: t.Tensor(data=grad.data / self.a.data.size, device=grad.device)}


class PowerBackward(BackwardFunction):
    def __init__(self, a: t.Tensor, power: float):
        super().__init__()

        self.a = a
        self.power = power

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        if self.power == 1:
            return {self.a: t.Tensor(grad.data, device=grad.device)}

        return {
            self.a: t.Tensor(
                (self.power * self.a.data ** (self.power - 1)) * grad.data,
                device=grad.device,
            )
        }


class ReluBackward(BackwardFunction):
    def __init__(self, a: t.Tensor):
        super().__init__()

        self.a = a

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        return {
            self.a: t.Tensor(
                data=self.a.data.astype(bool) * grad.data, device=grad.device
            )
        }


class SigmoidBackward(BackwardFunction):
    def __init__(self, a: t.Tensor):
        super().__init__()

        self.a = a

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_special_backend(grad.device)
        return {
            self.a: t.Tensor(
                data=backend.expit(self.a.data)
                * (1 - backend.expit(self.a.data))
                * grad.data,
                device=grad.device,
            )
        }


# TODO: possible optimization, cache softmax, faster but heavier memory constraint
# TODO: refactor computation to avoid if statement
class SoftmaxBackward(BackwardFunction):
    def __init__(self, a: t.Tensor, cache: types.DeviceArray):
        super().__init__()

        self.a = a
        self.cache = cache

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_backend(grad.device)
        outer_product = backend.matmul(
            self.cache[..., None], backend.expand_dims(self.cache, axis=-2)
        )
        diagonals = backend.zeros(self.cache.shape + self.cache.shape[-1:])
        temp = backend.diagonal(diagonals, axis1=-2, axis2=-1)

        if grad.device == types.Device.CPU:
            temp.setflags(write=True)

        temp[:] = self.cache

        new_grad = backend.squeeze(
            backend.matmul(
                backend.expand_dims(grad.data, axis=-2), diagonals - outer_product
            )
        )
        return {self.a: t.Tensor(new_grad, device=grad.device)}

    def to(self, device: types.Device) -> Self:
        self.cache = utils.convert_array(self.cache, device)
        return super().to(device)


class SoftsignBackward(BackwardFunction):
    def __init__(self, a: t.Tensor):
        super().__init__()

        self.a = a

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_backend(grad.device)
        return {
            self.a: t.Tensor(
                data=(1 / (1 + backend.abs(self.a.data) ** 2)) * grad.data,
                device=grad.device,
            )
        }


class SumBackward(BackwardFunction):
    def __init__(self, a: t.Tensor):
        super().__init__()

        self.a = a

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_backend(grad.device)
        return {
            self.a: t.Tensor(
                data=backend.ones(self.a.data.shape) * grad.data, device=grad.device
            )
        }


class TanhBackward(BackwardFunction):
    def __init__(self, a: t.Tensor):
        super().__init__()

        self.a = a

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_backend(grad.device)
        return {
            self.a: t.Tensor(
                data=(1 - backend.tanh(self.a.data) ** 2) * grad.data,
                device=grad.device,
            )
        }


# ---------- Binary functions ----------
# TODO : fix issue when multiplying grad with np.ones wiht a batch dimension
class AddBackward(BackwardFunction):
    def __init__(self, a: t.Tensor, b: t.Tensor):
        super().__init__()

        self.a = a
        self.b = b

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_backend(grad.device)
        return {
            self.a: t.Tensor(
                data=backend.ones(self.a.data.shape) * grad.data, device=grad.device
            ),
            self.b: t.Tensor(
                data=backend.ones(self.b.data.shape) * grad.data, device=grad.device
            ),
        }


class BinaryCrossEntropyBackward(BackwardFunction):
    def __init__(self, x: t.Tensor, y: t.Tensor):
        super().__init__()

        self.x = x
        self.y = y

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        return {
            self.x: t.Tensor(
                data=-(
                    f.safe_div(self.y.data, self.x.data)
                    - f.safe_div(1 - self.y.data, 1 - self.x.data)
                )
                / self.x.data.size
                * grad.data,
                device=grad.device,
            ),
            self.y: t.Tensor(
                data=-(f.safe_log(self.x.data) - f.safe_log(1 - self.x.data))
                / self.x.data.size
                * grad.data,
                device=grad.device,
            ),
        }


class CrossEntropyBackward(BackwardFunction):
    def __init__(self, x: t.Tensor, y: t.Tensor):
        super().__init__()

        self.x = x
        self.y = y

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        return {
            self.x: t.Tensor(
                data=-f.safe_div(self.y.data, self.x.data)
                / len(self.x.data)
                * grad.data,
                device=grad.device,
            ),
            self.y: t.Tensor(
                data=-f.safe_log(self.x.data) / len(self.x.data) * grad.data,
                device=grad.device,
            ),
        }


class DivideBackward(BackwardFunction):
    def __init__(self, a: t.Tensor, b: t.Tensor):
        super().__init__()

        self.a = a
        self.b = b

    def __call__(self, grad: t.Tensor):
        return {
            self.a: t.Tensor(data=1 / self.b.data * grad.data, device=grad.device),
            self.b: t.Tensor(
                data=-self.a.data / (self.b.data**2) * grad.data, device=grad.device
            ),
        }


class MatmulBackward(BackwardFunction):
    def __init__(self, a: t.Tensor, b: t.Tensor):
        super().__init__()

        self.a = a
        self.b = b

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_backend(grad.device)
        return {
            self.a: t.Tensor(
                data=grad.data @ backend.swapaxes(self.b.data, -2, -1),
                device=grad.device,
            ),
            self.b: t.Tensor(
                data=backend.swapaxes(self.a.data, -2, -1) @ grad.data,
                device=grad.device,
            ),
        }


class MeanSquaredErrorBackward(BackwardFunction):
    def __init__(self, x: t.Tensor, y: t.Tensor):
        super().__init__()

        self.x = x
        self.y = y

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        return {
            self.x: t.Tensor(
                (self.x.data - self.y.data) * grad.data / len(self.x.data),
                device=grad.device,
            ),
            self.y: t.Tensor(
                (self.y.data - self.x.data) * grad.data / len(self.x.data),
                device=grad.device,
            ),
        }


class MutliplyBackward(BackwardFunction):
    def __init__(self, a: t.Tensor, b: t.Tensor):
        super().__init__()

        self.a = a
        self.b = b

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        return {
            self.a: t.Tensor(data=self.b.data * grad.data, device=grad.device),
            self.b: t.Tensor(data=self.a.data * grad.data, device=grad.device),
        }


class SubtractBackward(BackwardFunction):
    def __init__(self, a: t.Tensor, b: t.Tensor):
        super().__init__()

        self.a = a
        self.b = b

    def __call__(self, grad: t.Tensor) -> dict[t.Tensor, t.Tensor]:
        backend = utils.get_backend(grad.device)
        return {
            self.a: t.Tensor(
                data=backend.ones(self.a.data.shape) * grad.data, device=grad.device
            ),
            self.b: t.Tensor(
                data=-backend.ones(self.b.data.shape) * grad.data, device=grad.device
            ),
        }
