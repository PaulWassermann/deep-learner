from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol

import cupy
import cupyx.scipy.special
import numpy
import scipy.special
from cupy.typing import NDArray as CupyArray
from numpy.typing import NDArray as NumpyArray

if TYPE_CHECKING:
    from deep_learner import Tensor
    from deep_learner.nn import Module


class Backend(Enum):
    CPU = numpy
    CUDA = cupy


class BackwardHook(Protocol):
    def __call__(self, grad: Tensor) -> None: ...


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"


class ForwardHook(Protocol):
    def __call__(
        self, module: Module, outputs: Tensor, *inputs_list, **inputs_dict
    ) -> None: ...


class SpecialBackend(Enum):
    CPU = scipy.special
    CUDA = cupyx.scipy.special


DeviceArray = CupyArray | NumpyArray
