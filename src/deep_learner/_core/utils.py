from collections.abc import Iterable
from typing import Any

import cupy as cp
import numpy as np

import deep_learner._core.types as types


class HookHandle:
    _count: int = 0

    def __new__(cls, *args, **kwargs):
        HookHandle._count += 1
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, hooks_dict: dict[str, Any]):
        self._hook_id = HookHandle._count
        self._hooks_dict = hooks_dict

    def remove(self):
        if self._hook_id in self._hooks_dict:
            del self._hooks_dict[self._hook_id]


class HookHandleGroup:
    def __init__(self, hook_handles: Iterable[HookHandle]):
        self._hooks = list(hook_handles)

    def remove(self):
        for hook in self._hooks:
            hook.remove()


def convert_array(x: types.DeviceArray, device: types.DeviceArray) -> types.DeviceArray:
    if isinstance(x, np.ndarray) and device == types.Device.CUDA:
        return cp.asarray(x)

    elif isinstance(x, cp.ndarray) and device == types.Device.CPU:
        return cp.asnumpy(x)

    else:
        return x


def get_backend(device: types.Device) -> types.Backend:
    match device:
        case types.Device.CPU:
            return types.Backend.CPU.value
        case types.Device.CUDA:
            return types.Backend.CUDA.value
        case _:
            raise ValueError(f"Unknown device: {device}")


def get_special_backend(device: types.Device) -> types.SpecialBackend:
    match device:
        case types.Device.CPU:
            return types.SpecialBackend.CPU.value
        case types.Device.CUDA:
            return types.SpecialBackend.CUDA.value
        case _:
            raise ValueError(f"Unknown device: {device}")


def indent_text(text: str, indent: int = 4) -> str:
    """
    Add the specified amount of white spaces before the provided text string.

    Parameters
    ----------
    text: str
        The string to be indented.

    indent: int
        Number of white spaces to prepend to the string.

    Returns
    -------
    str
        The indented text.
    """

    return "\n".join(indent * " " + line for line in text.splitlines())
