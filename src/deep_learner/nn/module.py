from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Self

import deep_learner._core.types as types
import deep_learner._core.utils as utils
import deep_learner._tensor as t


# TODO: Add docstrings
# TODO: Add type hints
class Module(ABC):
    def __init__(self):
        self._forward_hooks: dict[int, types.ForwardHook] = {}
        self._modules: dict[str, Self] = {}
        self._parameters: dict[str, t.Tensor] = {}

        self.training: bool = True

    def __call__(self, *args, **kwargs) -> Any:
        outputs = self.forward(*args, **kwargs)

        for hook in self._forward_hooks.values():
            hook(self, outputs, *args, **kwargs)

        return outputs

    def children(self) -> Generator[Self]:
        for _, child in self.named_children():
            yield child

    def eval(self) -> None:
        self.training = False

        for child in self.children():
            child.eval()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any: ...

    def modules(self) -> Generator[Self]:
        for _, module in self.named_modules():
            yield module

    def named_children(
        self, prefix: str | None = None, visited: set[Self] | None = None
    ) -> Generator[tuple[str, Self]]:
        if visited is None:
            visited = set()

        if prefix is None:
            prefix = self.__class__.__name__

        for child_name, child in self._modules.items():
            if child not in visited:
                visited.add(child)
                child_prefix = f"{prefix}.{child_name}"
                yield child_prefix, child
                yield from child.named_children(child_prefix, visited)

    def named_modules(self, prefix: str | None = None) -> Generator[tuple[str, Self]]:
        if prefix is None:
            prefix = self.__class__.__name__

        yield prefix, self
        yield from self.named_children(prefix)

    def named_parameters(self) -> Generator[tuple[str, t.Tensor]]:
        for module_name, module in self.named_modules():
            for parameter_name, parameter in module._parameters.items():
                yield f"{module_name}.{parameter_name}", parameter

    def parameters(self) -> Generator[t.Tensor]:
        for _, parameter in self.named_parameters():
            yield parameter

    def register_backward_hook(self, hook: types.BackwardHook) -> utils.HookHandleGroup:
        handles = []
        for parameter in self.parameters():
            handles.append(parameter.register_backward_hook(hook))
        return utils.HookHandleGroup(handles)

    def register_forward_hook(self, hook: types.ForwardHook) -> utils.HookHandle:
        utils.HookHandle._count += 1
        self._forward_hooks[utils.HookHandle._count] = hook
        return utils.HookHandle(self._forward_hooks)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + utils.indent_text(
                "\n".join(repr(module) for module in self._modules.values())
            )
            + "\n)"
        )

    def __setattr__(self, key, value):
        if isinstance(value, Module):
            self._modules[key] = value

        elif isinstance(value, t.Tensor) and value.requires_grad:
            self._parameters[key] = value

        super().__setattr__(key, value)

    def to(self, device: types.Device) -> Self:
        for parameter in self.parameters():
            parameter.to(device)

        return self

    def train(self) -> None:
        self.training = True

        for child in self.children():
            child.train()
