from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Optional, Self

from deep_learner._core.utils import indent_text
from deep_learner.tensor import Tensor


# TODO: Add docstrings
# TODO: Add type hints
class Module(ABC):
    def __init__(self):
        self._parameters: dict[str, Tensor] = {}
        self._modules: dict[str, Self] = {}

        self.training: bool = True

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def children(self) -> Generator[Self]:
        for _, child in self.named_children():
            yield child

    def eval(self) -> None:
        self.training = False

        for child in self.children():
            child.eval()

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        ...

    def modules(self) -> Generator[Self]:
        for _, module in self.named_modules():
            yield module

    def named_children(
        self, prefix: Optional[str] = None, visited: Optional[set[Self]] = None
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

    def named_modules(
        self, prefix: Optional[str] = None
    ) -> Generator[tuple[str, Self]]:
        if prefix is None:
            prefix = self.__class__.__name__

        yield prefix, self
        for module_name, module in self.named_children(prefix):
            yield module_name, module

    def named_parameters(self) -> Generator[tuple[str, Tensor]]:
        for module_name, module in self.named_modules():
            for parameter_name, parameter in module._parameters.items():
                yield f"{module_name}.{parameter_name}", parameter

    def parameters(self) -> Generator[Tensor]:
        for _, parameter in self.named_parameters():
            yield parameter

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            + indent_text("\n".join(repr(module) for module in self._modules.values()))
            + "\n)"
        )

    def __setattr__(self, key, value):
        if isinstance(value, Module):
            self._modules[key] = value

        elif isinstance(value, Tensor) and value.requires_grad:
            self._parameters[key] = value

        super().__setattr__(key, value)

    def train(self) -> None:
        self.training = True

        for child in self.children():
            child.train()
