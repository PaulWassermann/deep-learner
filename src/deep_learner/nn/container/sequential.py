from typing import Any, Iterable

from deep_learner._core.utils import indent_text
from deep_learner.nn import Module


class Sequential(Module):
    # TODO : remove the _modules_tuple attribute. Instead, register each module using
    #  setattr and use the index for the attribute name
    def __init__(self, *modules: Module):
        super().__init__()

        self._modules_tuple: tuple[Module, ...] = tuple(modules)

    def forward(self, input_: Any) -> Any:
        for module in self._modules_tuple:
            input_ = module(input_)
        return input_

    def __repr__(self):
        return (
            f"Sequential(\n"
            + indent_text(
                "\n".join(
                    f"({index}): {repr(module)}"
                    for index, module in enumerate(self._modules_tuple)
                )
            )
            + "\n)"
        )
