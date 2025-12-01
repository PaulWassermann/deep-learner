from typing import Any

import deep_learner._core.utils as utils
import deep_learner.nn.module as nn


class Sequential(nn.Module):
    def __init__(self, *modules: nn.Module):
        super().__init__()

        for index, module in enumerate(modules):
            setattr(self, str(index), module)

        self._length = index + 1

    def forward(self, input_: Any) -> Any:
        for module_index in range(self._length):
            module = getattr(self, str(module_index))
            input_ = module(input_)
        return input_

    def __len__(self) -> int:
        return self._length

    def __repr__(self):
        return (
            "Sequential(\n"
            + utils.indent_text(
                "\n".join(
                    f"({index}): {repr(getattr(self, str(index)))}"
                    for index in range(self._length)
                )
            )
            + "\n)"
        )
