from collections.abc import Callable
from typing import Any

from deep_learner import Tensor


# TODO: implement general case function
def gradient_checking(func: Callable[[Any], Tensor], *args): ...


def gradient_checking_vv(
    func: Callable[[Tensor], Tensor], parameters_vector: Tensor
): ...
