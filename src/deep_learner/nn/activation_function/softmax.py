from deep_learner import Tensor
from deep_learner.functional.functions import softmax
from deep_learner.nn.module import Module


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        return softmax(x)

    def __repr__(self):
        return "Softmax()"
