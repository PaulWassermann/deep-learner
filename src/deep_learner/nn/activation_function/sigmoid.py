from deep_learner import Tensor
from deep_learner.functional.functions import sigmoid
from deep_learner.nn.module import Module


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)

    def __repr__(self):
        return "Sigmoid()"
