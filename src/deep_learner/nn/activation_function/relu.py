from deep_learner import Tensor
from deep_learner.functional.functions import relu
from deep_learner.nn.module import Module


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

    def __repr__(self):
        return "ReLU()"
