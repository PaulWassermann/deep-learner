from math import sqrt

from deep_learner import Tensor, rand_uniform, zeros
from deep_learner.nn import Module


class Linear(Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()

        self.n_in: int = n_in
        self.n_out: int = n_out

        self.weights: Tensor = rand_uniform(
            (n_in, n_out),
            low=-sqrt(1 / n_in),
            high=sqrt(1 / n_in),
            requires_grad=True,
        )
        self.bias: Tensor = zeros((1, n_out), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weights + self.bias

    def __repr__(self):
        return f"Linear(n_in={self.n_in}, n_out={self.n_out})"
