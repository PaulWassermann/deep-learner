from math import sqrt

import deep_learner._tensor as t
import deep_learner.nn.module as m
import deep_learner.utils as utils


class Linear(m.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()

        self.n_in: int = n_in
        self.n_out: int = n_out

        self.weights: t.Tensor = utils.rand_uniform(
            (n_in, n_out),
            low=-sqrt(1 / n_in),
            high=sqrt(1 / n_in),
            requires_grad=True,
        )
        self.bias: t.Tensor = utils.zeros((1, n_out), requires_grad=True)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x @ self.weights + self.bias

    def __repr__(self):
        return f"Linear(n_in={self.n_in}, n_out={self.n_out})"
