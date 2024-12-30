from math import sqrt

from deep_learner import Tensor, rand_uniform
from deep_learner.nn import Module


class Embedding(Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()

        self.n_in: int = n_in
        self.n_out: int = n_out

        self.initialize_embeddings()

    def forward(self, x: Tensor) -> Tensor:
        ...

    # TODO: provide initialization variants
    def initialize_embeddings(self) -> None:
        self.weights = rand_uniform(
            (self.n_in, self.n_out),
            low=-sqrt(1 / self.n_in),
            high=sqrt(1 / self.n_in),
            requires_grad=True,
        )

    def __repr__(self):
        return f"Embedding(vocab_size={self.vocabulary_size}, n_features={self.embedding_size})"
