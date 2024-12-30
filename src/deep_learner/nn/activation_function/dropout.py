from deep_learner import Tensor
from deep_learner.functional.functions import dropout
from deep_learner.nn.module import Module


class Dropout(Module):
    def __init__(self, drop_proba: float):
        super().__init__()

        self.drop_proba: float = drop_proba

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return dropout(x, drop_proba=self.drop_proba)
        else:
            return x

    def __repr__(self):
        return f"Dropout(drop_proba={self.drop_proba})"
