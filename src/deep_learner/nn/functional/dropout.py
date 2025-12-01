import deep_learner._tensor as t
import deep_learner.functional.functions as f
import deep_learner.nn.module as m


class Dropout(m.Module):
    def __init__(self, drop_proba: float):
        super().__init__()

        self.drop_proba: float = drop_proba

    def forward(self, x: t.Tensor) -> t.Tensor:
        if self.training:
            return f.dropout(x, drop_proba=self.drop_proba)
        else:
            return x

    def __repr__(self):
        return f"Dropout(drop_proba={self.drop_proba})"
