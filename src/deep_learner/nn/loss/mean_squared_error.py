import deep_learner._tensor as t
import deep_learner.functional.functions as F
import deep_learner.nn.module as m


class MeanSquaredErrorLoss(m.Module):
    def forward(self, x: t.Tensor, y: t.Tensor) -> t.Tensor:
        return F.mean_squared_error(x, y)

    def __repr__(self) -> str:
        return "MeanSquaredErrorLoss()"
