import deep_learner._tensor as t
import deep_learner.functional.functions as F
import deep_learner.nn.module as m


class CrossEntropyLoss(m.Module):
    def forward(self, x: t.Tensor, y: t.Tensor) -> t.Tensor:
        return F.cross_entropy(x, y)

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"
