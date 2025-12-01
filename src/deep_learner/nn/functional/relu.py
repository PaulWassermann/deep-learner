import deep_learner._tensor as t
import deep_learner.functional.functions as f
import deep_learner.nn.module as m


class ReLU(m.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return f.relu(x)

    def __repr__(self):
        return "ReLU()"
