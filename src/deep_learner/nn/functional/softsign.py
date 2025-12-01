import deep_learner._tensor as t
import deep_learner.functional.functions as f
import deep_learner.nn.module as m


class Softsign(m.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return f.softsign(x)

    def __repr__(self) -> str:
        return "Softsign()"
