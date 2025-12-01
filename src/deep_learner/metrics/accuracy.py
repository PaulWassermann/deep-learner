import deep_learner._tensor as t


def accuracy(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    return t.Tensor(data=(x.data == y.data).sum() / len(x.data))
