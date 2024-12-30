from deep_learner import Tensor


def accuracy(x: Tensor, y: Tensor) -> Tensor:
    return Tensor(data=(x.data == y.data).sum() / len(x.data))
