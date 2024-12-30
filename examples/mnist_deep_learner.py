import cProfile
import itertools
import time

import mnist
import numpy as np

from deep_learner import Tensor
from deep_learner.datasets import mnist
from deep_learner.functional.functions import cross_entropy
from deep_learner.metrics.accuracy import accuracy
from deep_learner.nn.activation_function.dropout import Dropout
from deep_learner.nn.activation_function.relu import ReLU
from deep_learner.nn.activation_function.softmax import Softmax
from deep_learner.nn.layer.linear import Linear
from deep_learner.nn.module import Module
from deep_learner.nn.optimizer.sgd import SGD


def batch(x, /, *args, batch_size: int, shuffle: bool = True) -> ...:
    indices = np.arange(len(x))
    rng = np.random.default_rng()

    if shuffle:
        rng.shuffle(indices, axis=0)

    yield from zip(*(batched(arg[indices], batch_size) for arg in (x, *args)))


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def one_hot_encoding(x):
    # Assume x has 1 dimension for now
    return np.eye(x.max() + 1)[x]


def preprocess(x) -> ...:
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-3)


def train(module):
    train_X, train_Y, test_X, test_Y = mnist()
    train_X = preprocess(train_X.reshape(-1, 784))
    train_Y = one_hot_encoding(train_Y).astype(float)

    test_X = preprocess(test_X.reshape(-1, 784))
    test_Y = one_hot_encoding(test_Y).astype(float)

    optimizer = SGD(module, learning_rate=1e-2)
    batch_size: int = 64
    num_batches: int = train_X.shape[0] // batch_size

    for epoch in range(10):
        module.train()

        total_loss = Tensor(0)
        train_accuracy = Tensor(0)

        for batch_X, batch_Y in batch(train_X, train_Y, batch_size=batch_size):
            optimizer.zero_grad()

            batch_X = Tensor(batch_X)
            batch_Y = Tensor(batch_Y)

            y_hat = module(batch_X)

            loss = cross_entropy(y_hat, batch_Y)

            loss.backward()

            optimizer.step()

            total_loss = total_loss + loss

            train_accuracy = train_accuracy + accuracy(
                Tensor(np.argmax(y_hat.data, axis=-1)),
                Tensor(np.argmax(batch_Y.data, axis=-1)),
            )

        module.eval()
        test_y_hat = module(Tensor(test_X))

        test_accuracy = accuracy(
            Tensor(np.argmax(test_y_hat.data, axis=1)),
            Tensor(np.argmax(test_Y, axis=1)),
        ).data
        print(
            f"{epoch=}, "
            f"loss={total_loss.data / num_batches:.4f}, "
            f"train accuracy={train_accuracy.data / num_batches:.2%}, "
            f"test accuracy={test_accuracy:.2%}"
        )


if __name__ == "__main__":
    from functools import partial

    from deep_learner.functional._backward_functions import add_backward

    add_derivative = partial(add_backward, grad=Tensor([1, 1, 1]))
    input_vector = Tensor(np.random.random())

    class MyModule(Module):
        def __init__(self):
            super().__init__()

            self.linear_1 = Linear(n_in=784, n_out=128)
            self.linear_2 = Linear(n_in=128, n_out=128)
            self.linear_3 = Linear(n_in=128, n_out=10)

            self.dropout = Dropout(drop_proba=0.1)
            self.relu = ReLU()
            self.softmax = Softmax()

        def forward(self, x: Tensor) -> Tensor:
            return self.softmax(
                self.linear_3(
                    self.dropout(self.relu(self.linear_2(self.relu(self.linear_1(x)))))
                )
            )

    my_module = MyModule()

    from torch import nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()

            self.linear_1 = nn.Linear(in_features=784, out_features=128)
            self.linear_2 = nn.Linear(in_features=128, out_features=128)
            self.linear_3 = nn.Linear(in_features=128, out_features=10)

            self.dropout = nn.Dropout(0.1)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.linear_3(
                self.relu(self.dropout(self.linear_2(self.relu(self.linear_1(x)))))
            )

    torch_module = Model()

    # my_module.linear_1.weights = Tensor(torch_module.linear_1.weight.data.T, requires_grad=True)
    # my_module.linear_1.bias = Tensor(torch_module.linear_1.bias.data.T, requires_grad=True)
    # my_module.linear_2.weights = Tensor(torch_module.linear_2.weight.data.T, requires_grad=True)
    # my_module.linear_2.bias = Tensor(torch_module.linear_2.bias.data.T, requires_grad=True)
    # my_module.linear_3.weights = Tensor(torch_module.linear_3.weight.data.T, requires_grad=True)
    # my_module.linear_3.bias = Tensor(torch_module.linear_3.bias.data.T, requires_grad=True)

    # cProfile.run("train(my_module)", sort="tottime")

    start_time = time.perf_counter()

    train(my_module)

    print(
        "=" * 64
        + f"\n==> Deep Learner model trained in {time.perf_counter() - start_time:.2f} seconds."
    )
