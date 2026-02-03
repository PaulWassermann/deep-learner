"""Train one Deep Learner model and one PyTorch model on the same dataset.

This script exhibits a training example on a dataset using both Deep Learner and
PyTorch. Although some code could be shared, there is some duplication to
clearly show how it's currently done in Deep Learner VS how it's done in
PyTorch.
"""

import argparse
import time
from typing import Literal

import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, Dataset

import deep_learner as dl
from deep_learner._core.types import Device
from deep_learner.datasets import mnist, cifar10
from deep_learner.functional.functions import cross_entropy
from deep_learner.metrics.accuracy import accuracy
from deep_learner.nn.functional.dropout import Dropout
from deep_learner.nn.functional.relu import ReLU
from deep_learner.nn.functional.softmax import Softmax
from deep_learner.nn.layer.linear import Linear
from deep_learner.nn.module import Module
from deep_learner.nn.optimizer.sgd import SGD
from deep_learner.utils import batch


BATCH_SIZE: int = 64
EPOCHS: int = 10
LEARNING_RATE: float = 1e-2


# - DEEP LEARNER CODE ----------------------------------------------------------
class DeepLearnerModel(Module):
    def __init__(self, dataset: Literal["mnist", "cifar10"]):
        super().__init__()

        if dataset == "mnist":
            n_input = 784
        elif dataset == "cifar10":
            n_input = 3_072

        self.linear_1 = Linear(n_in=n_input, n_out=128)
        self.linear_2 = Linear(n_in=128, n_out=128)
        self.linear_3 = Linear(n_in=128, n_out=10)

        self.dropout = Dropout(drop_proba=0.1)
        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, x: dl.Tensor) -> dl.Tensor:
        return self.softmax(
            self.linear_3(
                self.dropout(self.relu(self.linear_2(self.relu(self.linear_1(x)))))
            )
        )


def dl_one_hot_encoding(x):
    # Assume x has 1 dimension for now
    return np.eye(x.max() + 1)[x]


def dl_preprocess(x) -> NDArray:
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-3)


def dl_load_dataset(
    dataset: Literal["mnist", "cifar10"],
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    if dataset == "mnist":
        train_X, train_Y, test_X, test_Y = mnist()
        train_X = dl_preprocess(train_X.reshape(-1, 784))
        train_Y = dl_one_hot_encoding(train_Y).astype(float)
        test_X = dl_preprocess(test_X.reshape(-1, 784))
        test_Y = dl_one_hot_encoding(test_Y).astype(float)
    elif dataset == "cifar10":
        train_X, train_Y, test_X, test_Y = cifar10()
        train_X = dl_preprocess(train_X)
        test_X = dl_preprocess(test_X)
    else:
        raise ValueError("Received dataset name {dataset}")

    return train_X, train_Y, test_X, test_Y


def dl_train(
    model: Module, dataset: Literal["mnist", "cifar10"], device: Device
) -> float:
    train_X, train_Y, test_X, test_Y = dl_load_dataset(dataset)

    optimizer = SGD(model, learning_rate=LEARNING_RATE)
    num_batches: int = train_X.shape[0] // BATCH_SIZE

    model.to(device)

    train_start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()

        total_loss = dl.Tensor(0)
        train_accuracy = dl.Tensor(0)

        for batch_X, batch_Y in batch(train_X, train_Y, batch_size=BATCH_SIZE):
            optimizer.zero_grad()

            batch_X = dl.Tensor(batch_X).to(device)
            batch_Y = dl.Tensor(batch_Y).to(device)

            y_hat = model(batch_X)

            loss = cross_entropy(y_hat, batch_Y)

            loss.backward()

            optimizer.step()

            total_loss = total_loss + loss.detach().to(Device.CPU)

            train_accuracy = train_accuracy + accuracy(
                dl.Tensor(np.argmax(y_hat.detach().to(Device.CPU).data, axis=-1)),
                dl.Tensor(np.argmax(batch_Y.detach().to(Device.CPU).data, axis=-1)),
            )

        model.eval()
        test_y_hat = model(dl.Tensor(test_X).to(device))

        test_accuracy = accuracy(
            dl.Tensor(np.argmax(test_y_hat.detach().to(Device.CPU).data, axis=-1)),
            dl.Tensor(np.argmax(test_Y, axis=-1)),
        ).data

        cumulative_time = time.perf_counter() - train_start

        print(
            f"Epoch {epoch + 1:>2}: "
            f"loss={total_loss.data / num_batches:.4f}, "
            f"train accuracy={train_accuracy.data / num_batches:2.2%}, "
            f"test accuracy={test_accuracy:2.2%} "
            f"[{cumulative_time // 60:0>2.0f}:{cumulative_time % 60:0>2.0f}]"
        )

    return test_accuracy


# - PYTORCH --------------------------------------------------------------------
def pt_preprocessing(images) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images, dtype=torch.float32)
    return (images - torch.std(images)) / torch.mean(images)


class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()

        self.images = pt_preprocessing(images)
        self.labels = torch.argmax(torch.tensor(labels), dim=-1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()

        self.images = pt_preprocessing(images).reshape(-1, 784)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class PyTorchModel(nn.Module):
    def __init__(self, dataset: Literal["mnist", "cifar10"]):
        super().__init__()

        if dataset == "mnist":
            in_features = 784
        elif dataset == "cifar10":
            in_features = 3_072

        self.linear_1 = nn.Linear(in_features=in_features, out_features=128)
        self.linear_2 = nn.Linear(in_features=128, out_features=128)
        self.linear_3 = nn.Linear(in_features=128, out_features=10)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear_3(
            self.relu(self.dropout(self.linear_2(self.relu(self.linear_1(x)))))
        )


def pt_train(
    model: nn.Module, dataset: Literal["mnist", "cifar10"], device: str
) -> float:
    if dataset == "mnist":
        train_X, train_Y, test_X, test_Y = mnist()

        train_dataloader = DataLoader(
            MNISTDataset(train_X, train_Y),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            MNISTDataset(test_X, test_Y),
            batch_size=1_024,
            shuffle=True,
        )
    elif dataset == "cifar10":
        train_X, train_Y, test_X, test_Y = cifar10()
        train_dataloader = DataLoader(
            CIFAR10Dataset(train_X, train_Y),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        test_dataloader = DataLoader(
            CIFAR10Dataset(test_X, test_Y),
            batch_size=1_024,
            shuffle=True,
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    model.to(device)

    train_start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0
        train_accuracy = 0

        for batch_X, batch_y in train_dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            y_hat = model(batch_X)

            loss = F.cross_entropy(y_hat, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            train_accuracy += (
                torch.argmax(y_hat, dim=-1) == batch_y
            ).sum().item() / len(batch_y)

        model.eval()
        test_accuracy = 0

        for batch_X, batch_y in test_dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            y_hat = model(batch_X)
            test_accuracy += (
                torch.argmax(y_hat, dim=-1) == batch_y
            ).sum().item() / len(batch_y)

        cumulative_time = time.perf_counter() - train_start

        print(
            f"Epoch {epoch + 1:>2}: "
            f"loss={total_loss / len(train_dataloader):.4f}, "
            f"train accuracy={train_accuracy / len(train_dataloader):.2%}, "
            f"test accuracy={test_accuracy / len(test_dataloader):.2%} "
            f"[{cumulative_time // 60:02.0f}:{cumulative_time % 60:02.0f}]"
        )

    return test_accuracy / len(test_dataloader)


def main(args: argparse.Namespace) -> int:
    # Deep Learner training
    device = Device(args.device)

    dl_model = DeepLearnerModel(dataset=args.dataset)
    print(f"\nTraining Deep Learner model on {args.dataset}:")
    print(f"--------------------------------{'-' * len(args.dataset)}\n")
    start_time = time.perf_counter()
    dl_accuracy = dl_train(dl_model, args.dataset, device)
    dl_training_time = time.perf_counter() - start_time
    print(
        "=" * 74 + "\n==> Deep Learner model trained in "
        f"{dl_training_time:.2f} seconds.\n\n"
    )

    # PyTorch training
    pt_model = PyTorchModel(dataset=args.dataset)
    print(f"Training PyTorch model on {args.dataset}:")
    print(f"---------------------------{'-' * len(args.dataset)}\n")
    start_time = time.perf_counter()
    pytorch_accuracy = pt_train(pt_model, args.dataset, args.device)
    pytorch_training_time = time.perf_counter() - start_time
    print(
        "=" * 74 + f"\n==> PyTorch model trained in "
        f"{pytorch_training_time:.2f} seconds.\n\n"
    )

    # Results summary
    timing_ratio = max(
        pytorch_training_time / dl_training_time,
        dl_training_time / pytorch_training_time,
    )
    accuracy_ratio = max(dl_accuracy / pytorch_accuracy, pytorch_accuracy / dl_accuracy)

    print("Summary:")
    print("--------\n")
    print(
        f"* TIMING  : Deep Learner trained {timing_ratio:>2.2f}x "
        f"{'faster' if dl_training_time < pytorch_training_time else 'slower'}"
        " than PyTorch"
    )
    print(
        f"* ACCURACY: Deep Learner scored  {accuracy_ratio:>2.2f}x "
        f"{'greater' if dl_accuracy > pytorch_accuracy else 'lower'} "
        "than PyTorch"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare the training process of a simple model in Deep "
        "Learner and PyTorch."
    )

    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "-d",
        "--device",
        choices=("cpu", "cuda"),
        required=True,
        help="The device to use when training the model.",
    )
    input_group.add_argument(
        "--dataset",
        choices=("mnist", "cifar10"),
        required=True,
        default="mnist",
        help="Dataset used for model training.",
    )

    cl_args = parser.parse_args()

    exit(main(cl_args))
