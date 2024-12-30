import argparse
import cProfile
import time

import mnist
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from deep_learner.datasets import mnist


def preprocessing(images) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        images = torch.tensor(images, dtype=torch.float32)
    return (images - torch.std(images)) / torch.mean(images)


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        super().__init__()

        self.images = preprocessing(images).reshape(-1, 784)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


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


def train(model: nn.Module, device: str):
    train_X, train_Y, test_X, test_Y = mnist()

    train_dataloader = DataLoader(
        MNISTDataset(train_X, train_Y),
        batch_size=64,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        MNISTDataset(test_X, train_Y),
        batch_size=1_024,
        shuffle=True,
    )

    epochs = 10
    optimizer = SGD(model.parameters(), lr=1e-2)

    model.to(device)

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        train_accuracy = 0

        for batch_X, batch_y in train_dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            y_hat = model(batch_X)

            loss = cross_entropy(y_hat, batch_y)

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

        print(
            f"{epoch=}, "
            f"loss={total_loss / len(train_dataloader):.4f}, "
            f"train accuracy={train_accuracy / len(train_dataloader):.2%}, "
            f"test accuracy={test_accuracy / len(test_dataloader):.2%}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program that trains a simple PyTorch MLP on MNIST."
    )

    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "-d",
        "--device",
        choices=("cpu", "cuda"),
        required=True,
        help="The device to use when training the model.",
    )

    cl_args = parser.parse_args()

    my_model = Model()

    start_time = time.perf_counter()

    # cProfile.run("train(my_model)", sort="tottime"

    train(my_model, cl_args.device)

    print(
        "=" * 64
        + f"\n==> PyTorch model trained in {time.perf_counter() - start_time:.2f} seconds."
    )
