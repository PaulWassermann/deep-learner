from pathlib import Path

import cifar10_web
import mnist as mnist_ext
from numpy.typing import NDArray

DATASETS_DIR = Path(__file__).parent.joinpath("data").absolute()

if not DATASETS_DIR.exists():
    DATASETS_DIR.mkdir(parents=False)


def cifar10() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    return cifar10_web.cifar10(DATASETS_DIR.joinpath("cifar10").as_posix())


def mnist() -> tuple[NDArray, NDArray, NDArray, NDArray]:
    mnist_dir = DATASETS_DIR.joinpath("mnist")
    mnist_ext.temporary_dir = lambda: mnist_dir.as_posix()
    mnist_ext.datasets_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

    mnist_dir.mkdir(parents=True, exist_ok=True)

    return (
        mnist_ext.train_images(),
        mnist_ext.train_labels(),
        mnist_ext.test_images(),
        mnist_ext.test_labels(),
    )
