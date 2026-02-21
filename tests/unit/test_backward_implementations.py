import pytest

from deep_learner import rand_int, rand_uniform
from deep_learner._gradient_checker import gradient_check_ss, gradient_check_vs
from deep_learner.functional.functions import (
    add,
    binary_cross_entropy,
    cross_entropy,
    divide,
    dropout,
    exponential,
    matmul,
    mean,
    mean_squared_error,
    multiply,
    power,
    relu,
    sigmoid,
    softsign,
    subtract,
    sum,
    tanh,
)

SHAPE = (3, 4)


@pytest.fixture
def a():
    return rand_uniform(SHAPE, -3, 3, requires_grad=True)


@pytest.fixture
def b():
    return rand_uniform(SHAPE, -1, 6, requires_grad=True)


@pytest.fixture
def probs():
    return rand_uniform(SHAPE, 1e-3, 1 - 1e-3, requires_grad=True)


@pytest.fixture
def binary_labels():
    return rand_int(SHAPE, 0, 2, requires_grad=True)


def test_add(a, b):
    gradient_check_ss(add, a, b)


def test_binary_cross_entropy(probs, binary_labels):
    gradient_check_vs(binary_cross_entropy, probs, binary_labels)


def test_cross_entropy(probs, binary_labels):
    gradient_check_vs(cross_entropy, probs, binary_labels)


def test_divide(a, b):
    gradient_check_ss(divide, a, b)


def test_exponential(a):
    gradient_check_ss(exponential, a)


def test_mean(a):
    gradient_check_vs(mean, a)


def test_mean_squared_error(a, b):
    gradient_check_vs(mean_squared_error, a, b)
