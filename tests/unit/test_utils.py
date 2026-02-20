import numpy as np
import pytest

from deep_learner import Tensor
from deep_learner.utils import (
    _batched,
    batch,
    constant,
    ones,
    rand_int,
    rand_normal,
    rand_uniform,
    zeros,
)


# FIXTURES =====================================================================
@pytest.fixture
def data():
    return np.array(range(5))


@pytest.fixture
def other_data():
    return np.array(range(6, 15))


# UNIT TESTS ===================================================================


# _batched / batch -------------------------------------------------------------
def test_batched(data):
    batches = list(_batched(data, n=2))

    assert batches == [(0, 1), (2, 3), (4,)]


def test_batched_size_one(data):
    batches = list(_batched(data, n=1))

    assert batches == [(i,) for i in data]


def test_batched_invalid_size(data):
    with pytest.raises(ValueError):
        list(_batched(data, n=0))


def test_batched_size_overflow(data):
    batches = list(_batched(data, n=6))

    assert batches == [tuple(data)]


def test_batched_multiple_times_on_iterator(data):
    _ = list(_batched(data, n=2))
    batches = list(_batched(data, n=2))

    assert batches == [(0, 1), (2, 3), (4,)]


def test_batched_multiple_times_on_generator():
    data = (i for i in range(5))
    _ = list(_batched(data, n=2))
    batches = list(_batched(data, n=2))

    assert batches == []


def test_batch_single_iterable_no_shuffle(data):
    batches = list(batch(data, batch_size=2, shuffle=False))

    # Note the tuple containing the data tuples; a normal usage of the function
    # would be to unpack the outer tuples
    assert batches == [((0, 1),), ((2, 3),), ((4,),)]


def test_batch_two_iterables_no_shuffle(data, other_data):
    batches = list(batch(data, other_data, batch_size=2, shuffle=False))

    # All iterables longer than the shortest one are truncated
    assert batches == [((0, 1), (6, 7)), ((2, 3), (8, 9)), ((4,), (10,))]


def test_batch_shuffle():
    data = list(range(1_000))

    batches = batch(data, batch_size=1, shuffle=True)

    # When shuffling data, there are 1000! possible combinations; therefore,
    # it is very unlikely that the random shuffle preserves the initial order
    assert batches != [(i,) for i in data]


# constant / ones / zeros ------------------------------------------------------
@pytest.mark.parametrize(
    ("shape", "value", "requires_grad"), [((2, 4), 3.14, False), ((1,), 42, True)]
)
def test_constant(shape, value, requires_grad):
    t = constant(shape, value, requires_grad)

    assert t.data.shape == shape
    assert (t.data == value).all()
    assert t.requires_grad == requires_grad


@pytest.mark.parametrize(("shape", "requires_grad"), [((7, 2), True), ((1, 1), False)])
def test_ones(shape, requires_grad):
    t = ones(shape, requires_grad)

    assert t.data.shape == shape
    assert (t.data == 1).all()
    assert t.requires_grad == requires_grad


@pytest.mark.parametrize(("shape", "requires_grad"), [((2, 5), False), ((5, 2), True)])
def test_zeros(shape, requires_grad):
    t = zeros(shape, requires_grad)

    assert t.data.shape == shape
    assert (t.data == 0).all()
    assert t.requires_grad == requires_grad


# rand_int / rand_normal / rand_uniform ----------------------------------------
@pytest.mark.parametrize(
    ("shape", "low", "high", "requires_grad"),
    [((10, 10), 5, 100, True), ((1,), -3, 3, False)],
)
def test_rand_int(shape, low, high, requires_grad):
    t = rand_int(shape, low, high, requires_grad)

    assert t.data.shape == shape
    assert (low <= t.data).all()
    assert (t.data < high).all()
    assert t.requires_grad == requires_grad


@pytest.mark.parametrize(
    ("shape", "mean", "std", "requires_grad"),
    [((10, 10), 0, 1, True), ((20, 20), 3.14, 2, False)],
)
def test_rand_normal(shape, mean, std, requires_grad):
    mean_arr = mean * np.ones(shape)
    std_arr = Tensor(std * np.ones(shape))

    t = rand_normal(mean_arr, std_arr, requires_grad)

    assert t.data.shape == shape
    assert abs(t.data.mean() - mean) < abs(t.data.mean() - 10 * mean) + 1e-6


@pytest.mark.parametrize(
    ("shape", "low", "high", "requires_grad"),
    [((3, 3), 0, 1, True), ((2, 16), -4, 10, False)],
)
def test_random_uniform(shape, low, high, requires_grad):
    t = rand_uniform(shape, low, high, requires_grad)

    assert t.data.shape == shape
    assert (low <= t.data).all()
    assert (t.data < high).all()
    assert t.requires_grad == requires_grad
