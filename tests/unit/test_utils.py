import numpy as np
import pytest

from deep_learner.utils import (
    _batched,
    batch,
    constant,
    ones,
    rand_int,
    rand_normal,
    rand_uniform,
    zeros
)

# FIXTURES
@pytest.fixture
def data():
    return np.array(range(5))


@pytest.fixture
def other_data():
    return np.array(range(6, 15))


# UNIT TESTS
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
    assert batches == [
        ((0, 1), (6, 7)),
        ((2, 3), (8, 9)),
        ((4,), (10,))
    ]

