import itertools
import multiprocessing

import numpy as np
import pytest

from ..utils import cpu_count, random_minibatches


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def test_cpu_count():
    assert cpu_count(_environ={'SLURM_CPUS_ON_NODE': '-3'}) == -3
    assert cpu_count(_environ={'PBS_NUM_PPN': '-5'}) == -5
    assert cpu_count(_environ={}) == multiprocessing.cpu_count()


def test_random_minibatches_batchsize_too_large():
    data = list(range(10))
    batchsize = 11
    with pytest.raises(ValueError):
        random_minibatches(batchsize, data)


def test_random_minibatches_strict():
    data = list(range(15))
    batchsize = 11
    with pytest.raises(ValueError):
        random_minibatches(batchsize, data, strict=True)


def test_random_minibatches_nonstrict():
    data = list(range(15))
    batchsize = 11
    with pytest.warns(None) as record:
        random_minibatches(batchsize, data)
    assert len(record) == 1


def test_random_minibatches_first_batch():
    data = np.arange(128)
    batchsize = 16
    num_batches = 8  # = len(data) // batchsize
    dataiter = random_minibatches(batchsize, data, strict=True)
    batches = take(num_batches, dataiter)
    assert all(len(batch) == batchsize for batch in batches)
    actual = sorted(np.concatenate(batches))
    assert actual == list(data)
