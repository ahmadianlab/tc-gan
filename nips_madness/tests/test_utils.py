import multiprocessing

import pytest

from ..utils import cpu_count, random_minibatches


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
