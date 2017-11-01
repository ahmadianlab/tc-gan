from types import SimpleNamespace

import numpy as np
import pytest

from ..cwgan import (
    ConditionalMinibatch, ConditionalProber,
    RandomChoiceSampler, NaiveRandomChoiceSampler,
)
from .test_conditional_prober import mock_model


def arangemd(shape, **kwargs):
    return np.arange(np.prod(shape), **kwargs).reshape(shape)


def make_conditional_minibatch(num_models=2, probes_per_model=3,
                               num_bandwidths=5):
    tc_md = arangemd((num_models, probes_per_model, num_bandwidths))
    conditions_md = arangemd((3, num_models, probes_per_model))
    bandwidths = np.arange(num_bandwidths)
    contrasts = np.arange(num_models)
    conditions_md[0] = contrasts.reshape(-1, 1)
    # See: [[../cwgan.py::select_minibatch]]
    return ConditionalMinibatch(tc_md, conditions_md, bandwidths, contrasts)


def access_all_attrs(obj):
    """ Access all attributes of `obj`, mostly for smoke tests. """
    for name in dir(obj):
        getattr(obj, name)


@pytest.mark.parametrize('num_models, probes_per_model, num_bandwidths', [
    (1, 3, 5),
    (2, 3, 5),
])
def test_conditional_minibatch(num_models, probes_per_model, num_bandwidths):
    batch = make_conditional_minibatch(num_models, probes_per_model,
                                       num_bandwidths)
    access_all_attrs(batch)

    assert batch.num_models == num_models
    assert batch.probes_per_model == probes_per_model
    assert batch.num_bandwidths == num_bandwidths

    assert batch.conditions.shape == (batch.batchsize, 3)

    gen_kwargs = SimpleNamespace(**batch.gen_kwargs)
    assert len(gen_kwargs.prober_probe_offsets) == batch.batchsize
    assert len(gen_kwargs.prober_cell_types) == batch.batchsize
    assert len(gen_kwargs.prober_model_ids) == batch.batchsize

    stimulator_shape = (num_models, num_bandwidths)
    assert gen_kwargs.stimulator_bandwidths.shape == stimulator_shape
    assert gen_kwargs.stimulator_contrasts.shape == stimulator_shape


@pytest.mark.parametrize('num_models, probes_per_model', [
    (1, 3),
    (2, 3),
    (5, 7),
    (8, 64),
])
def test_conditional_minibatch_prober_contrasts(num_models, probes_per_model):
    batch = make_conditional_minibatch(num_models, probes_per_model)
    prober = ConditionalProber(mock_model())

    gen_kwargs = SimpleNamespace(**batch.gen_kwargs)
    prober_contrasts = prober.contrasts.eval({
        prober.model.stimulator.contrasts: gen_kwargs.stimulator_contrasts,
        prober.model_ids: batch.model_ids,
    })

    batch_contrasts, _probe_offsets, _cell_types = batch.conditions.T
    # See:
    # * [[../cwgan.py::get_output]]
    # * [[../cwgan.py::gen_kwargs]]

    np.testing.assert_equal(prober_contrasts, batch_contrasts)


@pytest.mark.parametrize('sampler_class', [
    RandomChoiceSampler,
    NaiveRandomChoiceSampler,
])
def test_random_choice_sampler_shape(sampler_class):
    truth_size = 100
    num_bandwidths = 5
    num_contrasts = 2
    num_offsets = 6
    num_cell_types = 2
    shape = (truth_size, num_cell_types, num_offsets, num_contrasts,
             num_bandwidths)
    sampler = sampler_class(
        arangemd(shape),
        [np.arange(num) for num in shape[1:]],
    )

    num_models = 8
    probes_per_model = num_offsets * num_cell_types
    repeat = 3
    for _, batch in zip(range(repeat),
                        sampler.random_minibatches(num_models,
                                                   probes_per_model)):
        access_all_attrs(batch)
        assert batch.num_models == num_models
        assert batch.probes_per_model == probes_per_model
        assert batch.num_bandwidths == num_bandwidths


def test_random_choice_sampler_cells():
    truth_size = 100
    num_bandwidths = 5
    num_contrasts = 2
    num_offsets = 6
    num_cell_types = 2
    shape = (truth_size, num_cell_types, num_offsets, num_contrasts,
             num_bandwidths)
    sampler = RandomChoiceSampler(
        arangemd(shape),
        [np.arange(num) for num in shape[1:]],
    )

    num_models = 8
    probes_per_model = num_offsets * num_cell_types
    repeat = 3

    print()
    for n in range(repeat):
        print('{}-th repeat -- model: '.format(n), end='')
        ids_cell_type, ids_probe_offsets \
            = sampler.random_cells(num_models, probes_per_model)

        for im in range(num_models):
            print(im, end=' ')
            cells = set(zip(ids_cell_type[im], ids_probe_offsets[im]))
            assert len(cells) == probes_per_model
        print()
