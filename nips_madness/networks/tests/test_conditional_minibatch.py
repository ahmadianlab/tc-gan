from types import SimpleNamespace

import numpy as np
import pytest

from ..cwgan import ConditionalMinibatch, ConditionalProber
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


@pytest.mark.parametrize('num_models, probes_per_model, num_bandwidths', [
    (1, 3, 5),
    (2, 3, 5),
])
def test_conditional_minibatch(num_models, probes_per_model, num_bandwidths):
    batch = make_conditional_minibatch(num_models, probes_per_model,
                                       num_bandwidths)

    # Smoke tests -- all dynamical properties have to raise no error:
    for name in dir(batch):
        getattr(batch, name)

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
