from types import SimpleNamespace

import numpy as np
import pytest

from ..cwgan import ConditionalMinibatch


def arangemd(shape, **kwargs):
    return np.arange(np.prod(shape), **kwargs).reshape(shape)


def make_conditional_minibatch(num_models=2, probes_per_model=3,
                               num_bandwidths=5):
    tc_md = arangemd((num_models, probes_per_model, num_bandwidths))
    conditions_md = arangemd((3, num_models, probes_per_model))
    bandwidths = np.arange(num_bandwidths)
    contrasts = np.arange(num_models)
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
