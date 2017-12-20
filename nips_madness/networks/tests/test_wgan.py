import numpy as np
import pytest

from .. import wgan


TEST_PARAMS = dict(
    J0=np.ones((2, 2)) * 0.01,
    D0=np.ones((2, 2)) * 0.01,
    S0=np.ones((2, 2)) * 0.01,
    gen=dict(
        J_min=1e-3,
        J_max=10,
        D_min=1e-3,
        D_max=10,
        S_min=1e-3,
        S_max=10,
        dynamics_cost=1,
    ),
    disc=dict(
        layers=[],
        normalization='none',
        nonlinearity='rectify',
    ),
    critic_iters_init=1,
    critic_iters=1,
    include_inhibitory_neurons=True,
    lipschitz_cost=10,
    truth_size=1,
)


def emit_gan(**kwargs):
    config = dict(
        TEST_PARAMS,
        **kwargs)
    return wgan.make_gan(config)


def fake_data(gan, truth_size):
    ncols = len(gan.bandwidths) * len(gan.contrasts) * len(gan.sample_sites)
    if gan.include_inhibitory_neurons:
        ncols *= 2
    return gan.rng.randn(truth_size, ncols)


@pytest.mark.parametrize('config', [
    dict(ssn_type='heteroin'),
])
def test_smoke_wgan(config):
    gan, rest = emit_gan(**config)
    data = fake_data(gan, rest['truth_size'])
    gan.set_dataset(data)
    learning_it = gan.learning()

    info = next(learning_it)
    assert info.is_discriminator

    info = next(learning_it)
    assert not info.is_discriminator


def test_wgan_heteroin():
    gan, _rest = emit_gan(ssn_type='heteroin')
    # Those attributes must exist:
    gan.gen.model.stimulator.V
    gan.gen_trainer.V_min
    gan.gen_trainer.V_max
