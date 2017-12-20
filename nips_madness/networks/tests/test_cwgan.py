import numpy as np
import pytest

from .. import cwgan


def make_gan(norm_probes=[0],
             include_inhibitory_neurons=True,
             **kwargs):
    probes_per_model = len(norm_probes)
    if include_inhibitory_neurons:
        probes_per_model *= 2
    kwargs.setdefault('probes_per_model', probes_per_model)

    return cwgan.make_gan(dict(dict(
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
        norm_probes=norm_probes,
        include_inhibitory_neurons=include_inhibitory_neurons,
        lipschitz_cost=10,
        truth_size=1,
    ), **kwargs))


def fake_data(gan, truth_size):
    ncols = len(gan.bandwidths) * len(gan.contrasts) * len(gan.norm_probes)
    if gan.include_inhibitory_neurons:
        ncols *= 2
    return gan.rng.randn(truth_size, ncols)


@pytest.mark.parametrize('config', [
    {},
    dict(hide_cell_type=True),
    dict(ssn_type='heteroin'),
])
def test_smoke_cgan(config):
    gan, rest = make_gan(**config)
    data = fake_data(gan, rest['truth_size'])
    gan.set_dataset(data)
    learning_it = gan.learning()

    info = next(learning_it)
    assert info.is_discriminator

    info = next(learning_it)
    assert not info.is_discriminator


def test_hide_cell_type():
    gan, _rest = make_gan(hide_cell_type=True)
    assert isinstance(gan.disc, cwgan.CellTypeBlindDiscriminator)


def test_cwgan_heteroin():
    gan, _rest = make_gan(ssn_type='heteroin')
    # Those attributes must exist:
    gan.gen.model.stimulator.V
    gan.gen_trainer.V_min
    gan.gen_trainer.V_max
