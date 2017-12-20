import pytest

from .. import cwgan
from .test_wgan import TEST_PARAMS


def make_gan(norm_probes=[0],
             **kwargs):
    config = dict(
        TEST_PARAMS,
        norm_probes=norm_probes,
        **kwargs)
    del kwargs
    include_inhibitory_neurons = config['include_inhibitory_neurons']
    if 'probes_per_model' not in config:
        probes_per_model = len(norm_probes)
        if include_inhibitory_neurons:
            probes_per_model *= 2
        config['probes_per_model'] = probes_per_model

    return cwgan.make_gan(config)


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
