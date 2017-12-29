import pytest

from . import test_wgan
from .. import cwgan

TEST_PARAMS = dict(
    test_wgan.TEST_PARAMS,
    norm_probes=[0],
)


def make_config(**kwargs):
    config = dict(
        TEST_PARAMS,
        **kwargs)
    del kwargs
    norm_probes = config['norm_probes']
    include_inhibitory_neurons = config['include_inhibitory_neurons']
    if 'probes_per_model' not in config:
        probes_per_model = len(norm_probes)
        if include_inhibitory_neurons:
            probes_per_model *= 2
        config['probes_per_model'] = probes_per_model
    return config


def emit_gan(**kwargs):
    return cwgan.make_gan(make_config(**kwargs))


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
    gan, rest = emit_gan(**config)
    data = fake_data(gan, rest['truth_size'])
    gan.set_dataset(data)
    learning_it = gan.learning()

    info = next(learning_it)
    assert info.is_discriminator

    info = next(learning_it)
    assert not info.is_discriminator


def test_hide_cell_type():
    gan, _rest = emit_gan(hide_cell_type=True)
    assert isinstance(gan.disc, cwgan.CellTypeBlindDiscriminator)


@pytest.mark.parametrize('ssn_type', ['heteroin', 'deg-heteroin'])
def test_cwgan_heteroin(ssn_type):
    gan, _rest = emit_gan(ssn_type=ssn_type)
    # Those attributes must exist:
    gan.gen.model.stimulator.V
    gan.gen_trainer.V_min
    gan.gen_trainer.V_max


def normalize_to_gen_config(config):
    mixed = make_config(**config)
    config = dict(cwgan.DEFAULT_PARAMS)
    config.update(mixed)

    bandwidths = config['bandwidths']
    config.setdefault('num_tcdom', len(bandwidths))

    num_models = config.pop('num_models')
    probes_per_model = config.pop('probes_per_model')
    config.setdefault('batchsize', num_models * probes_per_model)

    test_wgan.normalize_to_gen_config_common(config)

    for key in ['hide_cell_type', 'e_ratio', 'norm_probes']:
        config.pop(key, None)
    return config


@pytest.mark.parametrize('config', [
    {},
    dict(ssn_type='heteroin'),
    dict(ssn_type='deg-heteroin'),
])
def test_cwgan_gen_to_config(config):
    test_wgan.test_wgan_gen_to_config(config,
                                      emit_gan,
                                      normalize_to_gen_config)
