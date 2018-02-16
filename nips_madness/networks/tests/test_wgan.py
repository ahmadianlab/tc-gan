import numpy as np
import pytest
import theano

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
        rate_cost=100,
        rate_penalty_threshold=200,
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
    dict(ssn_type='deg-heteroin'),
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


@pytest.mark.parametrize('ssn_type', ['heteroin', 'deg-heteroin'])
def test_wgan_heteroin(ssn_type):
    gan, _rest = emit_gan(ssn_type=ssn_type)
    # Those attributes must exist:
    gan.gen.model.stimulator.V
    gan.gen_trainer.V_min
    gan.gen_trainer.V_max


def normalize_to_gen_config(config):
    new = dict(wgan.DEFAULT_PARAMS)
    new.update(TEST_PARAMS)
    new.update(config)
    return normalize_to_gen_config_common(new)


def normalize_to_gen_config_common(config):
    bandwidths = config['bandwidths']
    contrasts = config['contrasts']
    config.setdefault('num_tcdom', len(bandwidths) * len(contrasts))

    num_sites = config['num_sites']
    include_inhibitory_neurons = config['include_inhibitory_neurons']

    for key in 'JDSV':
        key0 = key + '0'
        if key0 in config:
            config[key] = config.pop(key0)

    for key in 'JDSV':
        if key in config:
            config[key] = np.asarray(config[key], dtype=theano.config.floatX)

    for key in ['gen', 'disc',
                'bandwidths', 'contrasts', 'include_inhibitory_neurons',
                'critic_iters', 'critic_iters_init', 'lipschitz_cost',
                'truth_size']:
        config.pop(key, None)

    try:
        sample_sites = config.pop('sample_sites')
    except KeyError:
        pass
    else:
        assert 'probes' not in config
        config['probes'] = wgan.probes_from_stim_space(
            sample_sites, num_sites, include_inhibitory_neurons)

    config.setdefault('unroll_scan', False)
    config.setdefault('include_rate_penalty', True)
    config.setdefault('include_time_avg', False)
    config.setdefault('ssn_type', 'default')
    config.setdefault('ssn_impl', 'default')

    return config


@pytest.mark.parametrize('config', [
    {},
    dict(ssn_type='heteroin'),
    dict(ssn_type='deg-heteroin'),
])
def test_wgan_gen_to_config(config, emit_gan=emit_gan,
                            normalize=normalize_to_gen_config):
    desired = normalize(config)
    gan, _rest = emit_gan(**config)
    actual = gan.gen.to_config()
    print('set(actual) - set(desired) =', set(actual) - set(desired))
    print('set(desired) - set(actual) =', set(desired) - set(actual))
    np.testing.assert_equal(actual, desired)
