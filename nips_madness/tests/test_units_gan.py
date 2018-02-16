"""
Unit tests for GAN.
"""

import numpy as np
import pytest

from ..conftest import old_gan
from ..drivers import GANDriver
from ..gradient_expressions.utils import subsample_neurons, \
    sample_sites_from_stim_space
from ..run import gan as run_gan
from ..run.gan import GenerativeAdversarialNetwork, setup_gan, train_gan
from .. import ssnode
from .test_legacy_drivers import fake_datastore


def make_gan(
        n_sites=ssnode.DEFAULT_PARAMS['N'],
        loss_type='WD',
        sample_sites=[0],
        rate_cost=50000,
        track_offset_identity=True,
        include_inhibitory_neurons=False,
        WGAN_n_critic0=50,
        WGAN_n_critic=5,
        **kwargs):
    ssn_params = ssnode.DEFAULT_PARAMS.copy()
    for key in ['N', 'J', 'D', 'S', 'bandwidths', 'contrast', 'smoothness',
                'offset']:
        del ssn_params[key]
    ssn_params['r0'] = np.zeros(2 * n_sites)

    default_kwargs = dict(
        ssn_params=ssn_params,
        J0=ssnode.DEFAULT_PARAMS['J'],
        D0=ssnode.DEFAULT_PARAMS['D'],
        S0=ssnode.DEFAULT_PARAMS['S'],
        init_disturbance=-1.5,
        smoothness=ssnode.DEFAULT_PARAMS['smoothness'],
        bandwidths=ssnode.DEFAULT_PARAMS['bandwidths'],
        contrast=ssnode.DEFAULT_PARAMS['contrast'],
        n_sites=n_sites,
        n_samples=1,
        layers=[8],
        disc_normalization='none',
        disc_nonlinearity='rectify',
        disc_l1_regularization=0,
        disc_l2_regularization=0,
        gen_learn_rate=0.01,
        disc_learn_rate=0.0001,
        rate_penalty_threshold=150,
        rate_penalty_no_I=False,
        WGAN_lambda=10,
        gen_update='adam-wgan',
        disc_update='adam-wgan',
        gen_param_type='log',
        J_min=1e-3, J_max=10,
        D_min=1e-3, D_max=10,
        S_min=1e-3, S_max=10,
    )
    kwargs = dict(default_kwargs, **kwargs)

    gan = GenerativeAdversarialNetwork()
    gan.track_offset_identity = track_offset_identity
    gan.include_inhibitory_neurons = include_inhibitory_neurons
    gan.rate_cost = rate_cost
    gan.loss_type = loss_type

    n_sites = kwargs['n_sites']
    gan.sample_sites = sample_sites_from_stim_space(sample_sites, n_sites)

    gan.subsample_kwargs = dict(
        track_offset_identity=track_offset_identity,
        include_inhibitory_neurons=include_inhibitory_neurons,
    )

    setup_gan(
        gan,
        n_stim=len(kwargs['bandwidths']) * len(kwargs['contrast']),
        **kwargs)
    gan.WGAN_n_critic0 = WGAN_n_critic0
    gan.WGAN_n_critic = WGAN_n_critic
    return gan


def make_driver(
        gan, datastore,
        iterations=1, quiet=True,
        disc_param_save_interval=-1,
        disc_param_template="last.npz",
        disc_param_save_on_error=5,
        quit_JDS_threshold=-1,
        ):
    driver = GANDriver(
        gan, datastore,
        iterations=iterations, quiet=quiet,
        disc_param_save_interval=disc_param_save_interval,
        disc_param_template=disc_param_template,
        disc_param_save_on_error=disc_param_save_on_error,
        quit_JDS_threshold=quit_JDS_threshold,
    )
    return driver


def mock_data(gan, truth_size=200):
    shape = (truth_size, gan.NB)
    return np.arange(0, np.prod(shape)).reshape(shape)


@old_gan
@pytest.mark.parametrize('sample_sites', [[0], [0, 1], [0, 0.5, 1]])
@pytest.mark.parametrize('track_offset_identity', [False, True])
@pytest.mark.parametrize('include_inhibitory_neurons', [False, True])
def test_get_reduced_equal_subsample_neurons(
        sample_sites,
        track_offset_identity, include_inhibitory_neurons,
        ):
    gan = make_gan(
        sample_sites=sample_sites,
        track_offset_identity=track_offset_identity,
        include_inhibitory_neurons=include_inhibitory_neurons,
    )
    rate_shape = (gan.NZ, gan.NB, 2 * gan.N)
    rate = np.arange(np.prod(rate_shape)).reshape(rate_shape)
    reduced = gan.get_reduced(rate)
    subsampled = subsample_neurons(rate, gan.sample_sites,
                                   **gan.subsample_kwargs)
    np.testing.assert_equal(reduced, subsampled)


@old_gan
def test_smoke_train_gan():
    gan = make_gan()
    datastore = fake_datastore()
    driver = make_driver(gan, datastore)
    gan.data = mock_data(gan)
    train_gan(driver, gan, datastore, **vars(gan))


def prepped(**kwds):
    run_config = dict(
        bandwidths=[0.0625, 0.125, 0.25, 0.5, 0.75],
        J0=ssnode.DEFAULT_PARAMS['J'],
        D0=ssnode.DEFAULT_PARAMS['D'],
        S0=ssnode.DEFAULT_PARAMS['S'],
    )
    for key in ('J0', 'D0', 'S0'):
        run_config[key] = np.asarray(run_config[key]).tolist()
    run_config.update(kwds)
    return run_config


@pytest.mark.parametrize('run_config, desired', [
    ({}, prepped()),
    (dict(S0=[[1, 1], [1, 1]]), prepped(S0=[[1, 1], [1, 1]])),
    (dict(S0=1), prepped(S0=[[1, 1], [1, 1]])),
])
def test_preprocess(run_config, desired):
    run_config_0 = run_config
    run_config = dict(
        n_bandwidths=len(prepped()['bandwidths']),
        load_gen_param=None,
    )
    run_config.update(run_config_0)

    run_gan.preprocess(run_config)
    actual = {k: run_config[k] for k in desired}
    assert actual == desired
