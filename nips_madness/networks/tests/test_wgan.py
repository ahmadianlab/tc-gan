import numpy as np

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


def make_gan(**kwargs):
    config = dict(
        TEST_PARAMS,
        **kwargs)
    return wgan.make_gan(config)


def test_wgan_heteroin():
    gan, _rest = make_gan(ssn_type='heteroin')
    # Those attributes must exist:
    gan.gen.model.stimulator.V
    gan.gen_trainer.V_min
    gan.gen_trainer.V_max
