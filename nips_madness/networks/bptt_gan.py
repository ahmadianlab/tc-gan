from types import SimpleNamespace
import itertools

import lasagne
import numpy as np

from .. import ssnode
from ..gradient_expressions.utils import sample_sites_from_stim_space
from ..utils import (
    cached_property, cartesian_product, random_minibatches, StopWatch,
    theano_function,
)
from .core import BaseComponent, consume_subdict
from .ssn import TuningCurveGenerator


DEFAULT_PARAMS = dict(
    bandwidths=ssnode.DEFAULT_PARAMS['bandwidths'],
    contrasts=ssnode.DEFAULT_PARAMS['contrast'],
    smoothness=ssnode.DEFAULT_PARAMS['smoothness'],
    sample_sites=[0],
    # Stimulator:
    num_sites=ssnode.DEFAULT_PARAMS['N'],
    # Model / SSN:
    k=ssnode.DEFAULT_PARAMS['k'],
    n=ssnode.DEFAULT_PARAMS['n'],
    tau_E=10,
    tau_I=1,
    dt=0.1,
    seqlen=1200,
    batchsize=1,
    skip_steps=1000,
)


class UnConditionalDiscriminator(BaseComponent):

    def __init__(self, shape, loss_type,
                 layers, normalization, nonlinearity):
        from .simple_discriminator import make_net
        self.l_out = make_net(
            shape, loss_type,
            layers, normalization, nonlinearity,
        )

        from theano import tensor as T
        self.xg = xg = T.matrix()  # \tilde x  --  "fake image"
        self.xd = xd = T.matrix()  # x         --  ground truth
        Dg = self.get_output(xg)
        Dd = self.get_output(xd)
        self.accuracy_expr = Dg.mean() - Dd.mean()

    @cached_property
    def accuracy(self):
        return theano_function([self.xg, self.xd], self.accuracy_expr)

    def get_all_params(self):
        return lasagne.layers.get_all_params(self.l_out, trainable=True)

    def get_output(self, inputs=None):
        return lasagne.layers.get_output(self.l_out, inputs=inputs)


class BaseTrainer(BaseComponent):

    # These are the values recommended by Gulrajani et al. (2017)
    # -- http://arxiv.org/abs/1704.00028
    learning_rate = 0.0001
    beta1 = 0.5
    beta2 = 0.9

    @cached_property
    def updates(self):
        params = self.target.get_all_params()
        return lasagne.updates.adam(
            self.loss, params, learning_rate=self.learning_rate,
            beta1=self.beta1, beta2=self.beta2,
        )

    @cached_property
    def train(self):
        return theano_function(self.inputs, self.loss, updates=self.updates)


class CriticTrainer(BaseTrainer):
    """
    Discriminator/Critic trainer for WGAN.
    """

    def __init__(self, disc):
        self.target = disc

        from theano import tensor as T
        self.lmd = lmd = T.scalar()
        self.xg = xg = T.matrix()  # \tilde x  --  "fake image"
        self.xd = xd = T.matrix()  # x         --  ground truth
        self.xp = xp = T.matrix()  # \hat x    --  interpolated sample
        self.disc_gene = disc.get_output(xg).mean()
        self.disc_data = disc.get_output(xd).mean()
        self.disc_intp = disc.get_output(xp)[:, 0]
        self.dgrad = T.jacobian(self.disc_intp, xp)
        self.penalty = ((self.dgrad.norm(2, axis=(1, 2)) - 1)**2).mean()

        self.loss = self.disc_gene - self.disc_data + lmd * self.penalty
        self.inputs = (xg, xd, xp, lmd)


class GeneratorTrainer(BaseTrainer):

    def __init__(self, gen, disc):
        self.target = self.gen = gen
        self.disc = disc

        self.loss = - disc.get_output(gen.get_output()).mean()
        self.inputs = gen.inputs


class BPTTWassersteinGAN(BaseComponent):

    def __init__(self, gen, disc, gen_trainer, disc_trainer,
                 bandwidths, contrasts,
                 critic_iters_init, critic_iters, lipschitz_cost,
                 seed=0):
        self.gen = gen
        self.disc = disc
        self.gen_trainer = gen_trainer
        self.disc_trainer = disc_trainer

        self.critic_iters_init = critic_iters_init
        self.critic_iters = critic_iters
        self.lipschitz_cost = lipschitz_cost
        self.rng = np.random.RandomState(seed)

        self.bandwidths = bandwidths
        self.contrasts = contrasts
        self.stimulator_contrasts, self.stimulator_bandwidths \
            = cartesian_product(contrasts, bandwidths)

    batchsize = property(lambda self: self.gen.model.batchsize)
    num_neurons = property(lambda self: self.gen.num_neurons)

    # To be compatible with GANDriver:
    loss_type = 'WD'
    discriminator = property(lambda self: self.disc.l_out)
    NZ = property(lambda self: self.batchsize)
    J = property(lambda self: self.gen.model.J)
    D = property(lambda self: self.gen.model.D)
    S = property(lambda self: self.gen.model.S)

    def init_dataset(self, data):
        self.dataset = random_minibatches(self.batchsize, data)

    def next_minibatch(self):
        return next(self.dataset)

    def prepare(self):
        # Force compile:
        self.gen._forward
        self.gen_trainer.train
        self.disc.accuracy
        self.disc_trainer.train

    def gen_forward(self, zs):
        return self.gen.forward(
            model_zs=zs,
            stimulator_bandwidths=self.stimulator_bandwidths,
            stimulator_contrasts=self.stimulator_contrasts,
        )

    def train_discriminator(self, info):
        xd = self.next_minibatch()
        eps = self.rng.rand(self.batchsize).reshape((-1, 1))
        zg = self.rng.rand(self.batchsize, self.num_neurons, self.num_neurons)
        with self.gen_forward_watch:
            gen_out = self.gen_forward(zg)
        xg = gen_out.prober_tuning_curve
        xp = eps * xd + (1 - eps) * xg
        with self.disc_train_watch:
            info.disc_loss = self.disc_trainer.train(
                xg, xd, xp, self.lipschitz_cost,
            )

        info.accuracy = self.disc.accuracy(xg, xd)
        info.gen_out = gen_out
        info.dynamics_penalty = gen_out.model_dynamics_penalty
        info.xd = xd
        info.xg = xg
        info.xp = xp
        info.gen_time = self.gen_forward_watch.times[-1]
        info.disc_time = self.disc_train_watch.times[-1]
        return info

    def train_generator(self, info):
        zg = self.rng.rand(self.batchsize, self.num_neurons, self.num_neurons)
        with self.gen_train_watch:
            info.gen_loss = self.gen_trainer.train(
                self.stimulator_bandwidths,
                self.stimulator_contrasts,
                zg,
            )

        info.gen_forward_time = self.gen_forward_watch.sum()
        info.gen_time = self.gen_train_watch.sum() + info.gen_forward_time
        info.disc_time = self.disc_train_watch.sum()
        return info

    def _single_gen_step(self, gen_step, critic_iters):
        self.gen_forward_watch = StopWatch()
        self.gen_train_watch = StopWatch()
        self.disc_train_watch = StopWatch()

        for disc_step in range(self.critic_iters_init):
            info = SimpleNamespace(is_discriminator=True, gen_step=gen_step,
                                   disc_step=disc_step)
            yield self.train_discriminator(info)

        info = SimpleNamespace(is_discriminator=False, gen_step=gen_step)
        yield self.train_generator(info)

    def learning(self):
        for info in self._single_gen_step(0, self.critic_iters_init):
            yield info
        for gen_step in itertools.count(1):
            for info in self._single_gen_step(gen_step, self.critic_iters):
                yield info


def make_gan(config):
    """
    Initialize a GAN given `config` and return unconsumed part of `config`.
    """
    return _make_gan_from_kwargs(**dict(DEFAULT_PARAMS, **config))


def _make_gan_from_kwargs(
        J0, S0, D0, num_sites, bandwidths, contrasts, sample_sites,
        **rest):
    gen, rest = TuningCurveGenerator.consume_config(
        rest,
        # Stimulator:
        num_tcdom=len(bandwidths) * len(contrasts),
        num_sites=num_sites,
        # Model / SSN:
        J=J0,
        D=D0,
        S=S0,
        # Prober:
        probes=sample_sites_from_stim_space(sample_sites, num_sites),
    )
    disc, rest = consume_subdict(
        UnConditionalDiscriminator, 'disc', rest,
        shape=gen.output_shape,
        loss_type='WD',
    )
    gen_trainer, rest = consume_subdict(
        GeneratorTrainer, 'gen', rest,
        gen, disc,
    )
    disc_trainer, rest = consume_subdict(
        CriticTrainer, 'disc', rest,
        disc,
    )
    return BPTTWassersteinGAN.consume_kwargs(
        gen, disc, gen_trainer, disc_trainer, bandwidths, contrasts,
        **rest)
