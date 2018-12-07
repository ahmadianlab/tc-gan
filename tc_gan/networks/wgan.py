"""
Generative Adversarial Network for fitting tuning curve generator.

This module implements Wasserstein GAN (`BPTTWassersteinGAN`) to fit
`TuningCurveGenerator`.  It is composed of the following components:

- `BPTTWassersteinGAN`

  - `.TuningCurveGenerator`: generator
  - `UnConditionalDiscriminator`: discriminator
  - `GeneratorTrainer`: generator-trainer
  - `CriticTrainer`: discriminator (critic)-trainer

Generator and Discriminator only define the symbolic expression of the
forward-pass and gradients (backward-pass) are calculated in their
trainer.

`BPTTWassersteinGAN` is responsible for bundling them together,
managing numeric data, and looping through the learning argorithm.

"""

import itertools

import lasagne
import numpy as np

from .. import ssnode
from ..core import BaseComponent, consume_subdict
from ..gradient_expressions.utils import sample_sites_from_stim_space
from ..lasagne_toppings.rechack import largerrecursionlimit
from ..utils import (
    cached_property, cartesian_product, random_minibatches, StopWatch,
    theano_function, log_timing, Namespace,
)
from .ssn import make_tuning_curve_generator, maybe_mixin_noise, is_heteroin


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
    io_type='asym_tanh',
    tau_E=10,
    tau_I=1,
    dt=0.1,
    seqlen=1200,
    batchsize=1,
    skip_steps=1000,
    gen=dict(
        rate_cost=0.01,
        rate_penalty_threshold=200.0,
    ),
    disc=dict(
        rate_penalty_bound=-1.0,
    ),
)


class UnConditionalDiscriminator(BaseComponent):

    def __init__(self, shape, loss_type,
                 layers, normalization, nonlinearity,
                 net_options=None):
        from .simple_discriminator import make_net
        self.l_out = make_net(
            shape, loss_type,
            layers, normalization, nonlinearity,
            options=net_options,
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

    def prepare(self):
        """ Force compile Theano functions. """
        self.accuracy


def apply_l2_decay(updates, params, decay):
    for var in params:
        updates[var] -= decay * var


def apply_l1_decay(updates, params, decay):
    from theano.tensor import sgn
    for var in params:
        updates[var] -= decay * sgn(var)


class Updater(BaseComponent):

    _named_update_configs = {
        # Values recommended by Gulrajani et al. (2017)
        # -- http://arxiv.org/abs/1704.00028:
        'adam-wgan': ('adam', dict(beta1=0.5, beta2=0.9)),
    }

    def __init__(self, learning_rate=0.001, update_name='adam-wgan',
                 update_config={},
                 reg_l2_penalty=0.0, reg_l2_decay=0.0,
                 reg_l1_penalty=0.0, reg_l1_decay=0.0):
        self.learning_rate = learning_rate
        self.update_name = update_name
        self.update_config = update_config

        self.reg_l2_penalty = reg_l2_penalty
        self.reg_l1_penalty = reg_l1_penalty
        self.reg_l2_decay = reg_l2_decay  # Loshchilov & Hutter (2017)
        self.reg_l1_decay = reg_l1_decay

    def _get_updater(self):
        name, default = self._named_update_configs.get(
            self.update_name,
            (self.update_name, {}))
        updater = getattr(lasagne.updates, name)
        config = dict(default, **self.update_config)
        return updater, config

    def __call__(self, loss, params):
        updater, config = self._get_updater()

        # Add L2/L1 regularizatio penalty to `loss`:
        if self.reg_l2_penalty:
            loss += lasagne.regularization.apply_penalty(
                params, lasagne.regularization.l2,
            ) * self.reg_l2_penalty
        if self.reg_l1_penalty:
            loss += lasagne.regularization.apply_penalty(
                params, lasagne.regularization.l1,
            ) * self.reg_l1_penalty

        # Compute gradient & update formulae:
        updates = updater(
            loss, params, learning_rate=self.learning_rate,
            **config)

        # Add L2/L1 weight decay to the update:
        learning_rate = self.learning_rate
        if self.reg_l2_decay:
            apply_l2_decay(updates, params, learning_rate * self.reg_l2_decay)
        if self.reg_l1_decay:
            apply_l1_decay(updates, params, learning_rate * self.reg_l1_decay)

        return updates


class BaseTrainer(BaseComponent):

    @classmethod
    def consume_kwargs(self, *args, **kwargs):
        updater, rest = Updater.consume_kwargs(**kwargs)
        return super(BaseTrainer, self).consume_kwargs(
            *args, updater=updater, **rest)

    def get_updates(self):
        with log_timing("{}.update()".format(self.__class__.__name__)):
            return self.updater(self.loss, self.target.get_all_params())

    @cached_property
    def train_impl(self):
        with log_timing("compiling {}.train_impl"
                        .format(self.__class__.__name__)):
            return theano_function(self.inputs, self.loss,
                                   updates=self.get_updates())

    train = property(lambda self: self.train_impl)

    def prepare(self):
        """ Force compile Theano functions. """
        self.train_impl


class CriticTrainer(BaseTrainer):
    """
    Discriminator/Critic trainer for WGAN.
    """

    def __init__(self, disc, updater):
        self.target = disc
        self.updater = updater

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

    def __init__(self, gen, disc, dynamics_cost, rate_cost,
                 J_min, J_max, D_min, D_max, S_min, S_max,
                 updater):
        self.target = self.gen = gen
        self.disc = disc
        self.dynamics_cost = dynamics_cost
        self.rate_cost = rate_cost
        self.J_min = J_min
        self.J_max = J_max
        self.D_min = D_min
        self.D_max = D_max
        self.S_min = S_min
        self.S_max = S_max
        self.updater = updater
        self.post_init()

    def post_init(self):
        gen = self.gen
        disc = self.disc
        self.loss = - disc.get_output(gen.get_output()).mean()
        self.loss += self.dynamics_cost * gen.model.dynamics_penalty
        self.loss += self.rate_cost * gen.model.rate_penalty
        self.inputs = gen.inputs

    def clip_params(self, updates):
        for var in self.gen.get_all_params():
            p_min = getattr(self, var.name + '_min')
            p_max = getattr(self, var.name + '_max')
            p_min = np.array(p_min, dtype=updates[var].dtype)
            p_max = np.array(p_max, dtype=updates[var].dtype)
            updates[var] = updates[var].clip(p_min, p_max)
        return updates

    def get_updates(self):
        return self.clip_params(super(GeneratorTrainer, self).get_updates())

    def train(self, rng=None, **kwargs):
        kwargs = maybe_mixin_noise(self.gen, rng, kwargs)
        values = [kwargs.pop(k) for k in self.gen._input_names]
        assert not kwargs
        return self.train_impl(*values)


class HeteroInGeneratorTrainer(GeneratorTrainer):

    def __init__(self, gen, disc, dynamics_cost, rate_cost,
                 J_min, J_max, D_min, D_max, S_min, S_max,
                 updater,
                 V_min=0, V_max=1):
        self.target = self.gen = gen
        self.disc = disc
        self.dynamics_cost = dynamics_cost
        self.rate_cost = rate_cost
        self.J_min = J_min
        self.J_max = J_max
        self.D_min = D_min
        self.D_max = D_max
        self.S_min = S_min
        self.S_max = S_max
        self.V_min = V_min
        self.V_max = V_max
        self.updater = updater
        self.post_init()


def emit_generator_trainer(gen, *args, **kwargs):
    if is_heteroin(gen):
        trainer_class = HeteroInGeneratorTrainer
    else:
        trainer_class = GeneratorTrainer
    return trainer_class.consume_kwargs(gen, *args, **kwargs)


def grid_stimulator_inputs(contrasts, bandwidths, batchsize):
    product = cartesian_product(contrasts, bandwidths)
    return np.tile(product.reshape((1,) + product.shape),
                   (batchsize,) + (1,) * product.ndim).swapaxes(0, 1)


class BPTTWassersteinGAN(BaseComponent):

    def __init__(self, gen, disc, gen_trainer, disc_trainer,
                 bandwidths, contrasts,
                 include_inhibitory_neurons,
                 rate_penalty_threshold,
                 critic_iters_init, critic_iters, lipschitz_cost,
                 disc_rate_penalty_bound,
                 seed=0):
        self.gen = gen
        self.disc = disc
        self.gen_trainer = gen_trainer
        self.disc_trainer = disc_trainer

        self.critic_iters_init = critic_iters_init
        self.critic_iters = critic_iters
        self.lipschitz_cost = lipschitz_cost
        self.disc_rate_penalty_bound = disc_rate_penalty_bound
        self.rng = np.random.RandomState(seed)

        self.bandwidths = bandwidths
        self.contrasts = contrasts
        self.stimulator_contrasts, self.stimulator_bandwidths \
            = grid_stimulator_inputs(contrasts, bandwidths, self.batchsize)

        self.include_inhibitory_neurons = include_inhibitory_neurons
        # As of writing, this variable is not required for learning.
        # This is only used from `sample_sites` property which is only
        # used from bptt_wgan.learn.

        self.rate_penalty_threshold = rate_penalty_threshold

    batchsize = property(lambda self: self.gen.batchsize)
    num_neurons = property(lambda self: self.gen.num_neurons)

    @property
    def sample_sites(self):
        probes = list(self.gen.prober.probes)
        if self.include_inhibitory_neurons:
            return probes[:len(probes) // 2]
        else:
            return probes

    # To be compatible with GANDriver:
    loss_type = 'WD'
    discriminator = property(lambda self: self.disc.l_out)
    NZ = property(lambda self: self.batchsize)

    def get_gen_param(self):
        # To be called from GANDriver
        return [
            self.gen.model.J.get_value(),
            self.gen.model.D.get_value(),
            self.gen.model.S.get_value(),
        ]

    def set_dataset(self, data, **kwargs):
        kwargs.setdefault('seed', self.rng)
        self.dataset = random_minibatches(self.batchsize, data, **kwargs)

    def next_minibatch(self):
        return next(self.dataset)

    def prepare(self):
        """ Force compile Theano functions. """
        with largerrecursionlimit(self.gen.model.unroll_scan,
                                  self.gen.model.seqlen):
            self.gen.prepare()
            self.gen_trainer.prepare()
        self.disc.prepare()
        self.disc_trainer.prepare()

    def gen_forward(self):
        return self.gen.forward(
            rng=self.rng,
            stimulator_bandwidths=self.stimulator_bandwidths,
            stimulator_contrasts=self.stimulator_contrasts,
            model_rate_penalty_threshold=self.rate_penalty_threshold,
        )

    def train_discriminator(self, info):
        xd = self.next_minibatch()
        eps = self.rng.rand(self.batchsize).reshape((-1, 1))
        with self.gen_forward_watch:
            gen_out = self.gen_forward()
        xg = gen_out.prober_tuning_curve
        xp = eps * xd + (1 - eps) * xg

        info.gen_out = gen_out
        info.dynamics_penalty = gen_out.model_dynamics_penalty
        info.rate_penalty = gen_out.model_rate_penalty
        info.xd = xd
        info.xg = xg
        info.xp = xp
        info.gen_time = self.gen_forward_watch.times[-1]

        bound = self.disc_rate_penalty_bound
        if bound > 0 and gen_out.model_rate_penalty > bound:
            info.disc_loss = np.nan
            info.accuracy = np.nan
            info.disc_time = np.nan
            return info

        with self.disc_train_watch:
            info.disc_loss = self.disc_trainer.train(
                xg, xd, xp, self.lipschitz_cost,
            )

        info.accuracy = self.disc.accuracy(xg, xd)
        info.disc_time = self.disc_train_watch.times[-1]
        return info

    def train_generator(self, info):
        with self.gen_train_watch:
            info.gen_loss = self.gen_trainer.train(
                rng=self.rng,
                stimulator_bandwidths=self.stimulator_bandwidths,
                stimulator_contrasts=self.stimulator_contrasts,
                model_rate_penalty_threshold=self.rate_penalty_threshold,
            )

        info.gen_forward_time = self.gen_forward_watch.sum()
        info.gen_train_time = self.gen_train_watch.sum()
        info.gen_time = info.gen_train_time + info.gen_forward_time
        info.disc_time = self.disc_train_watch.sum()
        return info

    def _single_gen_step(self, gen_step, critic_iters):
        self.gen_forward_watch = StopWatch()
        self.gen_train_watch = StopWatch()
        self.disc_train_watch = StopWatch()

        for disc_step in range(critic_iters):
            info = Namespace(is_discriminator=True, gen_step=gen_step,
                             disc_step=disc_step)
            yield self.train_discriminator(info)

        info = Namespace(is_discriminator=False, gen_step=gen_step)
        yield self.train_generator(info)

    def learning(self):
        for info in self._single_gen_step(0, self.critic_iters_init):
            yield info
        for gen_step in itertools.count(1):
            for info in self._single_gen_step(gen_step, self.critic_iters):
                yield info


def probes_from_stim_space(stim_locs, num_sites, include_inhibitory_neurons):
    probes = sample_sites_from_stim_space(stim_locs, num_sites)
    if include_inhibitory_neurons:
        probes.extend(np.array(probes) + num_sites)
    return probes


def make_gan(config):
    """make_gan(config: dict) -> (GAN, dict)
    Initialize a GAN given `config` and return unconsumed part of `config`.
    """
    kwargs = dict(DEFAULT_PARAMS, **config)
    kwargs['gen'] = dict(DEFAULT_PARAMS['gen'], **config.get('gen', {}))
    kwargs['disc'] = dict(DEFAULT_PARAMS['disc'], **config.get('disc', {}))
    return _make_gan_from_kwargs(**kwargs)


def _make_gan_from_kwargs(
        J0, S0, D0, num_sites, bandwidths, contrasts, sample_sites,
        include_inhibitory_neurons,
        consume_union=True,
        **rest):
    probes = probes_from_stim_space(sample_sites, num_sites,
                                    include_inhibitory_neurons)
    if 'V0' in rest:
        rest['V'] = rest.pop('V0')
    rate_penalty_threshold = rest['gen'].pop('rate_penalty_threshold')  # FIXME
    disc_rate_penalty_bound = rest['disc'].pop('rate_penalty_bound')  # FIXME
    gen, rest = make_tuning_curve_generator(
        rest,
        consume_union=consume_union,
        # Stimulator:
        num_tcdom=len(bandwidths) * len(contrasts),
        num_sites=num_sites,
        # Model / SSN:
        J=J0,
        D=D0,
        S=S0,
        # Prober:
        probes=probes,
    )
    disc, rest = consume_subdict(
        UnConditionalDiscriminator.consume_kwargs, 'disc', rest,
        shape=gen.output_shape,
        loss_type='WD',
    )
    gen_trainer, rest = consume_subdict(
        emit_generator_trainer, 'gen', rest,
        gen, disc,
    )
    disc_trainer, rest = consume_subdict(
        CriticTrainer.consume_kwargs, 'disc', rest,
        disc,
    )
    if consume_union:
        for key in ['V_min', 'V_max']:
            rest.pop(key, None)
    return BPTTWassersteinGAN.consume_kwargs(
        gen, disc, gen_trainer, disc_trainer, bandwidths, contrasts,
        include_inhibitory_neurons=include_inhibitory_neurons,
        rate_penalty_threshold=rate_penalty_threshold,
        disc_rate_penalty_bound=disc_rate_penalty_bound,
        **rest)
