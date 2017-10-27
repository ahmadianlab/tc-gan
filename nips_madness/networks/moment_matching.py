from types import SimpleNamespace
import itertools

import numpy as np
import theano

from ..core import BaseComponent
from ..gradient_expressions.utils import sample_sites_from_stim_space
from ..utils import (
    cached_property, cartesian_product, StopWatch,
    theano_function, log_timing,
)
from .bptt_gan import DEFAULT_PARAMS, BaseTrainer
from .ssn import TuningCurveGenerator
from .utils import largerrecursionlimit


def sample_moments(samples):
    """
    Return sample mean and variance.

    Parameters
    ----------
    samples : numpy.ndarray or theano.tensor.matrix
        An array of shape ``(sample_size, channels)``.

    Returns
    -------
    moments : numpy.ndarray or theano.tensor.vector
        An array of shape ``(2, channels)``.

    """
    if isinstance(samples, np.ndarray):
        asarray = np.asarray
    else:
        asarray = theano.tensor.as_tensor_variable
    return asarray([samples.mean(axis=0), samples.var(axis=0)])


class MMGeneratorTrainer(BaseTrainer):

    def __init__(self, gen, dynamics_cost,
                 J_min, J_max, D_min, D_max, S_min, S_max,
                 updater):
        self.target = self.gen = gen
        self.dynamics_cost = dynamics_cost
        self.J_min = J_min
        self.J_max = J_max
        self.D_min = D_min
        self.D_max = D_max
        self.S_min = S_min
        self.S_max = S_max
        self.updater = updater

        self.moment_weights = theano.tensor.matrix('moment_weights')
        self.data_moments = theano.tensor.matrix('data_moments')
        self.inputs = gen.inputs + (self.moment_weights, self.data_moments)

        self.gen_moments = sample_moments(gen.get_output())
        self.gen_moments.name = 'gen_moments'
        diff = (self.data_moments - self.gen_moments)**2

        self.loss = (self.moment_weights * diff).mean()
        self.loss += self.dynamics_cost * self.gen.model.dynamics_penalty
        self.loss.name = 'loss'

    def clip_JDS(self, updates):
        for var in self.gen.get_all_params():
            p_min = getattr(self, var.name + '_min')
            p_max = getattr(self, var.name + '_max')
            updates[var] = updates[var].clip(p_min, p_max)
        return updates

    def get_updates(self):
        return self.clip_JDS(super(MMGeneratorTrainer, self).get_updates())

    @cached_property
    def train(self):
        outputs = [
            self.loss,
            self.gen.model.dynamics_penalty,
            self.gen_moments,
        ]
        with log_timing("compiling {}.train".format(self.__class__.__name__)):
            return theano_function(self.inputs, outputs,
                                   updates=self.get_updates())


class BPTTMomentMatcher(BaseComponent):

    def __init__(self, gen, gen_trainer, bandwidths, contrasts,
                 lam, seed=0):
        self.gen = gen
        self.gen_trainer = gen_trainer
        self.lam = lam

        self.rng = np.random.RandomState(seed)

        self.bandwidths = bandwidths
        self.contrasts = contrasts
        self.stimulator_contrasts, self.stimulator_bandwidths \
            = cartesian_product(contrasts, bandwidths)

    batchsize = property(lambda self: self.gen.model.batchsize)
    num_neurons = property(lambda self: self.gen.num_neurons)

    def get_gen_param(self):
        # To be called from MomentMatchingDriver
        return [
            self.gen.model.J.get_value(),
            self.gen.model.D.get_value(),
            self.gen.model.S.get_value(),
        ]

    def init_dataset(self, data):
        self.data_moments = sample_moments(data)
        r0 = self.data_moments[0]  # sample mean
        self.moment_weights = np.array([1 / r0**2, self.lam / r0**2])

    def prepare(self):
        """ Force compile Theno functions. """
        with largerrecursionlimit(self.gen.model.unroll_scan,
                                  self.gen.model.seqlen):
            self.gen.prepare()
            self.gen_trainer.prepare()

    def train_generator(self, info):
        zg = self.rng.rand(self.batchsize, self.num_neurons, self.num_neurons)
        with self.train_watch:
            (info.loss,
             info.dynamics_penalty,
             info.gen_moments) = self.gen_trainer.train(
                self.stimulator_bandwidths,
                self.stimulator_contrasts,
                zg,
                self.moment_weights,
                self.data_moments,
            )

        info.train_time = self.train_watch.sum()
        return info

    def learning(self):
        for step in itertools.count():
            self.train_watch = StopWatch()
            info = SimpleNamespace(step=step)
            yield self.train_generator(info)


def make_moment_matcher(config):
    return _make_mm_from_kwargs(**dict(DEFAULT_PARAMS, **config))


def _make_mm_from_kwargs(
        J0, S0, D0, num_sites, bandwidths, contrasts, sample_sites,
        include_inhibitory_neurons,
        **rest):
    probes = sample_sites_from_stim_space(sample_sites, num_sites)
    if include_inhibitory_neurons:
        probes.extend(np.array(probes) + num_sites)
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
        probes=probes,
    )
    gen_trainer, rest = MMGeneratorTrainer.consume_config(rest, gen)
    return BPTTMomentMatcher.consume_kwargs(
        gen, gen_trainer, bandwidths, contrasts, **rest)
