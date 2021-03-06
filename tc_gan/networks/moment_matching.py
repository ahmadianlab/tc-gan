import itertools

import numpy as np
import theano

from ..core import BaseComponent
from ..gradient_expressions.utils import sample_sites_from_stim_space
from ..lasagne_toppings.rechack import largerrecursionlimit
from ..utils import (
    cached_property, StopWatch, Namespace,
    theano_function, log_timing, asarray,
)
from .wgan import DEFAULT_PARAMS, BaseTrainer, grid_stimulator_inputs
from .ssn import make_tuning_curve_generator, maybe_mixin_noise, is_heteroin

DEFAULT_PARAMS = dict(
    DEFAULT_PARAMS,
    moment_weight_type='mean',
    **DEFAULT_PARAMS['gen']
)
del DEFAULT_PARAMS['gen']
del DEFAULT_PARAMS['disc']

MOMENT_WEIGHT_TYPES = ('mean', 'ew_mean', 'ew_relative')
r"""
Ways in which moments and variances are weighted.

Assigning one of the following key (`str`) to `.moment_weight_type`
specifies how :math:`w` in equation :eq:`MMGeneratorTrainer-moment-loss`
is calculated.

mean
    Total (unconditional) mean.

    .. math::
       :label: MM-mean-loss

       \mathtt{mean}_{d \in D} \left(
           \left(
               \frac{m_d - \mu_d}{\mathtt{mean}_{d \in D} \mu_d}
           \right)^2 +
           \lambda \left(
               \frac{s_d - \sigma_d}{(\mathtt{mean}_{d \in D} \mu_d)^2}
           \right)^2
       \right)

ew_mean
    Element-wise mean.

    .. math::
       :label: MM-ew_mean-loss

       \mathtt{mean}_{d \in D} \left(
           \left(
               \frac{m_d - \mu_d}{\mu_d + \epsilon}
           \right)^2 +
           \lambda \left(
               \frac{s_d - \sigma_d}{(\mu_d + \epsilon)^2}
           \right)^2
       \right)

ew_relative
    Element-wise relative.

    .. math::
       :label: MM-ew_relative-loss

       \mathtt{mean}_{d \in D} \left(
           \left(
               \frac{m_d - \mu_d}{\mu_d + \epsilon}
           \right)^2 +
           \lambda \left(
               \frac{s_d - \sigma_d}{\sigma_d + \epsilon}
           \right)^2
       \right)

where

* :math:`m, s, \mu, \sigma` are arrays of data sample means/variance
  and generator sample means/variance in a minibatch, as defined in
  :eq:`MMGeneratorTrainer-moment-loss`.

* :math:`\lambda` = `.lam` is the weight of the errors from the
  variances relative to the means.

* :math:`\epsilon` = `.moment_weights_regularization` is a reguralizer
  to avoid division by very small numbers.

"""


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
    return asarray([samples.mean(axis=0), samples.var(axis=0)])


class MMGeneratorTrainer(BaseTrainer):

    r"""
    Generator trainer using moment matching.

    It works by minimizing the following `loss` function:

    .. math::
       :label: MMGeneratorTrainer-loss

       L(x, \mu, \sigma)
       & = L_0((\mathtt{mean}_{b \in B}(x_{d,b}))_{d \in D},
             (\mathtt{var}_{b \in B}(x_{d,b}))_{d \in D},
             \mu, \sigma) \\
       & + L_p(x)

    where :math:`L_0` is the "pure" moment matching loss function (as
    defined below) and :math:`L_p` is the penalty function defined by
    `dynamics_cost` ``*`` `~.EulerSSNModel.dynamics_penalty`.

    Moment matching loss function :math:`L_0` is defined by

    .. math::
       :label: MMGeneratorTrainer-moment-loss

       L_0(m, s, \mu, \sigma) = \mathtt{mean}_{d \in D} \left(
           w_{0,d} (m_d - \mu_d)^2 +
           w_{1,d} (s_d - \sigma_d)^2
       \right)

    where

    * :math:`\mathtt{mean}_{x \in X} f(x)
      = \frac{1}{\# X} \sum_{x \in X} f(x)`.
    * :math:`\mathtt{var}_{x \in X} f(x) = \mathtt{mean}_{x \in X}
      \left(f(x) - \mathtt{mean}_{y \in X} f(y) \right)^2`.
    * :math:`m = (m_d)_{d \in D}
      = (\mathtt{mean}_{b \in B}(x_{d,b}))_{d \in D}`
      = `gen_moments`\ ``[0]``
      is an array of the minibatch means of the generator output in a
      minibatch :math:`B`.
    * :math:`s = (s_d)_{d \in D} = (\mathtt{var}_{b \in B}(x_{d,b}))_{d \in D}`
      = `gen_moments`\ ``[1]``
      is an array of the minibatch variances of the generator output in
      a minibatch :math:`B`.
    * :math:`\mu = (\mu_d)_{d \in D}` = `data_moments`\ ``[0]``
      is an array of the data sample means
    * :math:`\sigma = (\sigma_d)_{d \in D}` = `data_moments`\ ``[1]``
      is an array of the data
      sample variances
    * :math:`x = (x_{d,b})_{d \in D, b \in B}`
      = `gen.get_output() <.TuningCurveGenerator.get_output>`
      is an array of the generator outputs of a minibatch :math:`B`
      across all points :math:`D` in :term:`tuning curve domain`.

      That is to say, :math:`x_{\bullet,b} = (x_{d,b})_{d \in D}` is
      *an* generator output ("image") of a sample :math:`b` in a
      minbatch :math:`B`.
    * :math:`w` = `moment_weights` is a generic weights/coefficients
      for the all moments.  See :eq:`BPTTMomentMatcher-loss` and
      `.BPTTMomentMatcher.set_dataset` for a concrete example.

    Attributes
    ----------
    loss : Theano scalar expression
        The total loss defined by :eq:`MMGeneratorTrainer-loss`.

    gen_moments : Theano matrix expression
        :math:`(m, s)` in :eq:`MMGeneratorTrainer-moment-loss`.

    data_moments : `theano.tensor.matrix`
        :math:`(\mu, \sigma)` in :eq:`MMGeneratorTrainer-moment-loss`.

    moment_weights : `theano.tensor.matrix`
        :math:`w` in :eq:`MMGeneratorTrainer-moment-loss`.

    dynamics_cost : float
        The coefficient for `~.EulerSSNModel.dynamics_penalty`.

    gen : `.TuningCurveGenerator`
        Tuning curve generator.

    """

    def __init__(self, gen, dynamics_cost, rate_cost,
                 J_min, J_max, D_min, D_max, S_min, S_max,
                 updater):
        self.target = self.gen = gen
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
        self.moment_weights = theano.tensor.matrix('moment_weights')
        self.data_moments = theano.tensor.matrix('data_moments')
        self.inputs = gen.inputs + (self.moment_weights, self.data_moments)
        self.input_names = gen._input_names \
            + ['moment_weights', 'data_moments']

        self.gen_moments = sample_moments(gen.get_output())
        self.gen_moments.name = 'gen_moments'
        diff = (self.data_moments - self.gen_moments)**2

        self.loss = (self.moment_weights * diff).mean()
        self.loss += self.dynamics_cost * self.gen.model.dynamics_penalty
        self.loss += self.rate_cost * gen.model.rate_penalty
        self.loss.name = 'loss'

    def clip_params(self, updates):
        for var in self.gen.get_all_params():
            p_min = getattr(self, var.name + '_min')
            p_max = getattr(self, var.name + '_max')
            p_min = np.array(p_min, dtype=updates[var].dtype)
            p_max = np.array(p_max, dtype=updates[var].dtype)
            updates[var] = updates[var].clip(p_min, p_max)
        return updates

    def get_updates(self):
        return self.clip_params(super(MMGeneratorTrainer, self).get_updates())

    @cached_property
    def train_impl(self):
        outputs = [
            self.loss,
            self.gen.model.dynamics_penalty,
            self.gen.model.rate_penalty,
            self.gen_moments,
        ]
        with log_timing("compiling {}.train".format(self.__class__.__name__)):
            return theano_function(self.inputs, outputs,
                                   updates=self.get_updates())

    def train(self, rng=None, **kwargs):
        kwargs = maybe_mixin_noise(self.gen, rng, kwargs)
        values = [kwargs.pop(k) for k in self.input_names]
        assert not kwargs
        return self.train_impl(*values)


class HeteroInMMGeneratorTrainer(MMGeneratorTrainer):

    def __init__(self, gen, dynamics_cost, rate_cost,
                 J_min, J_max, D_min, D_max, S_min, S_max,
                 updater,
                 V_min=0, V_max=1):
        self.target = self.gen = gen
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


class BPTTMomentMatcher(BaseComponent):

    r"""
    BPTT-based moment-matching of a tuning curve generator.

    It works by minimizing the loss function :math:`L = L_0 + L_p`
    (see :eq:`MMGeneratorTrainer-loss`) where :math:`L_p` is the
    penalization of the dynamics and :math:`L_0` is one of the "pure"
    moment matching loss function noted in `MOMENT_WEIGHT_TYPES`,
    depending on the value of `.moment_weight_type`.

    Attributes
    ----------
    gen : `.TuningCurveGenerator`
        Tuning curve generator.

    gen_trainer : `.MMGeneratorTrainer`
        Generator trainer.

    lam : float
        :math:`\lambda` in Equations :eq:`MM-mean-loss`,
        :eq:`MM-ew_mean-loss` and :eq:`MM-ew_relative-loss`.

    moment_weights_regularization : float
        :math:`\epsilon` in Equations :eq:`MM-ew_mean-loss` and
        :eq:`MM-ew_relative-loss`.

    moment_weight_type : {'total_mean', 'relative_mean', 'relative'}
        How `moment_weights` are calculated. See: `MOMENT_WEIGHT_TYPES`.

    moment_weights : numpy.ndarray
        Numerical array to be "substituted" into
        `.MMGeneratorTrainer.moment_weights`.  It is calculated from
        `data_moments`, `moment_weights_regularization`, and `lam` to
        equate :eq:`BPTTMomentMatcher-loss` and
        :eq:`MMGeneratorTrainer-moment-loss`.

    data_moments : numpy.ndarray
        Numerical array to be "substituted" into
        `.MMGeneratorTrainer.data_moments`.

    bandwidths
    contrasts : numpy.ndarray
        Values of bandwidths and contrasts whose every combinations
        are fed to `.BandwidthContrastStimulator`.

    stimulator_bandwidths
    stimulator_contrasts : numpy.ndarray
        A matrix of shape ``(batchsize, len(bandwidths) * len(contrasts))``.
        These are the actual numerical arrays to be substituted to
        `.BandwidthContrastStimulator.bandwidths` and
        `.BandwidthContrastStimulator.contrasts`.  They are
        constructed from `bandwidths` and `contrasts` using
        `.grid_stimulator_inputs`.

    """

    def __init__(self, gen, gen_trainer, bandwidths, contrasts,
                 lam, moment_weights_regularization,
                 include_inhibitory_neurons,
                 rate_penalty_threshold,
                 moment_weight_type,
                 seed=0):
        self.gen = gen
        self.gen_trainer = gen_trainer
        self.lam = lam
        self.moment_weights_regularization = moment_weights_regularization
        self.moment_weight_type = moment_weight_type

        self.rng = np.random.RandomState(seed)

        self.bandwidths = bandwidths
        self.contrasts = contrasts
        self.stimulator_contrasts, self.stimulator_bandwidths \
            = grid_stimulator_inputs(contrasts, bandwidths, self.batchsize)

        self.include_inhibitory_neurons = include_inhibitory_neurons
        # See: BPTTWassersteinGAN

        self.rate_penalty_threshold = rate_penalty_threshold

    batchsize = property(lambda self: self.gen.batchsize)
    num_neurons = property(lambda self: self.gen.num_neurons)

    @property
    def num_mom_conds(self):
        """
        Number of conditions in which moments are evaluated.
        """
        return self.gen.num_tcdom * len(self.gen.probes)

    @property
    def sample_sites(self):
        probes = list(self.gen.prober.probes)
        if self.include_inhibitory_neurons:
            return probes[:len(probes) // 2]
        else:
            return probes

    def get_gen_param(self):
        # To be called from MomentMatchingDriver
        return [
            self.gen.model.J.get_value(),
            self.gen.model.D.get_value(),
            self.gen.model.S.get_value(),
        ]

    def set_dataset(self, data):
        """ Calculate `data_moments` (means & variances) from `data`. """
        self.data_moments = sample_moments(data)

        # Calculate self.moment_weights:
        eps = self.moment_weights_regularization
        num = np.broadcast_to([[1], [self.lam]], self.data_moments.shape)
        assert self.moment_weight_type in MOMENT_WEIGHT_TYPES
        if self.moment_weight_type == 'mean':
            den = data.mean()
            self.moment_weights = num / np.array([[den**2], [den**4]])
        elif self.moment_weight_type == 'ew_mean':
            r0 = self.data_moments[0]  # sample mean
            den = r0 + eps
            self.moment_weights = num / np.array([den**2, den**4])
        elif self.moment_weight_type == 'ew_relative':
            den = (self.data_moments + eps) ** 2
            self.moment_weights = num / den
        else:
            raise ValueError('Unknown moment_weight_type = {}'
                             .format(self.moment_weight_type))

    def prepare(self):
        """ Force compile Theano functions. """
        with largerrecursionlimit(self.gen.model.unroll_scan,
                                  self.gen.model.seqlen):
            self.gen.prepare()
            self.gen_trainer.prepare()

    def train_generator(self, info):
        with self.train_watch:
            (info.loss,
             info.dynamics_penalty,
             info.rate_penalty,
             info.gen_moments) = self.gen_trainer.train(
                rng=self.rng,
                stimulator_bandwidths=self.stimulator_bandwidths,
                stimulator_contrasts=self.stimulator_contrasts,
                model_rate_penalty_threshold=self.rate_penalty_threshold,
                moment_weights=self.moment_weights,
                data_moments=self.data_moments,
             )

        info.train_time = self.train_watch.sum()
        return info

    def learning(self):
        for step in itertools.count():
            self.train_watch = StopWatch()
            info = Namespace(step=step)
            yield self.train_generator(info)


def make_moment_matcher(config):
    """make_moment_matcher(config: dict) -> (MM, dict)
    Initialize an MM given `config` and return unconsumed part of `config`.
    """
    return _make_mm_from_kwargs(**dict(DEFAULT_PARAMS, **config))


def _make_mm_from_kwargs(
        J0, S0, D0, num_sites, bandwidths, contrasts, sample_sites,
        include_inhibitory_neurons,
        **rest):
    probes = sample_sites_from_stim_space(sample_sites, num_sites)
    if include_inhibitory_neurons:
        probes.extend(np.array(probes) + num_sites)
    gen, rest = make_tuning_curve_generator(
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
    if is_heteroin(gen):
        gen_trainer, rest \
            = HeteroInMMGeneratorTrainer.consume_kwargs(gen, **rest)
    else:
        gen_trainer, rest = MMGeneratorTrainer.consume_kwargs(gen, **rest)
    return BPTTMomentMatcher.consume_kwargs(
        gen, gen_trainer, bandwidths, contrasts,
        include_inhibitory_neurons=include_inhibitory_neurons,
        **rest)
