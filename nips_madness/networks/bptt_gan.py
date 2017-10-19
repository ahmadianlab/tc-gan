from types import SimpleNamespace
import collections
import itertools

import lasagne
import numpy as np
import theano

from .. import ssnode
from ..gradient_expressions.make_w_batch import make_W_with_x
from ..gradient_expressions.utils import sample_sites_from_stim_space
from ..utils import (
    cached_property, cartesian_product, random_minibatches, StopWatch,
)
from .core import BaseComponent


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
    num_steps=1200,
    batchsize=1,
    skip_steps=1000,
)


def theano_function(*args, **kwds):
    # from theano.compile.nanguardmode import NanGuardMode
    # kwds.setdefault('mode', NanGuardMode(
    #     nan_is_error=True, inf_is_error=True, big_is_error=True))
    kwds.setdefault('allow_input_downcast', True)
    # MAYBE: make sure to use theano.config.floatX everywhere and
    # remove allow_input_downcast.
    return theano.function(*args, **kwds)


def int_or_lscalr(value, name):
    if value is None:
        raise NotImplementedError
        # The intention was to support variable sizes (e.g.,
        # batchsize, num_sites, etc.) but it is not supported at the
        # moment.
        return theano.tensor.lscalar(name)
    else:
        return value


class BandwidthContrastStimulator(BaseComponent):
    """
    Stimulator for varying bandwidths and contrasts.

    .. attribute:: num_tcdom

       Number of points in tuning curve (TC) domain.

    """

    outputs = ()

    def __init__(self, num_sites, num_tcdom, smoothness):
        self.num_sites = int_or_lscalr(num_sites, 'num_sites')
        self.num_tcdom = int_or_lscalr(num_tcdom, 'num_tcdom')

        self.bandwidths = theano.tensor.vector('bandwidths')
        self.contrasts = theano.tensor.vector('contrasts')
        self.smoothness = smoothness
        self.site_to_band = np.linspace(-0.5, 0.5, self.num_sites)

        def sigm(x):
            from theano.tensor import exp
            return 1 / (1 + exp(-x / smoothness))

        x = self.site_to_band.reshape((1, -1))
        b = self.bandwidths.reshape((-1, 1))
        c = self.contrasts.reshape((-1, 1))
        stim = c * sigm(x + b/2) * sigm(b/2 - x)

        self.stimulus = theano.tensor.tile(stim, (1, 2))
        """
        A symbolic array of shape (`.num_tcdom`, 2 * `.num_sites`).
        """

        self.inputs = (self.bandwidths, self.contrasts)

    num_neurons = property(lambda self: self.num_sites * 2)


class SSN(BaseComponent):

    def __init__(self, stimulator, J, D, S, k, n, tau_E, tau_I, dt,
                 io_type='asym_power'):

        J = np.asarray(J)
        D = np.asarray(D)
        S = np.asarray(S)
        assert J.shape == D.shape == S.shape == (2, 2)

        self.stimulator = stimulator
        self.J = theano.shared(J, name='J')
        self.D = theano.shared(D, name='D')
        self.S = theano.shared(S, name='S')
        self.k = k
        self.n = n
        self.tau_E = tau_E
        self.tau_I = tau_I
        self.dt = dt
        self.io_type = io_type

        # num_tcdom = "batch dimension" when using Lasagne:
        num_sites = stimulator.num_sites
        num_neurons = stimulator.num_neurons

        # TODO: implement tau in theano
        tau = np.zeros(num_neurons, dtype=theano.config.floatX)
        tau[:num_sites] = tau_E
        tau[num_sites:] = tau_I
        self.tau = tau
        self.eps = dt / tau

        self.zmat = theano.tensor.matrix('zmat')
        self.Wt = make_W_with_x(theano.tensor.shape_padleft(self.zmat),
                                self.J, self.D, self.S,
                                num_sites,
                                np.linspace(-0.5, 0.5, num_sites))[0].T
        self.Wt.name = 'Wt'

        self.f = ssnode.make_io_fun(self.k, self.n, io_type=self.io_type)

    # MAYBE: move all the stuff in __init__ to get_output_for

    def get_output_for(self, r):
        f = self.f
        Wt = self.Wt
        I = self.stimulator.stimulus
        eps = self.eps.reshape((1, -1))
        co_eps = (1 - self.eps).reshape((1, -1))
        return co_eps * r + eps * f(r.dot(Wt) + I)


class EulerSSNLayer(lasagne.layers.Layer):

    # See:
    # http://lasagne.readthedocs.io/en/latest/user/custom_layers.html
    # http://lasagne.readthedocs.io/en/latest/modules/layers/base.html

    def __init__(self, incoming, **kwargs):
        # It's a bit scary to share namespace with lasagne; so let's
        # introduce only one namespace and keep SSN-specific stuff
        # there:
        self.ssn, rest = SSN.consume_kwargs(**kwargs)

        super(EulerSSNLayer, self).__init__(incoming, **rest)

        self.add_param(self.ssn.J, (2, 2), name='J')
        self.add_param(self.ssn.D, (2, 2), name='D')
        self.add_param(self.ssn.S, (2, 2), name='S')

    def get_output_for(self, input, **kwargs):
        return self.ssn.get_output_for(input, **kwargs)


class EulerSSNModel(BaseComponent):

    def __init__(self, stimulator, J, D, S, k, n, tau_E, tau_I, dt,
                 skip_steps=None, num_steps=None, batchsize=None):
        self.stimulator = stimulator
        self.skip_steps = int_or_lscalr(skip_steps, 'sample_beg')
        self.batchsize = int_or_lscalr(batchsize, 'batchsize')
        self.num_steps = int_or_lscalr(num_steps, 'num_steps')

        # num_tcdom = "batch dimension" when using Lasagne:
        num_neurons = self.stimulator.num_neurons
        shape = (self.stimulator.num_tcdom, self.num_steps, num_neurons)

        shape_rec = (shape[0],) + shape[2:]
        self.l_ssn = EulerSSNLayer(
            lasagne.layers.InputLayer(shape_rec),
            stimulator=stimulator,
            J=J, D=D, S=S, k=k, n=n, tau_E=tau_E, tau_I=tau_I, dt=dt,
        )

        # Since SSN is autonomous, we don't need to do anything for
        # input and input_to_hidden layers.  So let's zero them out:
        self.l_fake_input = lasagne.layers.InputLayer(
            shape,
            theano.tensor.zeros(shape),
        )
        self.l_zero = lasagne.layers.NonlinearityLayer(
            lasagne.layers.InputLayer(shape_rec),
            nonlinearity=lambda x: x * 0,
            # ...or maybe just nonlinearity=None; check if Theano
            # short-circuits if multiplied by zero.
        )

        self.l_rec = lasagne.layers.CustomRecurrentLayer(
            self.l_fake_input,
            input_to_hidden=self.l_zero,
            hidden_to_hidden=self.l_ssn,
            nonlinearity=None,  # let EulerSSNLayer handle the nonlinearity
            precompute_input=False,  # True (default) is maybe better?
        )

        self.trajectories = rates = lasagne.layers.get_output(self.l_rec)
        rs = rates[:, self.skip_steps:]
        time_avg = rs.mean(axis=1)  # shape: (num_tcdom, num_neurons)
        dynamics_penalty = ((rs[:, 1:] - rs[:, :-1]) ** 2).mean()

        self.zs = theano.tensor.tensor3('zs')
        self.time_avg, _ = theano.map(
            lambda z: theano.clone(time_avg, {self.zmat: z}),
            [self.zs],
        )
        penalties, _ = theano.map(
            lambda z: theano.clone(dynamics_penalty, {self.zmat: z}),
            [self.zs],
        )
        self.dynamics_penalty = penalties.mean()
        # TODO: make sure theano.clone is not bottleneck here.
        self.dynamics_penalty.name = 'dynamics_penalty'
        # TODO: find if assigning a name to an expression is a valid usecase

        self.inputs = (self.zs,)
        self.outputs = (self.dynamics_penalty,)

    zmat = property(lambda self: self.l_ssn.ssn.zmat)
    dt = property(lambda self: self.l_ssn.ssn.dt)
    io_type = property(lambda self: self.l_ssn.ssn.io_type)
    J = property(lambda self: self.l_ssn.ssn.J)
    D = property(lambda self: self.l_ssn.ssn.D)
    S = property(lambda self: self.l_ssn.ssn.S)

    @cached_property
    def compute_trajectories(self):
        return theano_function(
            (self.zmat,) + self.stimulator.inputs,
            self.trajectories,
        )


class FixedProber(BaseComponent):

    inputs = ()

    def __init__(self, model, probes):
        self.model = model
        self.probes = probes

        tc = self.model.time_avg[:, :, self.probes]
        self.tuning_curve = tc.reshape((self.model.batchsize, -1))
        self.tuning_curve.name = 'tuning_curve'

        self.outputs = (self.tuning_curve,)


def collect_names(prefixes, var_lists):
    names = []
    for prefix, var_list in zip(prefixes, var_lists):
        for var in var_list:
            names.append(prefix + var.name)
    return names


class TuningCurveGenerator(BaseComponent):

    @classmethod
    def consume_kwargs(cls, **kwargs):
        stimulator, rest = BandwidthContrastStimulator.consume_kwargs(**kwargs)
        model, rest = EulerSSNModel.consume_kwargs(stimulator, **rest)
        prober, rest = FixedProber.consume_kwargs(model, **rest)
        return cls(stimulator, model, prober), rest

    def __init__(self, stimulator, model, prober):
        self.stimulator = stimulator
        self.model = model
        self.prober = prober

        out_names = collect_names(['stimulator_', 'model_', 'prober_'],
                                  [self.stimulator.outputs,
                                   self.model.outputs,
                                   self.prober.outputs])
        self.OutType = collections.namedtuple('OutType', out_names)

        self._input_names = collect_names(
            ['stimulator_', 'model_', 'prober_'],
            [self.stimulator.inputs,
             self.model.inputs,
             self.prober.inputs])

    inputs = property(lambda self: (self.stimulator.inputs +
                                    self.model.inputs +
                                    self.prober.inputs))
    num_tcdom = property(lambda self: self.stimulator.num_tcdom)
    num_sites = property(lambda self: self.stimulator.num_sites)
    num_neurons = property(lambda self: self.stimulator.num_neurons)
    batchsize = property(lambda self: self.model.batchsize)
    dt = property(lambda self: self.model.dt)
    probes = property(lambda self: self.prober.probes)

    @property
    def inputs(self):
        return (self.stimulator.inputs +
                self.model.inputs +
                self.prober.inputs)

    @property
    def outputs(self):
        return (self.stimulator.outputs +
                self.model.outputs +
                self.prober.outputs)

    @cached_property
    def _forward(self):
        return theano_function(self.inputs, self.outputs)

    def forward(self, **kwargs):
        values = [kwargs.pop(k) for k in self._input_names]
        assert not kwargs
        return self.OutType(*self._forward(*values))

    def get_all_params(self):
        return [self.model.J, self.model.D, self.model.S]

    def get_output(self, **kwargs):
        if not kwargs:
            return self.prober.outputs[0]
        kv = zip(self._input_names,
                 (self.stimulator.inputs +
                  self.model.inputs +
                  self.prober.inputs))
        replace = {v: kwargs.pop(k) for k, v in kv if k in kwargs}
        assert not kwargs
        return theano.clone(self.prober.outputs[0], replace)


class UnConditionalDiscriminator(BaseComponent):

    def __init__(self, batchsize, num_tcdom, loss_type,
                 layers, normalization, nonlinearity):
        from .simple_discriminator import make_net
        self.l_out = make_net(
            (batchsize, num_tcdom), loss_type,
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

    @classmethod
    def consume_kwargs(cls, J0, S0, D0,
                       disc_layers, disc_normalization, disc_nonlinearity,
                       WGAN_lambda,
                       **kwargs):
        rest = dict(DEFAULT_PARAMS, **kwargs)
        num_sites = rest.pop('num_sites')
        bandwidths = rest.pop('bandwidths')
        contrasts = rest.pop('contrasts')
        sample_sites = rest.pop('sample_sites')
        gen, rest = TuningCurveGenerator.consume_kwargs(
            # Stimulator:
            num_tcdom=len(bandwidths) * len(contrasts),
            num_sites=num_sites,
            # Model / SSN:
            J=J0,
            D=D0,
            S=S0,
            # Prober:
            probes=sample_sites_from_stim_space(sample_sites, num_sites),
            **rest)
        disc, rest = UnConditionalDiscriminator.consume_kwargs(
            batchsize=gen.batchsize,
            num_tcdom=gen.num_tcdom,
            layers=disc_layers,
            normalization=disc_normalization,
            nonlinearity=disc_nonlinearity,
            loss_type='WD', **rest)
        gen_trainer, rest = GeneratorTrainer.consume_kwargs(
            gen, disc,
            **rest)
        disc_trainer, rest = CriticTrainer.consume_kwargs(
            disc,
            **rest)
        return super(BPTTWassersteinGAN, cls).consume_kwargs(
            gen, disc, gen_trainer, disc_trainer, bandwidths, contrasts,
            lmd=WGAN_lambda,
            **rest)

    def __init__(self, gen, disc, gen_trainer, disc_trainer,
                 bandwidths, contrasts,
                 n_critic_init, n_critic, lmd,
                 seed=0):
        self.gen = gen
        self.disc = disc
        self.gen_trainer = gen_trainer
        self.disc_trainer = disc_trainer

        self.n_critic_init = n_critic_init
        self.n_critic = n_critic
        self.lmd = lmd
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
                xg, xd, xp, self.lmd,
            )

        info.accuracy = self.disc.accuracy(xg, xd)
        info.gen_out = gen_out
        info.dynamics_penalty = gen_out.model_dynamics_penalty
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

    def _single_gen_step(self, gen_step, n_critic):
        self.gen_forward_watch = StopWatch()
        self.gen_train_watch = StopWatch()
        self.disc_train_watch = StopWatch()

        for disc_step in range(self.n_critic_init):
            info = SimpleNamespace(is_discriminator=True, gen_step=gen_step,
                                   disc_step=disc_step)
            yield self.train_discriminator(info)

        info = SimpleNamespace(is_discriminator=False, gen_step=gen_step)
        yield self.train_generator(info)

    def learning(self):
        for info in self._single_gen_step(0, self.n_critic_init):
            yield info
        for gen_step in itertools.count(1):
            for info in self._single_gen_step(gen_step, self.n_critic):
                yield info
