import collections

import lasagne
import numpy as np
import theano

from .. import ssnode
from ..core import BaseComponent
from ..gradient_expressions.make_w_batch import make_W_with_x
from ..utils import cached_property, theano_function, log_timing


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
    r"""
    Stimulator for varying bandwidths and contrasts.

    We set the stimulus input to excitatory and inhibitory neurons at
    site :math:`i` to

    .. math::
       :label: BandwidthContrastStimulator-input

       I_i(s) = A\,
       \sigma\!\left( \frac{{s}/{2} + x_i}{l} \right)\,
       \sigma\!\left( \frac{{s}/{2} - x_i}{l} \right)

    where :math:`\sigma(u) = (1+\exp(-u))^{-1}` is the logistic
    function, :math:`A` denotes the stimulus intensity (`contrast`),
    and :math:`s \in S` (= `bandwidths`) are the stimulus size.

    Attributes
    ----------
    num_sites : int
        Size of the SSN to be stimulated.

    num_tcdom : int
        Number of points in :term:`tuning curve domain` (TC dom).

    bandwidths : theano.vector.tensor
        :math:`s` in :eq:`BandwidthContrastStimulator-input`

    contrasts : theano.vector.tensor
        :math:`A` in :eq:`BandwidthContrastStimulator-input`

    smoothness : float
        :math:`l` in :eq:`BandwidthContrastStimulator-input`

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


class EulerSSNCore(BaseComponent):

    def __init__(self, stimulator, J, D, S, k, n, tau_E, tau_I, dt,
                 io_type):

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
                                stimulator.site_to_band)[0].T
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
        self.ssn, rest = EulerSSNCore.consume_kwargs(**kwargs)

        super(EulerSSNLayer, self).__init__(incoming, **rest)

        self.add_param(self.ssn.J, (2, 2), name='J')
        self.add_param(self.ssn.D, (2, 2), name='D')
        self.add_param(self.ssn.S, (2, 2), name='S')

    def get_output_for(self, input, **kwargs):
        return self.ssn.get_output_for(input, **kwargs)


class EulerSSNModel(BaseComponent):

    r"""
    Implementation of SSN in Theano.

    Attributes
    ----------
    dynamics_penalty : Theano scalar expression
        :math:`\mathtt{mean}_{b,i,t} (r_{b,i}(t) - r_{b,i}(t - \Delta t))^2`

    """

    def __init__(self, stimulator, J, D, S, k, n, tau_E, tau_I, dt,
                 io_type,
                 unroll_scan=False,
                 skip_steps=None, seqlen=None, batchsize=None):
        self.stimulator = stimulator
        self.skip_steps = int_or_lscalr(skip_steps, 'sample_beg')
        self.batchsize = int_or_lscalr(batchsize, 'batchsize')
        self.seqlen = int_or_lscalr(seqlen, 'seqlen')

        # num_tcdom = "batch dimension" when using Lasagne:
        num_neurons = self.stimulator.num_neurons
        shape = (self.stimulator.num_tcdom, self.seqlen, num_neurons)

        shape_rec = (shape[0],) + shape[2:]
        self.l_ssn = EulerSSNLayer(
            lasagne.layers.InputLayer(shape_rec),
            stimulator=stimulator,
            J=J, D=D, S=S, k=k, n=n, tau_E=tau_E, tau_I=tau_I, dt=dt,
            io_type=io_type,
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
            unroll_scan=unroll_scan,
        )
        self.unroll_scan = unroll_scan

        self.trajectories = rates = lasagne.layers.get_output(self.l_rec)
        rs = rates[:, self.skip_steps:]
        time_avg = rs.mean(axis=1)  # shape: (num_tcdom, num_neurons)
        dynamics_penalty = ((rs[:, 1:] - rs[:, :-1]) ** 2).mean()

        # self.zs.shape: (batchsize, num_neurons, num_neurons)
        self.zs = theano.tensor.tensor3('zs')
        # self.time_avg.shape: (batchsize, num_tcdom, num_neurons)
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

    @cached_property
    def compute_time_avg(self):
        return theano_function(
            (self.zs,) + self.stimulator.inputs,
            self.time_avg,
        )


class FixedProber(BaseComponent):

    """
    Probe `model` output with constant `probe`.

    Here, fixed/constant means that `probe` does not co-vary with
    batches and the points in :term:`tuning curve domain`.

    For a demonstration purpose, let's setup a fake model.  All
    `FixedProber` cares are `.time_avg` and `.batchsize` attributes:

    >>> from types import SimpleNamespace
    >>> batchsize = 3
    >>> num_tcdom = 5
    >>> num_neurons = 7
    >>> time_avg = np.arange(batchsize * num_tcdom * num_neurons)
    >>> time_avg = time_avg.reshape((batchsize, num_tcdom, num_neurons))
    >>> model = SimpleNamespace(
    ...     time_avg=theano.shared(time_avg),
    ...     batchsize=batchsize)

    To probe `model.time_avg`, give the indices of neurons to be
    probed as `probes` argument:

    >>> probes = np.array([0, 5])
    >>> prober = FixedProber(model, probes)

    Then, `FixedProber` mixes the probe axis into :term:`tuning curve
    domain` axis.  So from the point of view of the discriminator, the
    probe offset is yet another axis in the :term:`tuning curve
    domain`.

    >>> tuning_curve = prober.tuning_curve.eval()
    >>> assert tuning_curve.shape == (batchsize, num_tcdom * len(probes))
    >>> tuning_curve
    array([[  0,   5,   7,  12,  14,  19,  21,  26,  28,  33],
           [ 35,  40,  42,  47,  49,  54,  56,  61,  63,  68],
           [ 70,  75,  77,  82,  84,  89,  91,  96,  98, 103]])

    For each batches, the first ``len(probes)``-slice of
    `.tuning_curve` along the second axis corresponds to the first
    point in :term:`tuning curve domain` (i.e., index 0 of the second
    axis of `time_avg`):

    >>> np.testing.assert_equal(time_avg[:, 0, probes],
    ...                         tuning_curve[:, :len(probes)])

    Similarly, the second ``len(probes)``-slice corresponds to the
    second point in :term:`tuning curve domain` (i.e., index 1 of the
    second axis of `time_avg`):

    >>> np.testing.assert_equal(time_avg[:, 1, probes],
    ...                         tuning_curve[:, len(probes):2 * len(probes)])

    """

    inputs = ()

    def __init__(self, model, probes):
        self.model = model
        self.probes = probes

        # time_avg.shape: (batchsize, num_tcdom, num_neurons)
        tc = self.model.time_avg[:, :, self.probes]
        # tuning_curve.shape: (batchsize, num_tcdom * len(probes))
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
    output_shape = property(lambda self: (self.batchsize,
                                          self.num_tcdom * len(self.probes)))
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

    def prepare(self):
        """ Force compile Theano functions. """
        with log_timing("compiling {}._forward"
                        .format(self.__class__.__name__)):
            self._forward
