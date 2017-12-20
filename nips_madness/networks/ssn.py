"""
Tuning curve generator based on SSN.

This module implements tuning curve generator (`TuningCurveGenerator`)
based on SSN (`EulerSSNModel`).  For a high-level overview, see
:ref:`general-construction`.

The (default) tuning curve generator is composed of the following
components:

- `TuningCurveGenerator`

  - `BandwidthContrastStimulator` (:term:`stimulator`)
  - `EulerSSNModel` (:term:`fixed-point solver`)
  - `FixedProber` (:term:`prober`)

`TuningCurveGenerator` is designed in a way that each component is
pluggable.  See, e.g., `.ConditionalTuningCurveGenerator`.

`EulerSSNModel` is further composed of its sub-components
`EulerSSNCore` and `EulerSSNLayer`.  This additional complexity is for
safely using `lasagne.layers.CustomRecurrentLayer` so that its
`unroll_scan` option can be used to boost computation throughput.

This module also include `MapCloneEulerSSNModel` (and
`MapCloneEulerSSNCore`) which is the old version of `EulerSSNModel`
(and `EulerSSNCore`) implemented in a different strategy (i.e.,
combination of `theano.map` and `theano.clone`).  It seems that
non-MapClone version is faster but no extensive benchmark has been
done.  Let's keep those implementation at the moment.

For the usage of `TuningCurveGenerator`, see: `.BPTTWassersteinGAN`,
and `.BPTTMomentMatcher`.

"""

import abc
import collections

import lasagne
import numpy as np
import theano

from .. import ssnode
from ..core import BaseComponent, consume_config
from ..gradient_expressions.make_w_batch import make_W_with_x
from ..utils import cached_property, theano_function, log_timing, asarray, \
    is_theano, get_array_module


def int_or_lscalr(value, name):
    if value is None:
        raise NotImplementedError
        # The intention was to support variable sizes (e.g.,
        # batchsize, num_sites, etc.) but it is not supported at the
        # moment.
        return theano.tensor.lscalar(name)
    else:
        return value


def neu_array(pop_array, num_sites, pops=None):
    """
    Tile each element of `pop_array` for `num_sites` times.

    >>> neu_array([0, 1], 2)
    array([0, 0, 1, 1])

    It works with Theano array too, provided the third argument `pops`
    is provided:

    >>> pa = theano.tensor.vector('x', dtype='int16')
    >>> na = neu_array(pa, 2, 3)
    >>> na.eval({pa: np.array([0, 1, 2], dtype=pa.dtype)})
    array([0, 0, 1, 1, 2, 2], dtype=int16)

    """
    pop_array = asarray(pop_array)
    if not is_theano(pop_array):
        pops = pop_array.shape[0]
    xp = get_array_module(pop_array)

    return xp.concatenate([xp.tile(pop_array[i], num_sites)
                           for i in range(pops)])


def maybe_mixin_noise(self, rng=None, kwargs={}):
    if rng:
        noise = self.gen_noise(rng, **kwargs)
        if set(noise) & set(kwargs):
            raise ValueError("keys {} and 'rng' cannot be specified"
                             " simultaneously"
                             .format(sorted(noise)))
        kwargs.update(noise)
    return kwargs


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

    bandwidths : theano.vector.matrix
        shape: ``(batchsize, num_tcdom)``;
        :math:`s` in :eq:`BandwidthContrastStimulator-input`

    contrasts : theano.vector.matrix
        shape: ``(batchsize, num_tcdom)``;
        :math:`A` in :eq:`BandwidthContrastStimulator-input`

    smoothness : float
        :math:`l` in :eq:`BandwidthContrastStimulator-input`

    """

    outputs = ()

    def __init__(self, num_sites, num_tcdom, smoothness):
        self.num_sites = int_or_lscalr(num_sites, 'num_sites')
        self.num_tcdom = int_or_lscalr(num_tcdom, 'num_tcdom')

        self.bandwidths = theano.tensor.matrix('bandwidths')
        self.contrasts = theano.tensor.matrix('contrasts')
        self.smoothness = smoothness
        self.site_to_band = np.linspace(-0.5, 0.5, self.num_sites,
                                        dtype=theano.config.floatX)

        def sigm(x):
            from theano.tensor import exp
            return 1 / (1 + exp(-x / smoothness))

        # x, b, c stim: arrays broadcastable to:
        #     (batchsize, num_tcdom, num_sites)
        x = self.site_to_band.reshape((1, 1, -1))
        b = theano.tensor.shape_padright(self.bandwidths)
        c = theano.tensor.shape_padright(self.contrasts)
        stim = c * sigm(x + b/2) * sigm(b/2 - x)

        self.stimulus = theano.tensor.tile(stim, (1, 1, 2))
        """
        A symbolic array of shape (`.batchsize`, `.num_tcdom`, `.num_neurons`).
        """

        self.inputs = (self.bandwidths, self.contrasts)

    num_neurons = property(lambda self: self.num_sites * 2)

    def get_all_params(self):
        return []


class AbstractEulerSSNCore(BaseComponent, abc.ABC):

    """
    Abstract base class for implementing single Euler step for SSN.

    See: `EulerSSNCore`.

    """

    def __init__(self, stimulator, ext, J, D, S, k, n, tau_E, tau_I, dt,
                 io_type):

        J = np.asarray(J, dtype=theano.config.floatX)
        D = np.asarray(D, dtype=theano.config.floatX)
        S = np.asarray(S, dtype=theano.config.floatX)
        assert J.shape == D.shape == S.shape == (2, 2)

        self.stimulator = stimulator
        self.ext = ext
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

        self.f = ssnode.make_io_fun(self.k, self.n, io_type=self.io_type)

        self.post_init()

    @abc.abstractmethod
    def post_init(self):
        """ Called at the end of `__init__`. """

    @abc.abstractmethod
    def get_output_for(self, r):
        """
        Calculate next Euler step output of given input `r`.

        Called via `.EulerSSNLayer.get_output_for`.
        See also: `lasagne.layers.Layer.get_output_for`
        """

    def get_all_params(self):
        params = self.stimulator.get_all_params()
        return params + [self.J, self.D, self.S]


class MapCloneEulerSSNCore(AbstractEulerSSNCore):

    """
    Implementation of single Euler step for SSN.

    See: `EulerSSNCore`.

    """

    def post_init(self):
        stimulator = self.stimulator
        num_sites = stimulator.num_sites

        self.zmat = theano.tensor.matrix('zmat')
        self.Wt = make_W_with_x(theano.tensor.shape_padleft(self.zmat),
                                self.J, self.D, self.S,
                                num_sites,
                                stimulator.site_to_band)[0].T
        self.Wt.name = 'Wt'

    def get_output_for(self, r):
        f = self.f
        Wt = self.Wt
        I = self.ext
        eps = self.eps.reshape((1, -1))
        co_eps = (1 - self.eps).reshape((1, -1))
        r_next = co_eps * r + eps * f(r.dot(Wt) + I)
        r_next = theano.tensor.patternbroadcast(r_next, r.broadcastable)
        # patternbroadcast is required for the case num_tcdom=1.
        r_next.name = 'r_next'
        return r_next


class EulerSSNLayer(lasagne.layers.Layer):

    # See:
    # http://lasagne.readthedocs.io/en/latest/user/custom_layers.html
    # http://lasagne.readthedocs.io/en/latest/modules/layers/base.html

    def __init__(self, incoming, ssn, **kwargs):
        # It's a bit scary to share namespace with lasagne; so let's
        # introduce only one namespace and keep SSN-specific stuff
        # there:
        self.ssn = ssn

        super(EulerSSNLayer, self).__init__(incoming, **kwargs)

        for param in ssn.get_all_params():
            shape = param.get_value().shape
            self.add_param(param, shape)

    def get_output_for(self, input, **kwargs):
        return self.ssn.get_output_for(input, **kwargs)


class AbstractEulerSSNModel(BaseComponent, abc.ABC):

    r"""
    Abstract base class of SSN.

    Attributes
    ----------
    dynamics_penalty : Theano scalar expression
        :math:`\mathtt{mean}_{b,i,t} (r_{b,i}(t) - r_{b,i}(t - \Delta t))^2`

    """

    @abc.abstractmethod
    def __init__(self):
        pass

    def _setup_layers(self, ssn, shape, unroll_scan):
        shape_sym = shape
        if is_theano(shape[0]):
            shape = (None,) + shape[1:]
        shape_rec = (shape[0],) + shape[2:]
        self.l_ssn = EulerSSNLayer(
            lasagne.layers.InputLayer(shape_rec),
            ssn,
        )

        # Since SSN is autonomous, we don't need to do anything for
        # input and input_to_hidden layers.  So let's zero them out:
        self.l_fake_input = lasagne.layers.InputLayer(
            shape,
            theano.tensor.zeros(shape_sym),
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

    dt = property(lambda self: self.l_ssn.ssn.dt)
    io_type = property(lambda self: self.l_ssn.ssn.io_type)
    J = property(lambda self: self.l_ssn.ssn.J)
    D = property(lambda self: self.l_ssn.ssn.D)
    S = property(lambda self: self.l_ssn.ssn.S)
    get_all_params = property(lambda self: self.l_ssn.ssn.get_all_params)

    num_tcdom = property(lambda self: self.stimulator.num_tcdom)
    num_sites = property(lambda self: self.stimulator.num_sites)
    num_neurons = property(lambda self: self.stimulator.num_neurons)

    @property
    def _input_names(self):
        return collect_names(['', 'stimulator_'],
                             [self.inputs, self.stimulator.inputs])

    @cached_property
    def compute_trajectories_impl(self):
        return theano_function(
            self.inputs + self.stimulator.inputs,
            self.all_trajectories,
        )

    def compute_trajectories(self, rng=None, **kwargs):
        kwargs = maybe_mixin_noise(self, rng, kwargs)
        values = [kwargs.pop(k) for k in self._input_names]
        assert not kwargs
        return self.compute_trajectories_impl(*values)

    @cached_property
    def compute_time_avg_impl(self):
        return theano_function(
            self.inputs + self.stimulator.inputs,
            self.time_avg,
        )

    def compute_time_avg(self, rng=None, **kwargs):
        kwargs = maybe_mixin_noise(self, rng, kwargs)
        values = [kwargs.pop(k) for k in self._input_names]
        assert not kwargs
        return self.compute_time_avg_impl(*values)

    def gen_noise(self, rng, stimulator_bandwidths, **_):
        batchsize, _num_tcdom = stimulator_bandwidths.shape
        num_neurons = self.num_neurons
        return dict(
            zs=rng.rand(batchsize, num_neurons, num_neurons)
        )


class MapCloneEulerSSNModel(AbstractEulerSSNModel):

    """
    Implementation of SSN in Theano (based on map & clone combo).
    """

    @classmethod
    def consume_kwargs(cls, stimulator, **kwargs):
        ssn, kwargs = MapCloneEulerSSNCore.consume_kwargs(
            stimulator=stimulator,
            ext=theano.tensor.matrix('ext'),
            **kwargs)
        return super(MapCloneEulerSSNModel, cls).consume_kwargs(
            stimulator=stimulator,
            ssn=ssn,
            **kwargs)
    # Setting ext=theano.tensor.matrix('ext') here is not very clean.
    # Ideally, it should be implemented in MapCloneEulerSSNCore.
    # However, since AbstractEulerSSNCore.__init__ is not short and
    # its call signature has to be modified if ext were to be defined
    # in MapCloneEulerSSNCore.  This is why ext is created here.

    def __init__(self, stimulator, ssn,
                 unroll_scan=False,
                 skip_steps=None, seqlen=None,
                 include_time_avg=False):
        self.stimulator = stimulator
        self.skip_steps = int_or_lscalr(skip_steps, 'sample_beg')
        self.seqlen = int_or_lscalr(seqlen, 'seqlen')

        self.ext = ssn.ext
        # num_tcdom = "batch dimension" when using Lasagne:
        num_neurons = self.stimulator.num_neurons
        shape = (self.stimulator.num_tcdom, self.seqlen, num_neurons)
        self._setup_layers(ssn, shape, unroll_scan)

        self.trajectories = rates = lasagne.layers.get_output(self.l_rec)
        rs = rates[:, self.skip_steps:]
        time_avg = rs.mean(axis=1)  # shape: (num_tcdom, num_neurons)
        dynamics_penalty = ((rs[:, 1:] - rs[:, :-1]) ** 2).mean()

        # self.zs.shape: (batchsize, num_neurons, num_neurons)
        self.zs = theano.tensor.tensor3('zs')
        # self.time_avg.shape: (batchsize, num_tcdom, num_neurons)
        self.time_avg = self._map_clone(time_avg)
        self.time_avg.name = 'time_avg'
        self.dynamics_penalty = self._map_clone(dynamics_penalty).mean()
        # TODO: make sure theano.clone is not bottleneck here.
        self.dynamics_penalty.name = 'dynamics_penalty'
        # TODO: find if assigning a name to an expression is a valid usecase

        self.inputs = (self.zs,)
        if include_time_avg:
            self.outputs = (self.dynamics_penalty, self.time_avg)
        else:
            self.outputs = (self.dynamics_penalty,)

    zmat = property(lambda self: self.l_ssn.ssn.zmat)

    def _map_clone(self, expr):
        return theano.map(
            lambda z, I: theano.clone(expr, {
                self.zmat: z,
                self.ext: I,
            }),
            [self.zs, self.stimulator.stimulus],
        )[0]

    @cached_property
    def all_trajectories(self):
        """
        Symbolic expression for all trajectories across batches and stimuli.

        Array of shape (`.batchsize`, `.num_tcdom`, `.seqlen`, `.num_neurons`).
        """
        return self._map_clone(self.trajectories)


class EulerSSNCore(AbstractEulerSSNCore):

    """
    Implementation of single Euler step for SSN.

    Attributes
    ----------
    zs : theano.tensor.tensor3
        Array of shape (`.batchsize`, `.num_neurons`, `.num_neurons`).
        The randomness part of the connectivity matrix.
    Wt : theano.tensor.var.TensorVariable
        Array of shape (`.batchsize`, `.num_neurons`, `.num_neurons`).
        Transposed connectivity matrix; i.e., ``Wt[k].T[i, j]`` is the
        element :math:`W_{ij}` of the connectivity matrix for the
        ``k``-th batch.

    """

    def post_init(self):
        stimulator = self.stimulator
        num_sites = stimulator.num_sites

        self.zs = theano.tensor.tensor3('zs')
        self.Wt = make_W_with_x(
            self.zs, self.J, self.D, self.S, num_sites,
            stimulator.site_to_band,
        ).swapaxes(1, 2)
        self.Wt.name = 'Wt'

    def get_output_for(self, r):
        """
        Get computation graph representing a single Euler step.

        Parameters
        ----------
        r : theano.tensor.tensor3
            An array of shape (`.batchsize`, `.num_tcdom`, `.num_neurons`)
            holding network state.

        """
        f = self.f
        Wt = self.Wt
        I = self.ext
        eps = self.eps.reshape((1, 1, -1))
        co_eps = (1 - self.eps).reshape((1, 1, -1))
        Wr = theano.tensor.batched_dot(r, Wt)
        r_next = co_eps * r + eps * f(Wr + I)
        r_next = theano.tensor.patternbroadcast(r_next, r.broadcastable)
        # patternbroadcast is required for the case num_tcdom=1.
        r_next.name = 'r_next'
        return r_next


class EulerSSNModel(AbstractEulerSSNModel):

    """
    Implementation of SSN in Theano (based on `batched_dot`).
    """

    @classmethod
    def consume_kwargs(cls, stimulator, **kwargs):
        ssn, kwargs = EulerSSNCore.consume_kwargs(
            stimulator=stimulator,
            ext=stimulator.stimulus,
            **kwargs)
        return super(EulerSSNModel, cls).consume_kwargs(
            stimulator=stimulator,
            ssn=ssn,
            **kwargs)

    def __init__(self, stimulator, ssn,
                 unroll_scan=False,
                 skip_steps=None, seqlen=None,
                 include_time_avg=False):
        self.stimulator = stimulator
        self.skip_steps = int_or_lscalr(skip_steps, 'sample_beg')
        self.seqlen = int_or_lscalr(seqlen, 'seqlen')
        self.include_time_avg = include_time_avg

        num_neurons = self.stimulator.num_neurons
        # self.zs.shape: (batchsize, num_neurons, num_neurons)
        self.zs = ssn.zs
        batchsize = self.zs.shape[0]
        shape = (batchsize,
                 self.seqlen,
                 self.stimulator.num_tcdom,
                 num_neurons)
        self._setup_layers(ssn, shape, unroll_scan)

        # trajectories.shape: (batchsize, seqlen, num_tcdom, num_neurons)
        self.trajectories = rates = lasagne.layers.get_output(self.l_rec)
        rs = rates[:, self.skip_steps:]

        # self.time_avg.shape: (batchsize, num_tcdom, num_neurons)
        self.time_avg = rs.mean(axis=1)
        self.time_avg.name = 'time_avg'
        self.dynamics_penalty = ((rs[:, 1:] - rs[:, :-1]) ** 2).mean()
        self.dynamics_penalty.name = 'dynamics_penalty'

    @property
    def inputs(self):
        return (self.zs,)

    @property
    def outputs(self):
        if self.include_time_avg:
            return (self.dynamics_penalty, self.time_avg)
        else:
            return (self.dynamics_penalty,)

    @property
    def all_trajectories(self):
        """
        Symbolic expression for all trajectories across batches and stimuli.

        Array of shape (`.batchsize`, `.num_tcdom`, `.seqlen`, `.num_neurons`).
        """
        return self.trajectories.swapaxes(1, 2)


class HeteroInpuptWrapper(BaseComponent):
    """
    Stimulator wrapper.
    """

    def __init__(self, stimulator, V=0):
        self.stimulator = stimulator

        V = np.asarray(V, dtype=theano.config.floatX)
        V = np.broadcast_to(V, 2)
        self.V = theano.shared(V, name='V')

        # self.zs_in.shape: (batchsize, num_neurons)
        self.zs_in = theano.tensor.matrix('zs_in')

        # self.amplification, vs, zs: arrays broadcastable to:
        #     (batchsize, num_tcdom, num_neurons)
        vs = neu_array(self.V, stimulator.num_sites, 2).reshape((1, 1, -1))
        zs = self.zs_in.reshape((self.zs_in.shape[0], 1, self.zs_in.shape[1]))
        self.amplification = amp = 1 + vs * zs

        self.stimulus = amp * self.stimulator.stimulus

    def get_all_params(self):
        params = self.stimulator.get_all_params()
        return params + [self.V]

    def gen_noise(self, rng, stimulator_bandwidths, **_):
        batchsize, _num_tcdom = stimulator_bandwidths.shape
        num_neurons = self.num_neurons
        return dict(
            zs_in=rng.rand(batchsize, num_neurons) * 2 - 1,
        )

    def __getattr__(self, name):
        """ Trun ``self.XXX`` into ``self.stimulator.XXX``. """
        try:
            super(HeteroInpuptWrapper, self).__getattr__(name)
        except AttributeError:
            return getattr(self.stimulator, name)


class HeteroInEulerSSNModel(EulerSSNModel):
    """
    SSN with heterogeneous (random) input.
    """

    input_wrapper_class = HeteroInpuptWrapper

    @classmethod
    def consume_kwargs(cls, stimulator, **kwargs):
        stimulator, kwargs = cls.input_wrapper_class.consume_kwargs(
            stimulator=stimulator,
            **kwargs)
        return super(HeteroInEulerSSNModel, cls).consume_kwargs(
            stimulator=stimulator,
            **kwargs)

    @property
    def inputs(self):
        return (self.zs, self.stimulator.zs_in)

    def gen_noise(self, rng, **kwargs):
        noise = super(HeteroInEulerSSNModel, self).gen_noise(rng, **kwargs)
        noise.update(self.stimulator.gen_noise(rng, **kwargs))
        return noise


class FixedProber(BaseComponent):

    """
    Probe `model` output with constant `probe`.

    Here, fixed/constant means that `probe` does not co-vary with
    batches and the points in :term:`tuning curve domain`.

    For a demonstration purpose, let's setup a fake model.  Since all
    `FixedProber` cares are `.time_avg` and `.num_tcdom` attributes,
    it is easy to mock the model for it:

    >>> from types import SimpleNamespace
    >>> batchsize = 3
    >>> num_tcdom = 5
    >>> num_neurons = 7
    >>> time_avg = np.arange(batchsize * num_tcdom * num_neurons)
    >>> time_avg = time_avg.reshape((batchsize, num_tcdom, num_neurons))
    >>> model = SimpleNamespace(
    ...     time_avg=theano.shared(time_avg),
    ...     num_tcdom=num_tcdom)

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
        self.probes = asarray(probes)

        num_tcdom = self.model.num_tcdom
        num_probes = self.probes.shape[0]

        # time_avg.shape: (batchsize, num_tcdom, num_neurons)
        tc = self.model.time_avg[:, :, self.probes]
        # tuning_curve.shape: (batchsize, num_tcdom * len(probes))
        self.tuning_curve = tc.reshape((-1, num_tcdom * num_probes))
        self.tuning_curve.name = 'tuning_curve'

        self.outputs = (self.tuning_curve,)


def collect_names(prefixes, var_lists):
    names = []
    for prefix, var_list in zip(prefixes, var_lists):
        for var in var_list:
            names.append(prefix + var.name)
    return names


class TuningCurveGenerator(BaseComponent):

    def __init__(self, stimulator, model, prober, batchsize):
        self.stimulator = stimulator
        self.model = model
        self.prober = prober
        self.batchsize = batchsize

        out_names = collect_names(['stimulator_', 'model_', 'prober_'],
                                  [self.stimulator.outputs,
                                   self.model.outputs,
                                   self.prober.outputs])
        self.OutType = collections.namedtuple('OutType', out_names)

        # TODO: make _input_names public
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
    def forward_impl(self):
        return theano_function(self.inputs, self.outputs)

    def forward(self, rng=None, **kwargs):
        kwargs = maybe_mixin_noise(self, rng, kwargs)
        values = [kwargs.pop(k) for k in self._input_names]
        assert not kwargs
        return self.OutType(*self.forward_impl(*values))

    def gen_noise(self, rng, **kwargs):
        model_noise = self.model.gen_noise(rng, **kwargs)
        return {
            'model_' + k: v for k, v in model_noise.items()
        }
        # .keys() is ['model_zs'] for EulerSSNModel
        # .keys() is ['model_zs', 'model_zs_in'] for HeteroInEulerSSNModel

    @property
    def get_all_params(self):
        return self.model.get_all_params

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
        with log_timing("compiling {}.forward_impl"
                        .format(self.__class__.__name__)):
            self.forward_impl


_ssn_classes = {
    ('default', 'default'): EulerSSNModel,
    ('mapclone', 'default'): MapCloneEulerSSNModel,
    ('default', 'heteroin'): HeteroInEulerSSNModel,
}
"""
Mapping form ``(ssn_impl, ssn_type)`` to SSN model class.
"""

ssn_impl_choices = ('default', 'mapclone')
ssn_type_choices = ('default', 'heteroin')
assert set(ssn_impl_choices) == set(k for k, _ in _ssn_classes)
assert set(ssn_type_choices) == set(k for _, k in _ssn_classes)


def emit_ssn(*args, ssn_impl='default', ssn_type='default', **kwargs):
    cls = _ssn_classes[ssn_impl, ssn_type]
    return cls.consume_kwargs(*args, **kwargs)


def emit_tuning_curve_generator(
        emit_stimulator=BandwidthContrastStimulator.consume_kwargs,
        emit_model=emit_ssn,
        emit_prober=FixedProber.consume_kwargs,
        emit_tcg=TuningCurveGenerator.consume_kwargs,
        consume_union=True,
        **kwargs):
    stimulator, kwargs = emit_stimulator(**kwargs)
    model, kwargs = emit_model(stimulator=stimulator, **kwargs)
    prober, kwargs = emit_prober(model=model, **kwargs)
    tgc, kwargs = emit_tcg(stimulator, model, prober, **kwargs)
    if consume_union:
        for key in ['V']:
            kwargs.pop(key, None)
    return tgc, kwargs


def make_tuning_curve_generator(config, **kwargs):
    """make_tuning_curve_generator(config: dict, **kwargs) -> (TCG, dict)
    Return a tuple of a tuning curve generator and an unused part of `config`.
    """
    return consume_config(emit_tuning_curve_generator, config, **kwargs)
