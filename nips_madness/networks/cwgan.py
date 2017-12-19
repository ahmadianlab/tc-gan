from types import SimpleNamespace

import lasagne
import numpy as np
import theano

from . import simple_discriminator
from ..core import BaseComponent, consume_subdict
from ..evaluator import MagicEvaluator
from ..gradient_expressions.utils import sample_sites_from_stim_space_impl
from ..utils import (
    cached_property, cartesian_product, theano_function, StopWatch,
    as_randomstate,
)
from .bptt_gan import (
    BaseTrainer, BPTTWassersteinGAN, DEFAULT_PARAMS, GeneratorTrainer,
)
from .ssn import TuningCurveGenerator, emit_tuning_curve_generator

DEFAULT_PARAMS = dict(
    DEFAULT_PARAMS,
    num_models=1,
    probes_per_model=1,
    e_ratio=0.8,
    hide_cell_type=False,
)
del DEFAULT_PARAMS['batchsize']
del DEFAULT_PARAMS['sample_sites']


class ConditionalProber(BaseComponent):

    """
    Probe `model` output with varying probe.

    Attributes
    ----------
    norm_probes : theano.tensor.vector
        Normalized probes.
        Array of length `.batchsize` specifying probe offset in
        ``[-1, 1]`` bandwidth coordinate.

    cell_types : theano.tensor.vector
        Array of length `.batchsize` and type uint16, specifying cell
        type (0 means excitatory and 1 means inhibitory).

    model_ids : theano.tensor.vector
        Array of length `.batchsize` and type uint16, specifying
        instances of model; i.e., each element is an index `i` such
        that the matrix `zs[i] <.EulerSSNCore.zs>` specifies a matrix
        of shape (`.num_neurons`, `.num_neurons`) for the
        "randomness/noise part" of the connectivity matrix.  Must take
        values between 0 (inclusive) and `.num_models` (exclusive).

    probes : theano.tensor.var.TensorVariable
        Array of length `.batchsize` and type uint16.
        Same as `.norm_probes` but in neural index.

    contrasts : theano.tensor.var.TensorVariable
        Array of length `.batchsize`.  This is a reshaped one
        dimensional "view" of `stimulator.contrasts
        <.BandwidthContrastStimulator.contrasts>`.

    tuning_curve : theano.tensor.var.TensorVariable
        Array of shape (`.batchsize`, `.num_tcdom`).

    """

    def __init__(self, model):
        self.model = model

        self.norm_probes = theano.tensor.vector('norm_probes')
        self.cell_types = theano.tensor.vector('cell_types', 'uint16')
        self.model_ids = theano.tensor.vector('model_ids', 'uint16')
        # Invariants:
        #    * length of those vectors = batchsize
        #    * all(-1 <= norm_probes <= 1)
        #    * all(0 <= model_ids < num_models)
        #    * all(cell_types in {0, 1})

        # Assuming that contrast is not varied along tcdom
        # (bandwidths) axis (so it's OK to choose any index):
        self.contrasts = self.model.stimulator.contrasts[self.model_ids, 0]
        self.contrasts.name = 'contrasts'

        num_sites = self.model.num_sites

        self.probes = sample_sites_from_stim_space_impl(
            self.norm_probes, num_sites, type='uint16',
        ) + self.cell_types * num_sites
        self.probes.name = 'probes'

        # time_avg.shape: (num_models, num_tcdom, num_neurons)
        # self.tuning_curve.shape: (batchsize, num_tcdom)
        self.tuning_curve = self.model.time_avg[self.model_ids, :, self.probes]
        self.tuning_curve.name = 'tuning_curve'

        self.inputs = (self.norm_probes, self.model_ids, self.cell_types)
        self.outputs = (self.tuning_curve,)

        self.eval = MagicEvaluator(self)


class ConditionalTuningCurveGenerator(TuningCurveGenerator):

    output_shape = property(lambda self: (self.batchsize, self.num_tcdom))
    cond_shape = property(lambda self: (self.batchsize, 3))
    # See: get_output

    def get_output(self):
        conditions = theano.tensor.as_tensor_variable([
            self.prober.contrasts,
            self.prober.norm_probes,
            self.prober.cell_types.astype(theano.config.floatX),
        ]).T  # shape: (batchsize, 3)
        conditions.name = 'conditions'
        return [self.prober.outputs[0], conditions]


class ConditionalDiscriminator(BaseComponent):

    def __init__(self, shape, cond_shape, loss_type,
                 layers, normalization, nonlinearity):
        self.l_in = lasagne.layers.InputLayer(shape)
        self.l_cond = lasagne.layers.InputLayer(cond_shape)
        self.l_eff_in = lasagne.layers.ConcatLayer([self.l_in, self.l_cond])

        l_hidden = simple_discriminator.stack_hidden_layers(
            self.l_eff_in, layers=layers,
            normalization=normalization,
            nonlinearity=nonlinearity)
        self.l_out = simple_discriminator.make_output_layer(
            l_hidden, loss_type=loss_type)

        from theano import tensor as T
        self.xg = xg = T.matrix()  # \tilde x  --  "fake image"
        self.xd = xd = T.matrix()  # x         --  ground truth
        self.cg = cg = T.matrix()
        self.cd = cd = T.matrix()
        Dg = self.get_output([xg, cg])
        Dd = self.get_output([xd, cd])
        self.accuracy_expr = Dg.mean() - Dd.mean()

    @cached_property
    def accuracy(self):
        return theano_function([self.xg, self.cg, self.xd, self.cd],
                               self.accuracy_expr)

    def get_all_params(self):
        return lasagne.layers.get_all_params(self.l_out, trainable=True)

    def get_output(self, inputs=None):
        if isinstance(inputs, list):
            tc, condition = inputs
            inputs = [tc, self.preprocess_condition(condition)]
            inputs = dict(zip([self.l_in, self.l_cond], inputs))
        return lasagne.layers.get_output(self.l_out, inputs=inputs)

    def preprocess_condition(self, condition):
        # Feed absolute value of norm_probes since SSN is symmetric.
        return theano.tensor.as_tensor_variable([
            condition[:, 0],       # contrasts
            abs(condition[:, 1]),  # norm_probes
            condition[:, 2],       # cell_types
        ]).T

    def prepare(self):
        """ Force compile Theano functions. """
        self.accuracy


class CellTypeBlindDiscriminator(ConditionalDiscriminator):

    def preprocess_condition(self, condition):
        condition = super(CellTypeBlindDiscriminator, self) \
            .preprocess_condition(condition)
        # Feed absolute value of norm_probes since SSN is symmetric.
        return theano.tensor.as_tensor_variable([
            condition[:, 0],
            condition[:, 1],
            theano.tensor.zeros_like(condition[:, 2])  # cell_types
        ]).T


class ConditionalCriticTrainer(BaseTrainer):
    """
    Discriminator/Critic trainer for conditional WGAN.
    """

    def __init__(self, disc, updater):
        self.target = disc
        self.updater = updater

        from theano import tensor as T
        self.lmd = lmd = T.scalar()
        self.xg = xg = T.matrix()  # \tilde x  --  "fake image"
        self.xd = xd = T.matrix()  # x         --  ground truth
        self.xp = xp = T.matrix()  # \hat x    --  interpolated sample
        self.cg = cg = T.matrix()
        self.cd = cd = T.matrix()
        self.cp = cp = T.matrix()
        self.disc_gene = disc.get_output([xg, cg]).mean()
        self.disc_data = disc.get_output([xd, cd]).mean()
        self.disc_intp = disc.get_output([xp, cp])[:, 0]
        self.dgrad = T.jacobian(self.disc_intp, xp)
        self.penalty = ((self.dgrad.norm(2, axis=(1, 2)) - 1)**2).mean()

        self.loss = self.disc_gene - self.disc_data + lmd * self.penalty
        self.inputs = (xg, xd, xp, cg, cd, cp, lmd)


class ConditionalMinibatch(object):

    def __init__(self, tc_md, conditions_md, bandwidths, contrasts):
        self.tc_md = tc_md
        self.conditions_md = np.asarray(conditions_md)
        self.bandwidths = bandwidths
        self.contrasts = contrasts

        assert self.tc_md.shape[:-1] == self.conditions_md.shape[1:]
        assert self.tc_md.shape[-1] == len(bandwidths)

    num_models = property(lambda self: self.tc_md.shape[0])
    probes_per_model = property(lambda self: self.tc_md.shape[1])
    num_bandwidths = property(lambda self: self.tc_md.shape[2])

    @property
    def batchsize(self):
        return self.num_models * self.probes_per_model

    @property
    def gen_kwargs(self):
        contrasts, bandwidths = np.broadcast_arrays(
            self.contrasts.reshape((-1, 1)),
            self.bandwidths.reshape((1, -1)),
        )
        _, norm_probes, cell_types = self._conditions_T
        return dict(
            stimulator_bandwidths=bandwidths.astype(theano.config.floatX),
            stimulator_contrasts=contrasts.astype(theano.config.floatX),
            prober_norm_probes=norm_probes.astype(theano.config.floatX),
            prober_cell_types=cell_types.astype('uint16'),
            prober_model_ids=self.model_ids.astype('uint16'))
    # TODO: Find out why I need .astype(floatX) above.  Without them,
    # I have NaNs in gen_out.prober_tuning_curve.  However, since I'm
    # using allow_input_downcast=True, they should be unnecessarily.
    # It probably occurs when I have int type for contrasts.
    #
    # I didn't need .astype('uint16') for it to run but let's be extra
    # cautious.

    @property
    def tuning_curves(self):
        return self.tc_md.reshape((self.batchsize, self.num_bandwidths))

    @property
    def conditions(self):
        return self._conditions_T.T

    @property
    def _conditions_T(self):
        return self.conditions_md.reshape((-1, self.batchsize))

    @property
    def model_ids(self):
        model_ids = np.arange(self.num_models, dtype='uint16')
        model_ids = model_ids.reshape((-1, 1))
        model_ids = np.broadcast_to(model_ids, self.conditions_md.shape[1:])
        return model_ids.flatten()


class RandomChoiceSampler(object):
    """
    Minibatch sampler based on random choice (not shuffle-based).

    .. TODO: implement shuffle-based sampler

    """

    @classmethod
    def from_grid_data(cls, data, bandwidths, contrasts, norm_probes,
                       include_inhibitory_neurons,
                       **kwargs):
        """
        Initialize `RandomChoiceSampler`.

        Parameters
        ----------
        data : numpy.ndarray
            An array returned by `.subsample_neurons`.
        bandwidths : numpy.ndarray
            See `.ConditionalBPTTWassersteinGAN.bandwidths`.
        contrasts : numpy.ndarray
            See `.ConditionalBPTTWassersteinGAN.contrasts`.
        norm_probes : numpy.ndarray
            Numeric version of `.ConditionalProber.norm_probes`.
        include_inhibitory_neurons : bool
            See `.ConditionalBPTTWassersteinGAN.include_inhibitory_neurons`.

        """
        cell_types = [0, 1] if include_inhibitory_neurons else [0]

        # This is the order of dimensions varied in subsample_neurons:
        cond_values = [contrasts, bandwidths, cell_types, norm_probes]
        shape = (len(data),) + tuple(map(len, cond_values))
        nested = data.reshape(shape)

        # Shuffle the dimension to the order that RandomChoiceSampler
        # understands:
        nested = nested.transpose((0, 3, 4, 1, 2))
        cond_values = [cell_types, norm_probes, contrasts, bandwidths]
        return cls(nested, cond_values, **kwargs)

    def __init__(self, nested, cond_values, e_ratio,
                 seed=0):
        self.nested = np.asarray(nested)
        self.cond_values = cond_values = list(map(np.asarray, cond_values))
        self.e_ratio = e_ratio
        (self.cell_types,
         self.norm_probes,
         self.contrasts,
         self.bandwidths) = cond_values
        self.rng = as_randomstate(seed)

        assert tuple(self.cell_types) in [(0,), (0, 1)]

    def random_cells(self, num_models, probes_per_model):
        """
        Choose cells in such a way that every cell is chosen at most once.
        """
        cellids = cartesian_product(np.arange(len(self.cell_types)),
                                    np.arange(len(self.norm_probes)),
                                    dtype=int).T
        if len(self.cell_types) == 2:
            probs = np.zeros(len(cellids))
            probs[:len(self.norm_probes)] = self.e_ratio
            probs[len(self.norm_probes):] = 1 - self.e_ratio
            probs /= probs.sum()
        else:
            probs = None

        def choice():
            return self.rng.choice(len(cellids), probes_per_model,
                                   replace=False, p=probs)

        # ids.shape: (num_models, probes_per_model, 2)
        ids = np.asarray([cellids[choice()] for _ in range(num_models)])

        ids_cell_type, ids_norm_probes = ids.transpose((2, 0, 1))
        assert ids_cell_type.shape == ids_norm_probes.shape \
            == (num_models, probes_per_model)
        return ids_cell_type, ids_norm_probes

    def select_minibatch(self, num_models, probes_per_model):
        shape = (num_models, probes_per_model)
        ids_sample = self.rng.choice(len(self.nested), shape)
        (ids_cell_type, ids_norm_probes) = self.random_cells(*shape)
        ids_flat_contrast = self.rng.choice(len(self.contrasts),
                                            num_models)
        ids_contrast = ids_flat_contrast.reshape((-1, 1))
        ids_contrast = np.broadcast_to(ids_contrast, shape)
        tc_md = self.nested[
            ids_sample,
            ids_cell_type,
            ids_norm_probes,
            ids_contrast,
        ]
        assert tc_md.shape == (
            num_models, probes_per_model, len(self.bandwidths),
        )
        return ConditionalMinibatch(
            tc_md,
            [self.contrasts[ids_contrast],
             self.norm_probes[ids_norm_probes],
             self.cell_types[ids_cell_type]],
            self.bandwidths,
            self.contrasts[ids_flat_contrast],
        )

    def random_minibatches(self, *args, **kwargs):
        while True:
            yield self.select_minibatch(*args, **kwargs)


class NaiveRandomChoiceSampler(RandomChoiceSampler):
    # TODO: maybe add a way to use this class

    def random_cell_types(self, shape):
        if len(self.cell_types) == 2:
            return self.rng.choice(2, shape,
                                   p=[self.e_ratio, 1 - self.e_ratio])
        else:
            return np.zeros(shape, dtype='uint16')

    def random_cells(self, num_models, probes_per_model):
        shape = (num_models, probes_per_model)
        ids_cell_type = self.random_cell_types(shape)
        ids_norm_probes = self.rng.choice(len(self.norm_probes), shape)
        return ids_cell_type, ids_norm_probes


class ConditionalBPTTWassersteinGAN(BPTTWassersteinGAN):

    def __init__(self, gen, disc, gen_trainer, disc_trainer,
                 bandwidths, contrasts, norm_probes,
                 e_ratio, include_inhibitory_neurons,
                 num_models, probes_per_model,
                 critic_iters_init, critic_iters, lipschitz_cost,
                 seed=0):
        self.gen = gen
        self.disc = disc
        self.gen_trainer = gen_trainer
        self.disc_trainer = disc_trainer

        self.bandwidths = np.asarray(bandwidths)
        self.contrasts = np.asarray(contrasts)
        self.norm_probes = np.asarray(norm_probes)
        self.e_ratio = e_ratio
        self.include_inhibitory_neurons = include_inhibitory_neurons
        self.num_models = num_models
        self.probes_per_model = probes_per_model

        self.critic_iters_init = critic_iters_init
        self.critic_iters = critic_iters
        self.lipschitz_cost = lipschitz_cost
        self.rng = as_randomstate(seed)

        assert gen.batchsize == self.num_models * self.probes_per_model
        assert self.probes_per_model < gen.num_neurons

    num_sites = property(lambda self: self.gen.model.num_sites)

    @property
    def sample_sites(self):
        from ..gradient_expressions.utils import sample_sites_from_stim_space
        return sample_sites_from_stim_space(self.norm_probes,
                                            self.num_sites)

    def set_dataset(self, data, **kwargs):
        kwargs.setdefault('seed', self.rng)
        self.sampler = RandomChoiceSampler.from_grid_data(
            data,
            bandwidths=self.bandwidths,
            contrasts=self.contrasts,
            norm_probes=self.norm_probes,
            e_ratio=self.e_ratio,
            include_inhibitory_neurons=self.include_inhibitory_neurons,
            **kwargs)
        self.dataset = self.sampler.random_minibatches(
            self.num_models, self.probes_per_model,
        )

    def gen_forward(self, batch):
        return self.gen.forward(rng=self.rng, **batch.gen_kwargs)

    def train_discriminator(self, info):
        batch = self.next_minibatch()  # ConditionalMinibatch
        xd = batch.tuning_curves
        cd = batch.conditions
        batchsize = batch.batchsize
        eps = self.rng.rand(batchsize, 1)
        with self.gen_forward_watch:
            gen_out = self.gen_forward(batch)
        xg = gen_out.prober_tuning_curve
        xp = eps * xd + (1 - eps) * xg
        cg = cp = cd
        with self.disc_train_watch:
            info.disc_loss = self.disc_trainer.train(
                xg, xd, xp,
                cg, cd, cp,
                self.lipschitz_cost,
            )

        info.accuracy = self.disc.accuracy(xg, cg, xd, cd)
        info.gen_out = gen_out
        info.dynamics_penalty = gen_out.model_dynamics_penalty
        info.xd = xd
        info.xg = xg
        info.xp = xp
        info.cd = info.cg = info.cp = cd
        info.batch = batch
        info.gen_time = self.gen_forward_watch.times[-1]
        info.disc_time = self.disc_train_watch.times[-1]
        return info

    def train_generator(self, info, batch):
        gen_kwargs = batch.gen_kwargs
        with self.gen_train_watch:
            info.gen_loss = self.gen_trainer.train(rng=self.rng, **gen_kwargs)

        info.gen_forward_time = self.gen_forward_watch.sum()
        info.gen_time = self.gen_train_watch.sum() + info.gen_forward_time
        info.disc_time = self.disc_train_watch.sum()
        return info

    def _single_gen_step(self, gen_step, critic_iters):
        self.gen_forward_watch = StopWatch()
        self.gen_train_watch = StopWatch()
        self.disc_train_watch = StopWatch()

        for disc_step in range(critic_iters):
            info = SimpleNamespace(is_discriminator=True, gen_step=gen_step,
                                   disc_step=disc_step)
            info = self.train_discriminator(info)
            yield info
        batch = info.batch

        info = SimpleNamespace(is_discriminator=False, gen_step=gen_step)
        yield self.train_generator(info, batch)


def make_gan(config):
    """
    Initialize a GAN given `config` and return unconsumed part of `config`.
    """
    return _make_gan_from_kwargs(**dict(DEFAULT_PARAMS, **config))


def _make_gan_from_kwargs(
        J0, S0, D0, num_sites, bandwidths, contrasts,
        num_models, probes_per_model,
        hide_cell_type,
        **rest):
    gen, rest = emit_tuning_curve_generator(
        batchsize=num_models * probes_per_model,
        # Stimulator:
        num_tcdom=len(bandwidths),
        num_sites=num_sites,
        # Model / SSN:
        J=J0,
        D=D0,
        S=S0,
        # Use conditional generator:
        emit_prober=ConditionalProber.consume_kwargs,
        emit_tcg=ConditionalTuningCurveGenerator.consume_kwargs,
        **rest)
    disc, rest = consume_subdict(
        (CellTypeBlindDiscriminator if hide_cell_type else
         ConditionalDiscriminator),
        'disc', rest,
        shape=gen.output_shape,
        cond_shape=gen.cond_shape,
        loss_type='WD',
    )
    gen_trainer, rest = consume_subdict(
        GeneratorTrainer, 'gen', rest,
        gen, disc,
    )
    disc_trainer, rest = consume_subdict(
        ConditionalCriticTrainer, 'disc', rest,
        disc,
    )
    return ConditionalBPTTWassersteinGAN.consume_kwargs(
        gen, disc, gen_trainer, disc_trainer, bandwidths, contrasts,
        num_models=num_models, probes_per_model=probes_per_model,
        **rest)
