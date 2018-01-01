import numpy as np

from ..lasagne_toppings.rechack import largerrecursionlimit
from .ssn import make_tuning_curve_generator
from .wgan import (
    DEFAULT_PARAMS, grid_stimulator_inputs, probes_from_stim_space,
)


def _make_new_JDS():
    # Original SSN parameters:
    J = np.array([[.0957, .0638], [.1197, .0479]])
    D = np.array([[.7660, .5106], [.9575, .3830]])
    S = np.array([[.6667, .2], [1.333, .2]]) / 8

    # More stable parameters:
    D_new = D / 2
    J_new = J + D / 2 - D_new / 2
    return dict(J=J_new, D=D_new, S=S)

new_JDS = _make_new_JDS()

DEFAULT_PARAMS = dict(
    DEFAULT_PARAMS,
    V=0.5,  # make it work for both ssn_type='heteroin' and 'deg-heteroin'
    seed=0,
    norm_probes=[0],
    include_inhibitory_neurons=False,
    **new_JDS
)
del DEFAULT_PARAMS['sample_sites']
del DEFAULT_PARAMS['gen']


class FixedTimeTuningCurveSampler(object):

    @classmethod
    def from_dict(cls, dct):
        self, rest = cls.consume_kwargs(**dict(DEFAULT_PARAMS, **dct))
        assert not rest
        return self

    @classmethod
    def consume_kwargs(cls, bandwidths, contrasts, seed,
                       norm_probes, num_sites, include_inhibitory_neurons,
                       consume_union=True, **kwargs):
        gen, rest = make_tuning_curve_generator(
            kwargs,
            consume_union=consume_union,
            # Stimulator:
            num_tcdom=len(bandwidths) * len(contrasts),
            num_sites=num_sites,
            include_rate_penalty=False,
            # Prober:
            probes=probes_from_stim_space(norm_probes, num_sites,
                                          include_inhibitory_neurons),
        )
        return cls(gen, bandwidths, contrasts, seed), rest

    def __init__(self, gen, bandwidths, contrasts, seed):

        bandwidths = np.asarray(bandwidths)
        contrasts = np.asarray(contrasts)
        assert bandwidths.ndim == 1
        assert contrasts.ndim == 1

        self.gen = gen
        self.bandwidths = bandwidths
        self.contrasts = contrasts
        self.rng = np.random.RandomState(seed)

        self.stimulator_contrasts, self.stimulator_bandwidths \
            = grid_stimulator_inputs(contrasts, bandwidths, self.batchsize)

    num_neurons = property(lambda self: self.num_sites * 2)
    num_sites = property(lambda self: self.gen.num_sites)
    batchsize = property(lambda self: self.gen.batchsize)

    def forward(self, full_output=False):
        out = self.gen.forward(
            self.rng,
            stimulator_bandwidths=self.stimulator_bandwidths,
            stimulator_contrasts=self.stimulator_contrasts,
        )
        if full_output:
            return out
        return out.prober_tuning_curve

    def compute_trajectories(self):
        trajectories = self.gen.model.compute_trajectories(
            rng=self.rng,
            stimulator_bandwidths=self.stimulator_bandwidths,
            stimulator_contrasts=self.stimulator_contrasts,
        )
        return trajectories

    def timepoints(self):
        dt = self.gen.model.dt
        seqlen = self.gen.model.seqlen
        ts = np.linspace(dt, dt * seqlen, seqlen)
        return ts

    @property
    def dom_points(self):
        for contrast in self.contrasts:
            for bandwidth in self.bandwidths:
                yield dict(contrast=contrast, bandwidth=bandwidth)

    def prepare(self):
        """ Force compile Theano functions. """
        with largerrecursionlimit(self.gen.model.unroll_scan,
                                  self.gen.model.seqlen):
            self.gen.prepare()

    @classmethod
    def from_learner(cls, learner, **override):
        """
        Initialize FixedTimeTuningCurveSampler based on `learner`.

        Parameters not specified by keyword arguments are copied from
        `learner`.
        """
        config = dict(
            bandwidths=learner.bandwidths,
            contrasts=learner.contrasts,
        )
        config.update(override)
        if 'gen' not in config:
            gen_config = learner.gen.to_config()
            gen_config = {k: v for k, v in gen_config.items()
                          if k not in config}

            # num_tcdom has to be defined this way for cls to work.
            # See also [[def consume_kwargs]]
            bandwidths = config['bandwidths']
            contrasts = config['contrasts']
            gen_config['num_tcdom'] = len(bandwidths) * len(contrasts)

            if 'probes' not in gen_config and 'probes' not in config:
                # Then the learner is cWGAN.
                probes = probes_from_stim_space(
                    learner.norm_probes,   # = sample_sites
                    learner.num_sites,
                    learner.include_inhibitory_neurons)
                gen_config['probes'] = probes

            gen_config['include_rate_penalty'] = False

            # Pass gen_config as keyword arguments, to make sure all
            # of them are used up:
            gen, config = make_tuning_curve_generator(config, **gen_config)

            config['gen'] = gen
        return cls(**config)


def add_arguments(parser, exclude=()):
    from .ssn import ssn_impl_choices, ssn_type_choices, HeteroInputWrapper

    parser.add_argument(
        '--ssn-impl', default=ssn_impl_choices[0], choices=ssn_impl_choices,
        help="SSN implementation.")
    parser.add_argument(
        '--ssn-type', default=ssn_type_choices[0], choices=ssn_type_choices,
        help="SSN type.")
    parser.add_argument(
        '--dist-in',
        default=HeteroInputWrapper.dist_in_choices[0],
        choices=HeteroInputWrapper.dist_in_choices,
        help='''Input heterogeneity distribution type. Relevant only
        when --ssn-type=heteroin''')

    for key in sorted(DEFAULT_PARAMS):
        if key in exclude:
            continue

        val = DEFAULT_PARAMS[key]
        if isinstance(val, (str, float, int)):
            argtype = type(val)
        else:
            argtype = eval
        parser.add_argument(
            '--{}'.format(key.replace('_', '-')),
            type=argtype, default=val,
            help='SSN parameter')
