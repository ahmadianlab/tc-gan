import numpy as np

from ..gradient_expressions.utils import sample_sites_from_stim_space
from .ssn import make_tuning_curve_generator
from .tests import test_euler_ssn
from .utils import largerrecursionlimit
from .wgan import (
    DEFAULT_PARAMS, grid_stimulator_inputs, probes_from_stim_space,
)

DEFAULT_PARAMS = dict(
    DEFAULT_PARAMS,
    V=[0.3, 0],
    seed=0,
    **test_euler_ssn.JDS
)


class FixedTimeTuningCurveSampler(object):

    @classmethod
    def from_dict(cls, dct):
        self, rest = cls.consume_kwargs(**dict(DEFAULT_PARAMS, **dct))
        assert not rest
        return self

    @classmethod
    def consume_kwargs(cls, bandwidths, contrasts, seed,
                       sample_sites, num_sites, consume_union=True, **kwargs):
        gen, rest = make_tuning_curve_generator(
            kwargs,
            consume_union=consume_union,
            # Stimulator:
            num_tcdom=len(bandwidths) * len(contrasts),
            num_sites=num_sites,
            # Prober:
            probes=sample_sites_from_stim_space(sample_sites, num_sites),
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
        if 'gen' not in override:
            gen_config = learner.gen.to_config()
            gen_config = {k: v for k, v in gen_config.items()
                          if k not in override}
            if 'probes' not in gen_config and 'probes' not in override:
                # Then the learner is cWGAN.
                probes = probes_from_stim_space(
                    learner.norm_probes,   # = sample_sites
                    learner.num_sites,
                    learner.include_inhibitory_neurons)
                gen_config['probes'] = probes
            # Pass gen_config as keyword arguments, to make sure all
            # of them are used up.  Pass `override` as config to
            # retrieve unconsumed ones:
            gen, override = make_tuning_curve_generator(override, **gen_config)
            config['gen'] = gen
        config.update(override)
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
